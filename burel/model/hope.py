# burel/model/hope.py
#
# BurelLM — Transformer + Continuum Memory System (architettura MAC, "Memory as
# Context") per language modeling autoregressivo. Strettamente causale.
#   - input  : Embedding di token discreti
#   - output : LM head -> logit sul vocabolario
#   - loss   : cross-entropy next-token
#   - memoria: retrieval causale (query dal chunk precedente) + Test-Time Training

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AttentionPooling, Encoder, PositionalEncoding
from .memory import ContinuumMemorySystem


class BurelLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=4,
                 dim_feedforward=1024, dropout=0.1, persistent_length=4,
                 max_memory_length=256, chunk_size=16, mem_lr=1e-4,
                 memory_depth=2, use_silu=True, num_mem_levels=3, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.persistent_length = persistent_length
        self.max_memory_length = max_memory_length
        self.activation_fn = F.silu if use_silu else F.relu

        # --- INPUT: token discreti ---
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=chunk_size + 20)
        max_num_chunks = (max_memory_length // chunk_size) + 10
        self.global_chunk_pos_embed = nn.Embedding(max_num_chunks, d_model)

        self.encoder = Encoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout, use_silu)
        self.chunk_pooling = AttentionPooling(d_model)
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation=F.silu if use_silu else F.relu, batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_gate = nn.Linear(d_model, d_model)

        # --- OUTPUT: logit sul vocabolario ---
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.persistent_memory = nn.Parameter(torch.randn(1, persistent_length, d_model))
        # query di retrieval per il primo chunk (nessun chunk precedente da cui derivarla)
        self.init_query = nn.Parameter(torch.zeros(1, d_model))
        self.register_buffer("global_chunk_idx", torch.tensor(0, dtype=torch.long))

        self.cms = ContinuumMemorySystem(
            d_model, num_levels=num_mem_levels, memory_depth=memory_depth,
            mem_lr=mem_lr, use_silu=use_silu,
        )
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_dropout = nn.Dropout(0.1)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.update_memory_in_inference = True

        self._init_backbone_weights()

    def _init_backbone_weights(self):
        """Init GPT-style (std 0.02) sul backbone. NON tocca la CMS, che ha
        inizializzazioni proprie (kaiming sui fast weights, bias calibrati nei
        DeepOptimizer): sovrascriverle romperebbe il nested learning."""
        for name, module in self.named_modules():
            if name.startswith("cms"):
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.persistent_memory, mean=0.0, std=0.02)

    def _normalize_vector(self, x):
        return x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8)

    def reset_state(self):
        self.global_chunk_idx.zero_()
        self.cms.reset_memory_state()

    def _augmented_causal_mask(self, device):
        """Mask per attn_layer su [persistent | u_C | chunk]. True = bloccato.

        I token del chunk vedono: tutta la memoria-prefisso + se stessi causalmente.
        Il prefisso (persistent + u_C) non vede il chunk. Questo impedisce a una
        posizione i di attendere token futuri j>i dentro lo stesso chunk.
        """
        prefix = self.persistent_length + 1
        s = prefix + self.chunk_size
        mask = torch.zeros(s, s, dtype=torch.bool, device=device)
        chunk_causal = torch.triu(torch.ones(self.chunk_size, self.chunk_size, device=device), diagonal=1).bool()
        mask[prefix:, prefix:] = chunk_causal
        mask[:prefix, prefix:] = True
        return mask

    def forward(self, idx, targets=None):
        """idx: [B, T] long. Ritorna (logits [B, T, vocab], loss|None).

        Memoria fresca a ogni forward: le sequenze del batch sono indipendenti e
        train / eval / generate condividono lo stesso comportamento.
        """
        self.reset_state()
        device = idx.device
        batch_size, src_seq_len = idx.size()
        src_proj = self.token_embedding(idx)  # [B, T, D]

        num_chunks = (src_seq_len + self.chunk_size - 1) // self.chunk_size
        processed_chunks_repr = []

        all_chunk_indices = torch.arange(num_chunks, device=device)
        all_global_pos = self.global_chunk_pos_embed(all_chunk_indices)
        aug_mask = self._augmented_causal_mask(device)
        prev_repr = None  # rappresentazione del chunk precedente (per il retrieval causale)

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, src_seq_len)
            chunk_proj = src_proj[:, start:end, :]
            current_chunk_len = chunk_proj.size(1)

            src_key_padding_mask = torch.zeros(batch_size, self.chunk_size, dtype=torch.bool, device=device)
            if current_chunk_len < self.chunk_size:
                pad_len = self.chunk_size - current_chunk_len
                pad = torch.zeros(batch_size, pad_len, self.d_model, device=device)
                chunk_proj = torch.cat([chunk_proj, pad], dim=1)
                src_key_padding_mask[:, current_chunk_len:] = True

            causal_mask = torch.triu(
                torch.ones(self.chunk_size, self.chunk_size, device=device), diagonal=1
            ).bool()

            chunk_proj = self.pos_encoder(chunk_proj)
            encoded_chunk = self.encoder(chunk_proj, src_mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
            encoded_chunk = encoded_chunk + all_global_pos[i].unsqueeze(0).unsqueeze(0)

            # --- retrieval dalla memoria (CMS) — STRETTAMENTE CAUSALE ---
            # La query nasce dal chunk PRECEDENTE (o da init_query per il primo): il
            # contesto u_C iniettato nel chunk corrente dipende solo dai token gia' visti.
            if prev_repr is None:
                query_basis = self.init_query.expand(batch_size, -1)
            else:
                query_basis = prev_repr
            query = self._normalize_vector(self.query_proj(query_basis))
            u_C = self.cms(query).unsqueeze(1)

            # --- attention aumentata (MAC), causale ---
            persistent = self.persistent_memory.expand(batch_size, -1, -1)
            augmented_seq = torch.cat([persistent, u_C, encoded_chunk], dim=1)
            attn_output = self.attn_layer(augmented_seq, src_mask=aug_mask)
            chunk_attn_output = attn_output[:, -self.chunk_size:, :]
            processed_chunks_repr.append(chunk_attn_output)

            # --- update memoria (Test-Time Training) ---
            # key/value dal chunk corrente: alimenta la memoria letta dai chunk SUCCESSIVI.
            chunk_attn_repr = self.chunk_pooling(chunk_attn_output, padding_mask=src_key_padding_mask)
            key = self._normalize_vector(self.key_proj(self.key_dropout(chunk_attn_repr)))
            value = self._normalize_vector(self.value_proj(chunk_attn_repr))
            self.cms.update(key, value, self.global_chunk_idx.item(), self.update_memory_in_inference)
            self.global_chunk_idx += 1
            prev_repr = chunk_attn_repr  # per il retrieval del prossimo chunk

        full_sequence_repr = torch.cat(processed_chunks_repr, dim=1)[:, :src_seq_len, :]
        gated = self.activation_fn(self.output_gate(self.output_norm(full_sequence_repr)))
        logits = self.lm_head(gated)  # [B, T, vocab]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generazione autoregressiva. idx: [B, T0] long (prompt)."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_memory_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        if was_training:
            self.train()
        return idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
