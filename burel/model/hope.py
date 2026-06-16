# burel/model/hope.py
#
# BurelLM — Transformer + Continuum Memory System (MAC architecture, "Memory as
# Context") for autoregressive language modeling. Strictly causal.
#   - input  : embedding of discrete tokens
#   - output : LM head -> logits over the vocabulary
#   - loss   : next-token cross-entropy
#   - memory : causal retrieval (query from the previous chunk) + Test-Time Training
#
# How a sequence is processed
# ---------------------------
# The sequence is split into fixed-size chunks. Each chunk is encoded by a causal
# Transformer encoder, then enriched with a "memory context" vector u_C retrieved from
# the Continuum Memory System (CMS). Crucially, u_C is retrieved using a query built
# from the PREVIOUS chunk, so a chunk never reads memory written from its own (future)
# tokens -> strict causality. After processing, the chunk writes a new (key, value)
# into the CMS via Test-Time Training, which only later chunks can read.

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
        self.persistent_length = persistent_length      # learned "always-visible" memory slots
        self.max_memory_length = max_memory_length       # = max context the model attends to
        self.activation_fn = F.silu if use_silu else F.relu

        # --- INPUT: discrete tokens ---
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional info is twofold: sinusoidal positions WITHIN a chunk, plus a learned
        # embedding for WHICH chunk we are in (global position along the sequence).
        self.pos_encoder = PositionalEncoding(d_model, max_len=chunk_size + 20)
        max_num_chunks = (max_memory_length // chunk_size) + 10
        self.global_chunk_pos_embed = nn.Embedding(max_num_chunks, d_model)

        # Causal encoder over the chunk, then attention pooling to compress the chunk to
        # a single vector (used as the memory key/value and the next chunk's query).
        self.encoder = Encoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout, use_silu)
        self.chunk_pooling = AttentionPooling(d_model)
        # The "augmented" attention layer that mixes [persistent | memory u_C | chunk].
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation=F.silu if use_silu else F.relu, batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_gate = nn.Linear(d_model, d_model)

        # --- OUTPUT: logits over the vocabulary ---
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            # Weight tying: share the embedding matrix with the output projection.
            self.lm_head.weight = self.token_embedding.weight

        # Persistent memory: learned slots prepended to every chunk's attention input,
        # always visible (a small, task-level scratchpad independent of the input).
        self.persistent_memory = nn.Parameter(torch.randn(1, persistent_length, d_model))
        # Retrieval query for the FIRST chunk (no previous chunk to derive it from).
        self.init_query = nn.Parameter(torch.zeros(1, d_model))
        # Running chunk counter, used by the CMS to schedule which memory levels update.
        self.register_buffer("global_chunk_idx", torch.tensor(0, dtype=torch.long))

        # The Continuum Memory System: the nested-learning memory (see memory.py).
        self.cms = ContinuumMemorySystem(
            d_model, num_levels=num_mem_levels, memory_depth=memory_depth,
            mem_lr=mem_lr, use_silu=use_silu,
        )
        # Projections that build the memory query/key/value from chunk representations.
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_dropout = nn.Dropout(0.1)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.update_memory_in_inference = True

        self._init_backbone_weights()

    def _init_backbone_weights(self):
        """GPT-style init (std 0.02) on the backbone. Does NOT touch the CMS, which has
        its own initializations (Kaiming on the fast weights, calibrated biases in the
        DeepOptimizers): overwriting them would break the nested learning."""
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
        # L2-normalize the memory query/key/value for stable associative retrieval.
        return x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8)

    def reset_state(self):
        # Fresh memory for every forward: zero the chunk counter and clear the CMS state.
        self.global_chunk_idx.zero_()
        self.cms.reset_memory_state()

    def _augmented_causal_mask(self, device):
        """Mask for attn_layer over [persistent | u_C | chunk]. True = blocked.

        The chunk tokens see: all of the memory prefix + themselves causally.
        The prefix (persistent + u_C) does NOT see the chunk. This prevents a position
        i from attending to future tokens j>i inside the same chunk.
        """
        prefix = self.persistent_length + 1                 # persistent slots + 1 memory vector u_C
        s = prefix + self.chunk_size
        mask = torch.zeros(s, s, dtype=torch.bool, device=device)
        # Standard causal (upper-triangular) mask among the chunk's own tokens.
        chunk_causal = torch.triu(torch.ones(self.chunk_size, self.chunk_size, device=device), diagonal=1).bool()
        mask[prefix:, prefix:] = chunk_causal
        mask[:prefix, prefix:] = True                       # prefix cannot look into the chunk
        return mask

    def forward(self, idx, targets=None):
        """idx: [B, T] long. Returns (logits [B, T, vocab], loss|None).

        Fresh memory on every forward: the batch sequences are independent and
        train / eval / generate share the same behavior.
        """
        self.reset_state()
        device = idx.device
        batch_size, src_seq_len = idx.size()
        src_proj = self.token_embedding(idx)  # [B, T, D]

        # Number of chunks (ceil division), and the buffer collecting each chunk's output.
        num_chunks = (src_seq_len + self.chunk_size - 1) // self.chunk_size
        processed_chunks_repr = []

        # Precompute per-chunk global position embeddings and the augmented mask once.
        all_chunk_indices = torch.arange(num_chunks, device=device)
        all_global_pos = self.global_chunk_pos_embed(all_chunk_indices)
        aug_mask = self._augmented_causal_mask(device)
        prev_repr = None  # representation of the previous chunk (for causal retrieval)

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, src_seq_len)
            chunk_proj = src_proj[:, start:end, :]
            current_chunk_len = chunk_proj.size(1)

            # Pad the last (possibly short) chunk up to chunk_size and mark the pad with
            # a key-padding mask so attention ignores it.
            src_key_padding_mask = torch.zeros(batch_size, self.chunk_size, dtype=torch.bool, device=device)
            if current_chunk_len < self.chunk_size:
                pad_len = self.chunk_size - current_chunk_len
                pad = torch.zeros(batch_size, pad_len, self.d_model, device=device)
                chunk_proj = torch.cat([chunk_proj, pad], dim=1)
                src_key_padding_mask[:, current_chunk_len:] = True

            # Causal mask within the chunk for the encoder self-attention.
            causal_mask = torch.triu(
                torch.ones(self.chunk_size, self.chunk_size, device=device), diagonal=1
            ).bool()

            # Encode the chunk (intra-chunk positions + chunk-level position).
            chunk_proj = self.pos_encoder(chunk_proj)
            encoded_chunk = self.encoder(chunk_proj, src_mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
            encoded_chunk = encoded_chunk + all_global_pos[i].unsqueeze(0).unsqueeze(0)

            # --- memory retrieval (CMS) — STRICTLY CAUSAL ---
            # The query comes from the PREVIOUS chunk (or init_query for the first one):
            # the context u_C injected into the current chunk depends only on past tokens.
            if prev_repr is None:
                query_basis = self.init_query.expand(batch_size, -1)
            else:
                query_basis = prev_repr
            query = self._normalize_vector(self.query_proj(query_basis))
            u_C = self.cms(query).unsqueeze(1)              # [B, 1, D] memory-as-context vector

            # --- augmented attention (MAC), causal ---
            # Concatenate [persistent slots | memory u_C | encoded chunk] and self-attend.
            persistent = self.persistent_memory.expand(batch_size, -1, -1)
            augmented_seq = torch.cat([persistent, u_C, encoded_chunk], dim=1)
            attn_output = self.attn_layer(augmented_seq, src_mask=aug_mask)
            chunk_attn_output = attn_output[:, -self.chunk_size:, :]    # keep only the chunk tokens
            processed_chunks_repr.append(chunk_attn_output)

            # --- memory update (Test-Time Training) ---
            # key/value come from the CURRENT chunk: it feeds the memory read by LATER chunks.
            chunk_attn_repr = self.chunk_pooling(chunk_attn_output, padding_mask=src_key_padding_mask)
            key = self._normalize_vector(self.key_proj(self.key_dropout(chunk_attn_repr)))
            value = self._normalize_vector(self.value_proj(chunk_attn_repr))
            self.cms.update(key, value, self.global_chunk_idx.item(), self.update_memory_in_inference)
            self.global_chunk_idx += 1
            prev_repr = chunk_attn_repr  # for the next chunk's retrieval

        # Concatenate all chunk outputs, trim padding back to the true length, gate, project.
        full_sequence_repr = torch.cat(processed_chunks_repr, dim=1)[:, :src_seq_len, :]
        gated = self.activation_fn(self.output_gate(self.output_norm(full_sequence_repr)))
        logits = self.lm_head(gated)  # [B, T, vocab]

        loss = None
        if targets is not None:
            # Standard next-token cross-entropy, flattened over batch and time.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation. idx: [B, T0] long (prompt)."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            # Crop the context to the model's max memory length, run a forward, and sample
            # the next token from the last position's logits.
            idx_cond = idx[:, -self.max_memory_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)      # temperature scaling
            if top_k is not None:
                # Top-k filtering: keep only the k most likely tokens.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        if was_training:
            self.train()
        return idx


def count_parameters(model):
    # Number of trainable parameters (used to report model size).
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
