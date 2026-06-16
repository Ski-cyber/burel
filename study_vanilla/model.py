# study_vanilla/model.py
#
# VanillaGPT — a textbook decoder-only Transformer (GPT-style), the BASELINE for the
# nested-vs-vanilla A/B. Pre-norm blocks, full causal self-attention, learned position
# embeddings, weight tying. No memory, no test-time training: a frozen model that adapts
# to context ONLY through attention. This is exactly what BurelLM's memory must beat.
#
# Parameter parity (the whole point of a fair A/B):
#   d_model=384, n_layer=8, d_ff=1536, n_head=6, context=256, vocab=16000, tied
#   -> ~20.43M params, vs BurelLM ~20.17M (within ~1.3%). Tune n_layer for exact parity.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention over the FULL window (no chunking).

    Uses F.scaled_dot_product_attention with is_causal=True. Unlike Burel, the vanilla
    baseline has no second-order gradients, so the fast flash/mem-efficient kernels are
    fair game — part of the vanilla's legitimate efficiency advantage."""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0, "d_model deve essere divisibile per n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)      # fused query/key/value projection
        self.proj = nn.Linear(d_model, d_model)         # output projection
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, d_head]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: d_model -> d_ff -> d_model."""

    def __init__(self, d_model, d_ff, dropout, activation="gelu"):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff)
        self.proj = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = F.gelu if activation == "gelu" else F.silu

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """Pre-norm Transformer block: x + attn(ln(x)); x + mlp(ln(x))."""

    def __init__(self, d_model, n_head, d_ff, dropout, activation):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout, activation)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VanillaGPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_head=6, n_layer=8, d_ff=1536,
                 context=256, dropout=0.2, activation="gelu", tie_weights=True):
        super().__init__()
        self.context = context                 # max sequence length (matches Burel's max_memory_length)
        self.n_layer = n_layer

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff, dropout, activation) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.tok_emb.weight   # share embedding with output projection

        self.apply(self._init_weights)
        # GPT-2 scaled init for residual projections: keeps activation variance stable in deep stacks.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx: [B, T] long. Returns (logits [B, T, vocab], loss|None)."""
        B, T = idx.shape
        assert T <= self.context, f"sequenza {T} > context {self.context}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context:]
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


def build_from_config(mc, vocab_size, device):
    """Instantiate VanillaGPT from a model-config dict (mirrors the trainer/loader)."""
    return VanillaGPT(
        vocab_size=vocab_size,
        d_model=mc["d_model"], n_head=mc["n_head"], n_layer=mc["n_layer"],
        d_ff=mc["d_ff"], context=mc["context"], dropout=mc.get("dropout", 0.0),
        activation=mc.get("activation", "gelu"), tie_weights=mc.get("tie_weights", True),
    ).to(device)
