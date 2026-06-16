# burel/model/layers.py
#
# Basic Transformer building blocks: an encoder with causal masking, sinusoidal
# positional encoding, and attention pooling. No nested-learning logic lives here
# (see memory.py); these are the standard, well-understood pieces.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#   Encoder (with intra-chunk causal masking)
# =============================================================================

# A single post-norm Transformer encoder block: self-attention + feed-forward, each
# wrapped in a residual connection and a LayerNorm. Used to encode one chunk.
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, use_silu=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.silu if use_silu else F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention sublayer. src_mask carries the causal mask; is_causal=False
        # because the mask is passed explicitly rather than inferred.
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=False,
        )
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        # Position-wise feed-forward sublayer.
        ff_output = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        return src


# Stack of EncoderLayers applied in sequence; mask is threaded through unchanged.
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1, use_silu=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dim_feedforward, dropout, use_silu) for _ in range(num_layers)]
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return src


# =============================================================================
#   Positional Encoding (intra-chunk)
# =============================================================================

# Classic fixed sinusoidal positional encoding (Vaswani et al.). Added to the chunk
# embeddings to encode position WITHIN a chunk; precomputed once as a buffer.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Geometric series of frequencies; even dims use sin, odd dims use cos.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # Add the positional encodings for the first x.size(1) positions.
        return x + self.pe[:, :x.size(1), :]


# =============================================================================
#   Attention Pooling (chunk compression)
# =============================================================================

# Compresses a [B, S, D] chunk into a single [B, D] vector by a learned, softmax-
# weighted average over the sequence positions (a 1-head additive attention).
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x, padding_mask=None):
        scores = self.attention_weights(x)  # [B, S, 1]
        # Mask out padded positions so they get zero weight after softmax.
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(scores, dim=1)
        # If a whole row was masked (all -inf), softmax yields NaN -> replace with 0.
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.sum(weights * x, dim=1)  # [B, D]
