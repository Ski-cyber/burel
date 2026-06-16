# Re-export the public model API so callers can `from burel.model import ...`.
# Top-level model and its parameter-counting helper.
from .hope import BurelLM, count_parameters
# Transformer encoder building blocks.
from .layers import AttentionPooling, Encoder, EncoderLayer, PositionalEncoding
# Nested-learning memory components (continuum memory, deep optimizer, memory module).
from .memory import ContinuumMemorySystem, DeepOptimizer, FunctionalMemoryModule

__all__ = [
    "BurelLM", "count_parameters",
    "Encoder", "EncoderLayer", "PositionalEncoding", "AttentionPooling",
    "ContinuumMemorySystem", "DeepOptimizer", "FunctionalMemoryModule",
]
