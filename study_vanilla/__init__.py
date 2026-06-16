# study_vanilla — a standalone vanilla Transformer baseline, kept SEPARATE from Burel.
#
# Purpose: the controlled A/B for the thesis "nested memory buys intelligence-per-param".
# Same vocabulary, same data, same context, ~same parameter count as BurelLM, but a
# plain decoder-only Transformer (full causal attention, no Continuum Memory System,
# no Test-Time Training). The ONLY variable is the architecture: attention vs memory.
#
# This package does NOT import `burel`: it reads the same data_cache directly so it can
# be published and run on its own. The three-way comparison lives in scripts/compare_three.py.
