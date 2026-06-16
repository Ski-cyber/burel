# burel/model/memory.py
#
# The heart of NESTED LEARNING (Titans / HOPE), ported faithfully from
# ModelMango/model/timegpt.py and NOT altered:
#   - FunctionalMemoryModule  : fast weights (stateless memory, "functional" params)
#   - DeepOptimizer           : learned optimizer (predicts LR / momentum / decay)
#   - ContinuumMemorySystem   : multi-level memory with Test-Time Training (2nd-order grads)
#
# Big picture
# -----------
# A vanilla Transformer learns ONCE: you train it, freeze the weights, and at
# inference it no longer adapts. Nested Learning adds a memory whose weights keep
# being updated WHILE the model reads/generates (Test-Time Training). The memory is
# itself a small neural net (the "fast weights"); a tiny learned optimizer decides
# HOW to update it; and several copies of this memory run at different speeds. Hence
# "nested": an optimization loop (the memory update) lives inside the forward pass of
# the outer model, and that update is itself learned.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#   Functional Memory Module (fast weights, stateless)
# =============================================================================

# A plain MLP whose weights are passed in EXPLICITLY at call time instead of being
# the usual nn.Parameter state. This is what lets the outer system keep many evolving
# copies of the same memory: the module holds only the INITIAL weights; the current
# (fast) weights live outside and are fed via `current_weights`/`current_biases`.
class FunctionalMemoryModule(nn.Module):
    def __init__(self, d_model, memory_depth=2, use_silu=True):
        super().__init__()
        self.d_model = d_model
        self.memory_depth = memory_depth        # number of linear layers in the MLP
        self.use_silu = use_silu

        # Initial weights/biases: the starting point of the memory at chunk 0. These
        # ARE learnable (slow weights, trained by backprop); the fast weights derived
        # from them during Test-Time Training are NOT stored here.
        self.initial_weights = nn.ParameterList()
        self.initial_biases = nn.ParameterList()
        for _ in range(memory_depth):
            w = nn.Parameter(torch.empty(d_model, d_model))
            b = nn.Parameter(torch.empty(d_model))
            # Standard nn.Linear initialization (Kaiming uniform + matching bias bound).
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)
            self.initial_weights.append(w)
            self.initial_biases.append(b)

    def forward(self, x, current_weights=None, current_biases=None):
        # If no fast weights are supplied, fall back to the initial (slow) ones.
        weights = current_weights if current_weights is not None else self.initial_weights
        biases = current_biases if current_biases is not None else self.initial_biases
        out = x
        # Manual MLP via F.linear so the weights can be arbitrary tensors (the fast
        # weights), with a nonlinearity between layers but not after the last one.
        for i in range(self.memory_depth):
            out = F.linear(out, weights[i], biases[i])
            if i < self.memory_depth - 1:
                out = F.silu(out) if self.use_silu else F.relu(out)
        return out

    def get_initial_params(self):
        # Returns the slow weights as plain lists, used as the fast-weight seed at the
        # first chunk before any Test-Time Training update has happened.
        return list(self.initial_weights), list(self.initial_biases)


# =============================================================================
#   Deep Optimizer (Nested Learning component)
# =============================================================================

class DeepOptimizer(nn.Module):
    """Predicts the memory update's LR/momentum/decay instead of using fixed values."""

    def __init__(self, d_model, base_lr, use_silu=True):
        super().__init__()
        self.base_log_lr = math.log(base_lr)        # learning rate is predicted in log space
        # Tiny network: from a context vector it outputs 3 numbers that parameterize
        # the update rule applied to the fast weights.
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU() if use_silu else nn.ReLU(),
            nn.Linear(d_model // 4, 3),  # delta_log_lr, logit_eta (momentum), logit_alpha (decay)
        )
        # Start as a near-identity / well-behaved optimizer: zero weights so the first
        # outputs come purely from the bias -> momentum ~0.9, decay ~0.01, LR = base_lr.
        with torch.no_grad():
            self.net[-1].weight.fill_(0.0)
            self.net[-1].bias.data = torch.tensor([0.0, 2.2, -4.6])  # momentum ~0.9, decay ~0.01

    def forward(self, context_key):
        # Average over the batch so the optimizer hyper-params are shared per step.
        params = self.net(context_key).mean(dim=0)
        delta_lr, logit_eta, logit_alpha = params[0], params[1], params[2]
        # Bounded range to avoid the inner LR exploding (which would produce NaNs in the
        # 2nd-order gradients of Test-Time Training).
        delta_lr = torch.tanh(delta_lr) * 1.0
        current_lr = torch.exp(self.base_log_lr + delta_lr)     # LR in (base/e, base*e)
        current_eta = torch.sigmoid(logit_eta)                  # momentum coefficient in (0, 1)
        current_alpha = torch.sigmoid(logit_alpha)              # decay coefficient in (0, 1)
        return current_lr, current_eta, current_alpha


# =============================================================================
#   Continuum Memory System (the Nested Learner)
# =============================================================================

# Stacks several FunctionalMemoryModules ("levels") that update at exponentially
# different frequencies, each driven by its own learned DeepOptimizer. Reading the
# memory (forward) is a residual pass through all levels; writing it (update) is the
# Test-Time Training step that adapts the fast weights to the current context.
class ContinuumMemorySystem(nn.Module):
    def __init__(self, d_model, num_levels=3, memory_depth=2, mem_lr=1e-4, use_silu=True):
        super().__init__()
        self.num_levels = num_levels
        self.d_model = d_model
        # Exponential frequencies: level 0 updates every chunk (fastest, short-term),
        # level N every 2^N chunks (slowest, long-term). This is the "continuum".
        self.update_frequencies = [2 ** i for i in range(num_levels)]

        # One fast-weight memory per level...
        self.memory_levels = nn.ModuleList(
            [FunctionalMemoryModule(d_model, memory_depth, use_silu) for _ in range(num_levels)]
        )
        self.mem_criterion = nn.MSELoss()   # memory learns to map key -> value (associative)
        # ...one learned optimizer per level...
        self.optimizers = nn.ModuleList(
            [DeepOptimizer(d_model, mem_lr, use_silu) for _ in range(num_levels)]
        )
        # ...a forget gate (extra, context-dependent weight decay) and a norm per level.
        self.forget_gates = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_levels)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_levels)])

        # Per-sequence mutable state (NOT nn.Parameters): the current fast weights and
        # the momentum buffers. Reset at the start of every forward (see reset_memory_state).
        self.active_params = {}
        self.momentum_buffer = {}

    def reset_memory_state(self):
        # Clear the fast weights and momentum so each forward starts from the slow
        # (learned initial) weights -> sequences in a batch don't leak memory into each other.
        self.active_params = {}
        self.momentum_buffer = {}

    def get_current_params(self, level_idx):
        # Current fast weights for a level, or the initial slow weights if not yet updated.
        if level_idx in self.active_params:
            return self.active_params[level_idx]
        return self.memory_levels[level_idx].get_initial_params()

    def forward(self, x):
        # READ path: push the query through every level as a residual stack, so each
        # level refines the retrieval of the one before it.
        current_input = x
        for i in range(self.num_levels):
            curr_w, curr_b = self.get_current_params(i)
            out = self.memory_levels[i](current_input, curr_w, curr_b)
            current_input = self.norms[i](current_input + out)
        return current_input

    def update(self, key, value, global_chunk_idx, update_in_inference=True):
        """Meta-learning step (Test-Time Training): minimize ||M(key) - value||."""
        # WRITE path. For each level due to update at this chunk index, take ONE gradient
        # step on the fast weights so the memory better maps key->value. The step uses
        # create_graph=True (2nd-order) so the SLOW weights and the learned optimizer can
        # be trained through this inner update by the outer backprop.
        if not update_in_inference:
            return
        with torch.enable_grad():
            for i in range(self.num_levels):
                # Skip levels not scheduled at this chunk (slow levels update rarely).
                if global_chunk_idx % self.update_frequencies[i] != 0:
                    continue

                # Learned hyper-parameters for this level's update.
                current_lr, current_eta, current_decay = self.optimizers[i](key)
                module = self.memory_levels[i]
                curr_w, curr_b = self.get_current_params(i)

                # Inner loss: how well the current memory associates key -> value.
                pred = module(key, curr_w, curr_b)
                loss = self.mem_criterion(pred, value)

                # Gradient of the inner loss wrt the fast weights. create_graph=True keeps
                # this differentiable so the outer training can shape the memory dynamics.
                params_to_diff = curr_w + curr_b
                grads = torch.autograd.grad(loss, params_to_diff, create_graph=True)
                num_w = len(curr_w)
                grads_w, grads_b = grads[:num_w], grads[num_w:]

                # Context-dependent forgetting: shrink old weights a bit before the step.
                forget_factor = torch.sigmoid(self.forget_gates[i](key)).mean()
                decay = 1.0 - (current_decay * forget_factor)

                # Lazily allocate momentum buffers for this level.
                if i not in self.momentum_buffer:
                    self.momentum_buffer[i] = {
                        "w": [torch.zeros_like(p) for p in curr_w],
                        "b": [torch.zeros_like(p) for p in curr_b],
                    }
                mom_buf = self.momentum_buffer[i]

                # Momentum-SGD-with-decay update of the fast weights, with predicted LR,
                # predicted momentum (eta) and predicted/forget-gated decay.
                new_weights, new_biases = [], []
                for idx, (w, g) in enumerate(zip(curr_w, grads_w)):
                    if w.shape != g.shape:
                        raise RuntimeError(f"grad shape mismatch lvl {i} w[{idx}]: {w.shape} vs {g.shape}")
                    m_new = current_eta * mom_buf["w"][idx] + (1.0 - current_eta) * g
                    mom_buf["w"][idx] = m_new
                    new_weights.append(w * decay - current_lr * m_new)

                for idx, (b, g) in enumerate(zip(curr_b, grads_b)):
                    if b.shape != g.shape:
                        raise RuntimeError(f"grad shape mismatch lvl {i} b[{idx}]: {b.shape} vs {g.shape}")
                    m_new = current_eta * mom_buf["b"][idx] + (1.0 - current_eta) * g
                    mom_buf["b"][idx] = m_new
                    new_biases.append(b * decay - current_lr * m_new)

                # Commit the new fast weights; they will be used (read) by later chunks.
                self.active_params[i] = (new_weights, new_biases)
