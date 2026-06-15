# burel/model/memory.py
#
# Il cuore del NESTED LEARNING (Titans / HOPE), portato fedelmente da
# ModelMango/model/timegpt.py e NON alterato:
#   - FunctionalMemoryModule  : fast weights (memoria stateless, parametri "funzionali")
#   - DeepOptimizer           : optimizer appreso (predice LR/momentum/decay)
#   - ContinuumMemorySystem   : memoria multi-livello con Test-Time Training (grad 2 ordine)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#   Functional Memory Module (fast weights, stateless)
# =============================================================================

class FunctionalMemoryModule(nn.Module):
    def __init__(self, d_model, memory_depth=2, use_silu=True):
        super().__init__()
        self.d_model = d_model
        self.memory_depth = memory_depth
        self.use_silu = use_silu

        self.initial_weights = nn.ParameterList()
        self.initial_biases = nn.ParameterList()
        for _ in range(memory_depth):
            w = nn.Parameter(torch.empty(d_model, d_model))
            b = nn.Parameter(torch.empty(d_model))
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)
            self.initial_weights.append(w)
            self.initial_biases.append(b)

    def forward(self, x, current_weights=None, current_biases=None):
        weights = current_weights if current_weights is not None else self.initial_weights
        biases = current_biases if current_biases is not None else self.initial_biases
        out = x
        for i in range(self.memory_depth):
            out = F.linear(out, weights[i], biases[i])
            if i < self.memory_depth - 1:
                out = F.silu(out) if self.use_silu else F.relu(out)
        return out

    def get_initial_params(self):
        return list(self.initial_weights), list(self.initial_biases)


# =============================================================================
#   Deep Optimizer (componente Nested Learning)
# =============================================================================

class DeepOptimizer(nn.Module):
    """Predice LR/momentum/decay dell'update di memoria invece di usarli fissi."""

    def __init__(self, d_model, base_lr, use_silu=True):
        super().__init__()
        self.base_log_lr = math.log(base_lr)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU() if use_silu else nn.ReLU(),
            nn.Linear(d_model // 4, 3),  # delta_log_lr, logit_eta (momentum), logit_alpha (decay)
        )
        with torch.no_grad():
            self.net[-1].weight.fill_(0.0)
            self.net[-1].bias.data = torch.tensor([0.0, 2.2, -4.6])  # momentum ~0.9, decay ~0.01

    def forward(self, context_key):
        params = self.net(context_key).mean(dim=0)
        delta_lr, logit_eta, logit_alpha = params[0], params[1], params[2]
        # range contenuto per evitare esplosione del LR interno (NaN nei grad di 2 ordine)
        delta_lr = torch.tanh(delta_lr) * 1.0
        current_lr = torch.exp(self.base_log_lr + delta_lr)
        current_eta = torch.sigmoid(logit_eta)
        current_alpha = torch.sigmoid(logit_alpha)
        return current_lr, current_eta, current_alpha


# =============================================================================
#   Continuum Memory System (il Nested Learner)
# =============================================================================

class ContinuumMemorySystem(nn.Module):
    def __init__(self, d_model, num_levels=3, memory_depth=2, mem_lr=1e-4, use_silu=True):
        super().__init__()
        self.num_levels = num_levels
        self.d_model = d_model
        # frequenze esponenziali: livello 0 = fastest, livello N = slowest
        self.update_frequencies = [2 ** i for i in range(num_levels)]

        self.memory_levels = nn.ModuleList(
            [FunctionalMemoryModule(d_model, memory_depth, use_silu) for _ in range(num_levels)]
        )
        self.mem_criterion = nn.MSELoss()
        self.optimizers = nn.ModuleList(
            [DeepOptimizer(d_model, mem_lr, use_silu) for _ in range(num_levels)]
        )
        self.forget_gates = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_levels)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_levels)])

        self.active_params = {}
        self.momentum_buffer = {}

    def reset_memory_state(self):
        self.active_params = {}
        self.momentum_buffer = {}

    def get_current_params(self, level_idx):
        if level_idx in self.active_params:
            return self.active_params[level_idx]
        return self.memory_levels[level_idx].get_initial_params()

    def forward(self, x):
        current_input = x
        for i in range(self.num_levels):
            curr_w, curr_b = self.get_current_params(i)
            out = self.memory_levels[i](current_input, curr_w, curr_b)
            current_input = self.norms[i](current_input + out)
        return current_input

    def update(self, key, value, global_chunk_idx, update_in_inference=True):
        """Meta-learning step (Test-Time Training): minimizza ||M(key) - value||."""
        if not update_in_inference:
            return
        with torch.enable_grad():
            for i in range(self.num_levels):
                if global_chunk_idx % self.update_frequencies[i] != 0:
                    continue

                current_lr, current_eta, current_decay = self.optimizers[i](key)
                module = self.memory_levels[i]
                curr_w, curr_b = self.get_current_params(i)

                pred = module(key, curr_w, curr_b)
                loss = self.mem_criterion(pred, value)

                params_to_diff = curr_w + curr_b
                grads = torch.autograd.grad(loss, params_to_diff, create_graph=True)
                num_w = len(curr_w)
                grads_w, grads_b = grads[:num_w], grads[num_w:]

                forget_factor = torch.sigmoid(self.forget_gates[i](key)).mean()
                decay = 1.0 - (current_decay * forget_factor)

                if i not in self.momentum_buffer:
                    self.momentum_buffer[i] = {
                        "w": [torch.zeros_like(p) for p in curr_w],
                        "b": [torch.zeros_like(p) for p in curr_b],
                    }
                mom_buf = self.momentum_buffer[i]

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

                self.active_params[i] = (new_weights, new_biases)
