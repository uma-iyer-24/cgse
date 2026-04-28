"""
Per-candidate critic used by CGSE-on-SEArch (replaces the teacher's
modification-value scorer with a learned policy over (stage, op) pairs).

Input per candidate:  global_state (8-dim, from build_critic_state)
                    ⊕ local_descriptor (5-dim from _make_local_descriptor)
Output per candidate: scalar score. Softmax over candidates → action policy.

Trained via REINFORCE with Δval reward on the next stage and an entropy
bonus to discourage noop-collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerCandidateCritic(nn.Module):
    def __init__(self, state_dim: int, local_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.in_dim = int(state_dim + local_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (K, state_dim + local_dim) → (K, 1) scores."""
        return self.net(x)
