"""Policy over a small discrete set of mutation candidates (Tier 1b CGSE)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiscreteMutationCritic(nn.Module):
    """
    Maps global training state to logits over up to ``max_actions`` mutation slots.
    Invalid slots should be masked before softmax (see ``masked_logits``).
    """

    def __init__(self, state_dim: int, max_actions: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.max_actions = max_actions
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, max_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (1, state_dim) -> logits (max_actions,)"""
        h = self.encoder(state)
        return self.head(h).squeeze(0)
