"""
Structural critic for CGSE (Critic-Guided Self-Evolution).

Experimental design (this repository)
-------------------------------------
* **SEArch-style control baseline:** a frozen *teacher* network supplies soft targets
  (knowledge distillation) while the student may undergo structural mutations. That
  is the external-guidance condition we compare against.

* **CGSE:** the same training and mutation machinery, but the teacher is *removed* and
  a *critic* trained on internal optimization statistics scores proposed structural
  actions (when/where to mutate). The critic does not replace the label loss for
  classification; it replaces the *teacher* as the auxiliary signal for evolution.

Other ideas explored in early manuscript drafts (multi-objective arbitration, predictive
selection, staged teacher–critic hybrids, etc.) are **out of scope** for this codebase;
the only comparison axis we implement toward is **teacher vs critic** as the source of
non-label guidance for structural change.

`train.py` wires this module for **mutation gating** (see **`critics/state_features.py`**
and YAML **`critic:`** block).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StructuralCritic(nn.Module):
    """
    Maps a vector of hand-crafted or pooled optimization statistics to a scalar score.

    The intended use in CGSE is to rank or gate structural operations (e.g. widen
    this linear now vs later), *not* to predict class logits (that remains the student's
    job with cross-entropy on labels).
    """

    def __init__(self, state_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (batch, state_dim) -> scores (batch,)"""
        return self.net(state).squeeze(-1)
