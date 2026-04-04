"""Fixed-size vectors of training statistics for StructuralCritic (CGSE gating)."""

from __future__ import annotations

import math

import torch

STATE_DIM = 8


def build_critic_state(
    *,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    epoch: int,
    max_epochs: int,
    num_params: int,
    prev_train_loss: float | None,
    prev_val_acc: float | None,
    anchor_train_loss: float | None,
    device: torch.device,
) -> torch.Tensor:
    """
    Hand-crafted features (no gradients). Order must match what the critic was trained on.
    Values are roughly in [-1, 1] or [0, 1] for stability.
    """
    denom = max(anchor_train_loss or train_loss, 1e-6)
    tl_n = min(train_loss / denom, 3.0) / 3.0

    e_n = epoch / max(max_epochs - 1, 1)

    p_n = min(math.log2(max(num_params, 1)) / 22.0, 1.0)

    dva = 0.0 if prev_val_acc is None else val_acc - prev_val_acc
    dva_n = max(-0.2, min(0.2, dva)) / 0.2

    dtl = 0.0 if prev_train_loss is None else train_loss - prev_train_loss
    dtl_n = max(-1.0, min(1.0, dtl / denom))

    vl_n = min(val_loss / 5.0, 2.0) / 2.0

    feats = [
        tl_n,
        val_acc,
        e_n,
        p_n,
        dva_n,
        dtl_n,
        vl_n,
        1.0,
    ]
    assert len(feats) == STATE_DIM
    return torch.tensor(feats, device=device, dtype=torch.float32).unsqueeze(0)
