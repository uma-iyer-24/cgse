"""
SEArch deepen operator — paper §3.4 "Deepening the network".

Stacks a new sep-conv 3x3 residual block onto the chosen edge in the
student. Function-preserving at insertion via zero-init of the inserted
block's last BN (paper-compatible identity init).

In our CIFAR-ResNet adaptation, "edge (a, b)" between two adjacent
``BasicBlock``s in stage ``s`` is realised by appending a ``DeepenBlock``
into ``model.layer{s}`` immediately after the corresponding base block.

API matches the existing ops/* style: takes ``model`` and a target site
descriptor, mutates ``model`` in place, and returns nothing. Caller is
responsible for refreshing the optimizer afterwards.
"""

from __future__ import annotations

import torch.nn as nn

from models.resnet_cifar import BasicBlock, ResNetCifar
from models.searh_blocks import DeepenBlock


def deepen_resnet_stage(
    model: ResNetCifar,
    *,
    stage: int,
    after_block_idx: int | None = None,
) -> int:
    """
    Insert one ``DeepenBlock`` into ``model.layer{stage}`` after the block at
    ``after_block_idx`` (0-indexed). When ``after_block_idx is None`` the new
    block is appended at the end of the stage (paper default for SEArch's
    "deepen at end-of-stage"). Returns the new total block count of that stage.
    """
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1/2/3, got {stage}")
    layer: nn.Sequential = getattr(model, f"layer{stage}")
    blocks = list(layer)
    if not blocks:
        raise ValueError(f"layer{stage} has no blocks")
    if after_block_idx is None:
        after_block_idx = len(blocks) - 1
    if after_block_idx < 0 or after_block_idx >= len(blocks):
        raise IndexError(f"after_block_idx out of range for layer{stage}")

    base: BasicBlock = blocks[after_block_idx]  # type: ignore[assignment]
    if not isinstance(base, BasicBlock):
        # If we previously deepened/widened, it might be a DeepenBlock or WidenedBlock.
        # We still infer channels and proceed — channels are preserved by both ops.
        pass

    # Infer channel width from the block's last BN.
    if hasattr(base, "bn2"):
        ch = int(base.bn2.num_features)
    elif hasattr(base, "base") and hasattr(base.base, "bn2"):
        ch = int(base.base.bn2.num_features)  # WidenedBlock wraps a BasicBlock as .base
    elif hasattr(base, "body") and hasattr(base.body, "bn2"):
        ch = int(base.body.bn2.num_features)  # already a DeepenBlock — channels match
    else:
        raise ValueError("Cannot infer channel width from base block")

    new_block = DeepenBlock(ch, init_zero=True).to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    blocks.insert(after_block_idx + 1, new_block)
    setattr(model, f"layer{stage}", nn.Sequential(*blocks))
    return len(blocks)
