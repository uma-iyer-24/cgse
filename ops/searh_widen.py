"""
SEArch widen operator — paper §3.4 "Widening the network".

Wraps the chosen base block with a parallel residual sep-conv branch
(paper Fig. 4b: "add a new node n + two new convolutional operations
in parallel to the original edge"). Identity-init via zero-init of the
new branch's final BN — function-preserving on insertion.

In our CIFAR-ResNet adaptation, "widen edge (a, b)" wraps a single
``BasicBlock`` in stage ``s`` with a ``WidenedBlock``. The wrapped block's
output is summed with the new sep-conv branch's output.
"""

from __future__ import annotations

import torch.nn as nn

from models.resnet_cifar import BasicBlock, ResNetCifar
from models.searh_blocks import WidenedBlock


def widen_resnet_stage(
    model: ResNetCifar,
    *,
    stage: int,
    block_idx: int | None = None,
) -> str:
    """
    Wrap ``model.layer{stage}[block_idx]`` with a ``WidenedBlock`` (parallel
    sep-conv branch summed in). Returns a short description of what was wrapped.

    When ``block_idx is None`` the *last unwrapped ``BasicBlock``* in the stage
    is chosen (matches SEArch's end-of-stage edge-splitting semantics).
    """
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1/2/3, got {stage}")
    layer: nn.Sequential = getattr(model, f"layer{stage}")
    blocks = list(layer)
    if not blocks:
        raise ValueError(f"layer{stage} has no blocks")
    if block_idx is None:
        # Pick the last plain BasicBlock that is safe to wrap (in_ch == out_ch,
        # stride 1) — the first block of each non-stage-1 layer has stride 2
        # and channel doubling, so wrapping it with a same-shape parallel
        # branch is not legal. Paper's "edge at end of stage" naturally
        # corresponds to a stride-1 same-channel block.
        candidates: list[int] = []
        for i, b in enumerate(blocks):
            if not isinstance(b, BasicBlock):
                continue
            in_ch = int(b.conv1.in_channels)
            out_ch = int(b.bn2.num_features)
            stride = (int(b.conv1.stride[0])
                      if isinstance(b.conv1.stride, tuple)
                      else int(b.conv1.stride))
            if in_ch == out_ch and stride == 1:
                candidates.append(i)
        if not candidates:
            raise ValueError(
                f"layer{stage} has no widenable BasicBlock (need stride=1, in==out)"
            )
        block_idx = candidates[-1]
    if block_idx < 0 or block_idx >= len(blocks):
        raise IndexError(f"block_idx out of range for layer{stage}")

    base = blocks[block_idx]
    # Infer channels.
    if isinstance(base, BasicBlock):
        ch = int(base.bn2.num_features)
        wrapped_kind = "BasicBlock"
    elif isinstance(base, WidenedBlock):
        # Wrap an already-widened block — supported (paper does not forbid
        # repeated widens). We use the channels of its base.
        if isinstance(base.base, BasicBlock):
            ch = int(base.base.bn2.num_features)
        else:
            raise ValueError("WidenedBlock with unknown inner type")
        wrapped_kind = "WidenedBlock"
    else:
        # DeepenBlock has its own channels via .body.bn2
        if hasattr(base, "body") and hasattr(base.body, "bn2"):
            ch = int(base.body.bn2.num_features)
            wrapped_kind = type(base).__name__
        else:
            raise ValueError(f"Cannot wrap unknown block type {type(base).__name__}")

    wrapped = WidenedBlock(base, ch).to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    blocks[block_idx] = wrapped
    setattr(model, f"layer{stage}", nn.Sequential(*blocks))
    return wrapped_kind
