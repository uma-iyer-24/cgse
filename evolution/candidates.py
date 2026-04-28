"""
Candidate-site enumeration for SEArch / CGSE evolution.

A "candidate" is an edge-splitting decision at a stage boundary:
``(stage, op)`` where ``op ∈ {deepen, widen}``. The outer loop turns the
chosen candidate into a structural edit using ``ops/searh_deepen``
(insert sep-conv block at the end of the stage) or ``ops/searh_widen``
(wrap the last block in the stage with a parallel sep-conv branch).

Following SEArch §3.4: deepen first until ``B_op_max`` ops have stacked in
a stage; then widen. We expose both options at every stage so the policy
(teacher MV or critic) can override the deepen-first heuristic if it wants.

Candidate ``node_id`` matches ``training/searh_node_map`` ("stage1",
"stage2", "stage3") so MV scores from the attention KD module can be
indexed directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch.nn as nn

from models.resnet_cifar import BasicBlock, ResNetCifar
from models.searh_blocks import DeepenBlock, WidenedBlock


@dataclass(frozen=True)
class Candidate:
    stage: int            # 1, 2, or 3
    op: str               # "deepen" or "widen"
    node_id: str          # "stage1" | "stage2" | "stage3"

    def describe(self) -> str:
        return f"{self.op} stage{self.stage} (end-of-stage)"


def count_deepens_in_stage(model: ResNetCifar, stage: int) -> int:
    """Number of ``DeepenBlock`` instances in the given stage."""
    layer = getattr(model, f"layer{stage}")
    return sum(1 for b in layer if isinstance(b, DeepenBlock))


def count_widens_in_stage(model: ResNetCifar, stage: int) -> int:
    """Number of ``WidenedBlock`` instances in the given stage."""
    layer = getattr(model, f"layer{stage}")
    return sum(1 for b in layer if isinstance(b, WidenedBlock))


def _is_widenable_basic_block(block: nn.Module) -> bool:
    """True iff `block` is a plain ``BasicBlock`` whose parallel-branch wrap
    is safe (input channels == output channels and stride 1).

    The first block of each non-stage-1 layer in CIFAR-ResNet has stride 2
    + channel doubling and is *not* safe to wrap with a same-shape parallel
    branch — those blocks are excluded from the widen candidate set.
    """
    if not isinstance(block, BasicBlock):
        return False
    in_ch = int(block.conv1.in_channels)
    out_ch = int(block.bn2.num_features)
    stride = int(block.conv1.stride[0]) if isinstance(block.conv1.stride, tuple) else int(block.conv1.stride)
    return in_ch == out_ch and stride == 1


def enumerate_candidates(
    model: ResNetCifar,
    *,
    stages: Tuple[int, ...] = (1, 2, 3),
    b_op_cap: int = 7,
    deepen_first: bool = True,
) -> List[Candidate]:
    """
    Enumerate one (stage, op) candidate per stage per legal op.

    * ``deepen`` is offered while the stage's deepen-count < ``b_op_cap``.
    * ``widen`` is offered when the stage still has at least one plain
      ``BasicBlock`` (so the wrap is non-trivial). When ``deepen_first``
      is True (paper default), widen is suppressed until the stage's
      deepen-count reaches the cap.
    """
    out: List[Candidate] = []
    for stage in stages:
        layer = getattr(model, f"layer{stage}")
        deepens = count_deepens_in_stage(model, stage)
        deepen_full = deepens >= int(b_op_cap)
        has_widenable = any(_is_widenable_basic_block(b) for b in layer)
        if not deepen_full:
            out.append(Candidate(stage=stage, op="deepen", node_id=f"stage{stage}"))
        if has_widenable and ((not deepen_first) or deepen_full):
            out.append(Candidate(stage=stage, op="widen", node_id=f"stage{stage}"))
    return out
