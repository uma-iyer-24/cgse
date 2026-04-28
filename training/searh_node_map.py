"""
Student/teacher node correspondence for the SEArch outer loop.

Following the paper's intuition (§3.2-§3.3) but adapted to CIFAR-ResNets,
we treat each **stage output** as a "node":

    stage1_out:  feature map after layer1   (16-ch, 32x32)
    stage2_out:  feature map after layer2   (32-ch, 16x16)
    stage3_out:  feature map after layer3   (64-ch, 8x8)

Both student (ResNet-20) and teacher (ResNet-56) share these spatial/
channel widths at these three points, so the channel-attention KD module
can match them directly. Any mutation inside a stage (deepen / widen on
its blocks) preserves the stage's I/O shape and therefore preserves the
node identity — hooks remain valid across mutations without rebuilding.

For γ in Eq. 6, the stage-to-stage correspondence is exact (1↔1, 2↔2,
3↔3), so γ does not need to choose an inner block. We retain the γ
hyperparameter only for use inside the (Eq. 6) widen-supervision rule
within future intra-stage candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch.nn as nn

from models.resnet_cifar import ResNetCifar


@dataclass(frozen=True)
class SearhNodeMap:
    pairs: Tuple[Tuple[str, str], ...]
    student_lookup: Dict[str, nn.Module]
    teacher_lookup: Dict[str, nn.Module]

    def node_ids(self) -> List[str]:
        return [s for s, _ in self.pairs]


def build_node_map(
    student: ResNetCifar,
    teacher: ResNetCifar,
    *,
    gamma: float = 0.5,
) -> SearhNodeMap:
    """Build (student_stage, teacher_stage) module pairs for stages 1, 2, 3."""
    del gamma  # reserved for future intra-stage Eq.6 selection
    pairs = (
        ("stage1", "stage1"),
        ("stage2", "stage2"),
        ("stage3", "stage3"),
    )
    student_lookup = {
        "stage1": student.layer1,
        "stage2": student.layer2,
        "stage3": student.layer3,
    }
    teacher_lookup = {
        "stage1": teacher.layer1,
        "stage2": teacher.layer2,
        "stage3": teacher.layer3,
    }
    return SearhNodeMap(
        pairs=pairs,
        student_lookup=student_lookup,
        teacher_lookup=teacher_lookup,
    )
