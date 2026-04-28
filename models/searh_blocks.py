"""
Building blocks for the SEArch / CGSE-on-SEArch outer loop.

Paper reference: Liang, Xiang & Li (2025), Neurocomputing 651, "SEArch: A
self-evolving framework for network architecture optimization", §3.4 and §3.5.

We provide three modules:

* ``SepConv3x3``     — paper's "3x3 residual separable conv" body
                       (depthwise 3x3 → BN → ReLU → pointwise 1x1 → BN).
* ``DeepenBlock``    — wraps a SepConv3x3 with a residual connection, used
                       as the "deepen" edge-splitting operator. Identity-init
                       (last BN gamma = 0) so insertion is function-preserving.
* ``WidenedBlock``   — wraps an existing ``BasicBlock`` and adds one parallel
                       residual sep-conv branch, summed in before the final
                       ReLU. Used as the "widen" edge-splitting operator;
                       identity-init keeps the wrapped block's behavior at insert.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv3x3(nn.Module):
    """Depthwise + pointwise 3x3 separable conv (paper §3.5)."""

    def __init__(self, ch: int, init_zero: bool = False):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        nn.init.kaiming_normal_(self.dw.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw.weight, mode="fan_out", nonlinearity="relu")
        if init_zero:
            nn.init.zeros_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.dw(x)))
        out = self.bn2(self.pw(out))
        return out


class DeepenBlock(nn.Module):
    """
    Residual sep-conv block used by the deepen operator.

    forward: out = ReLU(x + SepConv(x))

    With ``init_zero=True``, ``SepConv(x) ≡ 0`` at init, so this block is
    function-preserving on insert (the previous post-ReLU input passes through).
    """

    def __init__(self, ch: int, init_zero: bool = True):
        super().__init__()
        self.body = SepConv3x3(ch, init_zero=init_zero)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x))


class WidenedBlock(nn.Module):
    """
    Wraps an existing ``BasicBlock`` and adds one parallel residual sep-conv
    branch summed into the original output. Implements the paper's "widen"
    edge-splitting (Fig. 4b): a new node n is inserted with two new
    convolutional operations (here merged into a single ``SepConv3x3``) that
    operates in parallel to the original edge.

    Initialisation: ``branch`` uses ``init_zero=True`` so its output is 0
    at init and the wrapped block's behavior is exactly preserved.
    """

    def __init__(self, base_block: nn.Module, ch: int):
        super().__init__()
        self.base = base_block
        self.branch = SepConv3x3(ch, init_zero=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.branch(x)
