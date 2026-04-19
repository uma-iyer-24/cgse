"""
ResNet for CIFAR-10/100 (32x32), following the standard 3-stage CIFAR design.

This is used for Tier 2 parity experiments (SEArch-style KD table setting):
- Teacher: ResNet-56
- Student: ResNet-20 (~0.27M params)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            # Option A (CIFAR ResNet): downsample with 1x1 conv
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


@dataclass(frozen=True)
class ResNetCifarSpec:
    depth: int
    num_classes: int = 10
    base_width: int = 16


class ResNetCifar(nn.Module):
    """
    CIFAR ResNet with BasicBlock.

    Depth must satisfy: depth = 6n + 2 (e.g. 20, 32, 44, 56, 110).
    """

    def __init__(self, *, depth: int, num_classes: int = 10, base_width: int = 16):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"ResNetCifar depth must be 6n+2, got {depth}")
        n = (depth - 2) // 6
        w = int(base_width)

        self.depth = int(depth)
        self.num_classes = int(num_classes)
        self.base_width = w

        self.conv1 = _conv3x3(3, w, stride=1)
        self.bn1 = nn.BatchNorm2d(w)

        self.layer1 = self._make_layer(in_ch=w, out_ch=w, nblocks=n, stride=1)
        self.layer2 = self._make_layer(in_ch=w, out_ch=2 * w, nblocks=n, stride=2)
        self.layer3 = self._make_layer(in_ch=2 * w, out_ch=4 * w, nblocks=n, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4 * w, self.num_classes)

        self._init_weights()

    def _make_layer(self, *, in_ch: int, out_ch: int, nblocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, nblocks):
            blocks.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def build_resnet_cifar(spec: ResNetCifarSpec) -> ResNetCifar:
    return ResNetCifar(depth=spec.depth, num_classes=spec.num_classes, base_width=spec.base_width)

