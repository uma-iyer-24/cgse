from __future__ import annotations

import torch
import torch.nn as nn

from models.resnet_cifar import ResNetCifar


def widen_resnet_head(
    model: ResNetCifar,
    *,
    hidden_delta: int = 64,
) -> None:
    """
    Net2Net-style widening of the ResNet CIFAR classifier head.

    Replace:
      fc:  (d -> C)
    with:
      fc1: (d -> d+Δ)  (initialized to [I; 0])
      relu
      fc2: (d+Δ -> C)  (initialized to [W, 0] and b copied)

    This preserves logits at initialization for any input (up to numerical noise),
    making it safe to apply mid-training as a "scheduled widen" baseline.
    """
    if not isinstance(model, ResNetCifar):
        raise TypeError("widen_resnet_head expects a ResNetCifar model")
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError("model.fc must be nn.Linear")

    old_fc: nn.Linear = model.fc
    d = int(old_fc.in_features)
    c = int(old_fc.out_features)
    new_d = d + int(hidden_delta)

    device = old_fc.weight.device
    dtype = old_fc.weight.dtype

    fc1 = nn.Linear(d, new_d).to(device=device, dtype=dtype)
    fc2 = nn.Linear(new_d, c).to(device=device, dtype=dtype)

    with torch.no_grad():
        fc1.weight.zero_()
        fc1.bias.zero_()
        fc1.weight[:d, :d].copy_(torch.eye(d, device=device, dtype=dtype))

        fc2.weight.zero_()
        fc2.bias.copy_(old_fc.bias)
        fc2.weight[:, :d].copy_(old_fc.weight)

    model.fc = nn.Sequential(fc1, nn.ReLU(), fc2)

