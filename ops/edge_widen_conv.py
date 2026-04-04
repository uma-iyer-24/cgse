"""Widen the last conv block on CifarGraphNet (conv3 → bn3 → … → fc1 resize)."""

import torch
import torch.nn as nn


def edge_widen_conv3_cifar(model, delta: int) -> None:
    """
    Increase conv3 out_channels by delta, match bn3, widen fc1.in_features
    (flatten is 128*4*4 → (128+delta)*4*4 for default CifarGraphNet).
    """
    if delta <= 0:
        raise ValueError("delta must be positive")

    conv = model.nodes["conv3"]
    bn = model.nodes["bn3"]
    fc = model.nodes["fc1"]

    if not isinstance(conv, nn.Conv2d):
        raise TypeError("conv3 must be Conv2d")
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("bn3 must be BatchNorm2d")
    if not isinstance(fc, nn.Linear):
        raise TypeError("fc1 must be Linear")

    old_out = conv.out_channels
    old_in = conv.in_channels
    new_out = old_out + delta
    spatial = 16  # 4 * 4 after pool3 on CIFAR tail
    old_flat = old_out * spatial
    new_flat = new_out * spatial

    dev, dt = conv.weight.device, conv.weight.dtype

    new_conv = nn.Conv2d(
        old_in,
        new_out,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=conv.bias is not None,
    ).to(device=dev, dtype=dt)
    with torch.no_grad():
        new_conv.weight[:old_out].copy_(conv.weight)
        if conv.bias is not None:
            new_conv.bias[:old_out].copy_(conv.bias)

    new_bn = nn.BatchNorm2d(new_out).to(device=dev, dtype=dt)
    with torch.no_grad():
        new_bn.weight[:old_out].copy_(bn.weight)
        new_bn.bias[:old_out].copy_(bn.bias)
        new_bn.running_mean[:old_out].copy_(bn.running_mean)
        new_bn.running_var[:old_out].copy_(bn.running_var)

    new_fc = nn.Linear(new_flat, fc.out_features).to(device=dev, dtype=dt)
    with torch.no_grad():
        new_fc.weight[:, :old_flat].copy_(fc.weight)
        new_fc.bias.copy_(fc.bias)

    model.nodes["conv3"] = new_conv
    model.nodes["bn3"] = new_bn
    model.nodes["fc1"] = new_fc

    # split_before_fc1 inserts Linear(old_flat, old_flat) between flatten and fc1; resize it.
    order = model.execution_order
    if "flatten" in order and "fc1" in order:
        i0 = order.index("flatten") + 1
        i1 = order.index("fc1")
        for nid in order[i0:i1]:
            layer = model.nodes[nid]
            if (
                isinstance(layer, nn.Linear)
                and layer.in_features == old_flat
                and layer.out_features == old_flat
            ):
                new_lin = nn.Linear(new_flat, new_flat).to(device=dev, dtype=dt)
                with torch.no_grad():
                    nn.init.eye_(new_lin.weight)
                    new_lin.bias.zero_()
                model.nodes[nid] = new_lin
