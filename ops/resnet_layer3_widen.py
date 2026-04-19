from __future__ import annotations

import torch
import torch.nn as nn

from models.resnet_cifar import BasicBlock, ResNetCifar


def _new_conv_like(old: nn.Conv2d, *, in_ch: int, out_ch: int) -> nn.Conv2d:
    m = nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    return m.to(device=old.weight.device, dtype=old.weight.dtype)


def _new_bn_like(old: nn.BatchNorm2d, *, num_features: int) -> nn.BatchNorm2d:
    m = nn.BatchNorm2d(
        num_features,
        eps=old.eps,
        momentum=old.momentum,
        affine=old.affine,
        track_running_stats=old.track_running_stats,
    )
    return m.to(device=old.weight.device, dtype=old.weight.dtype)


def _copy_bn(old: nn.BatchNorm2d, new: nn.BatchNorm2d, old_c: int) -> None:
    with torch.no_grad():
        new.weight[:old_c].copy_(old.weight)
        new.bias[:old_c].copy_(old.bias)
        new.running_mean[:old_c].copy_(old.running_mean)
        new.running_var[:old_c].copy_(old.running_var)
        # Initialize new channels to identity BN defaults
        new.weight[old_c:].fill_(1.0)
        new.bias[old_c:].zero_()
        new.running_mean[old_c:].zero_()
        new.running_var[old_c:].fill_(1.0)


def widen_resnet_layer3(model: ResNetCifar, *, delta: int = 16) -> None:
    """
    Widen the channel width of CIFAR ResNet layer3 by `delta`.

    Design goal: be function-preserving at initialization:
    - New channels are initialized to 0 output by making their producing weights 0.
    - Downstream layers ignore new channels (weights on new input channels set to 0).
    - Final classifier ignores new channels (new input weights = 0).
    """
    if not isinstance(model, ResNetCifar):
        raise TypeError("widen_resnet_layer3 expects a ResNetCifar model")
    if not isinstance(model.layer3, nn.Sequential) or len(model.layer3) < 1:
        raise ValueError("model.layer3 must be a non-empty nn.Sequential")

    blocks = list(model.layer3)
    first: BasicBlock = blocks[0]  # type: ignore[assignment]
    old_out = int(first.bn2.num_features)
    new_out = old_out + int(delta)
    if int(delta) <= 0:
        return

    # Update each block
    prev_out = None
    for i, b in enumerate(blocks):
        assert isinstance(b, BasicBlock)
        old_b_out = int(b.bn2.num_features)
        if old_b_out != old_out:
            raise ValueError("layer3 blocks must share the same out channel width")

        # conv1: in depends on position (first block receives layer2 width)
        in_ch = int(b.conv1.in_channels) if i == 0 else new_out
        # Rebuild conv1 with widened out channels
        old_conv1 = b.conv1
        new_conv1 = _new_conv_like(old_conv1, in_ch=in_ch, out_ch=new_out)
        with torch.no_grad():
            # Copy overlap region
            overlap_in = min(int(old_conv1.in_channels), in_ch)
            new_conv1.weight[:old_out, :overlap_in].copy_(old_conv1.weight[:, :overlap_in])
            # Ensure new output channels are zero so they don't affect the forward
            new_conv1.weight[old_out:].zero_()
            if new_conv1.bias is not None:
                new_conv1.bias.zero_()
        b.conv1 = new_conv1

        # bn1: widen to new_out
        old_bn1 = b.bn1
        new_bn1 = _new_bn_like(old_bn1, num_features=new_out)
        _copy_bn(old_bn1, new_bn1, old_out)
        b.bn1 = new_bn1

        # conv2: in=new_out, out=new_out
        old_conv2 = b.conv2
        new_conv2 = _new_conv_like(old_conv2, in_ch=new_out, out_ch=new_out)
        with torch.no_grad():
            # Copy old weights into top-left block
            new_conv2.weight[:old_out, :old_out].copy_(old_conv2.weight)
            # Zero weights involving new channels to keep function-preserving behavior
            new_conv2.weight[old_out:, :].zero_()
            new_conv2.weight[:, old_out:].zero_()
            if new_conv2.bias is not None:
                new_conv2.bias.zero_()
        b.conv2 = new_conv2

        # bn2: widen to new_out
        old_bn2 = b.bn2
        new_bn2 = _new_bn_like(old_bn2, num_features=new_out)
        _copy_bn(old_bn2, new_bn2, old_out)
        b.bn2 = new_bn2

        # shortcut: if projection, widen its out channels to new_out; ensure new outputs are zero
        if isinstance(b.shortcut, nn.Sequential):
            if len(b.shortcut) != 2 or not isinstance(b.shortcut[0], nn.Conv2d) or not isinstance(
                b.shortcut[1], nn.BatchNorm2d
            ):
                raise ValueError("Unexpected shortcut structure in BasicBlock")
            sc_conv: nn.Conv2d = b.shortcut[0]
            sc_bn: nn.BatchNorm2d = b.shortcut[1]
            new_sc_conv = _new_conv_like(sc_conv, in_ch=int(sc_conv.in_channels), out_ch=new_out)
            with torch.no_grad():
                new_sc_conv.weight[:old_out].copy_(sc_conv.weight)
                new_sc_conv.weight[old_out:].zero_()
                if new_sc_conv.bias is not None:
                    new_sc_conv.bias.zero_()
            new_sc_bn = _new_bn_like(sc_bn, num_features=new_out)
            _copy_bn(sc_bn, new_sc_bn, old_out)
            b.shortcut = nn.Sequential(new_sc_conv, new_sc_bn)
        elif isinstance(b.shortcut, nn.Identity):
            # Identity shortcut passes through all channels from the previous block.
            # Since new channels are kept at 0, it's function-preserving.
            pass
        else:
            raise ValueError("Unexpected shortcut type")

        prev_out = new_out

    model.layer3 = nn.Sequential(*blocks)

    # Update classifier head input dim: either Linear or Sequential(fc1, relu, fc2)
    if isinstance(model.fc, nn.Linear):
        old_fc = model.fc
        new_fc = nn.Linear(new_out, int(old_fc.out_features)).to(
            device=old_fc.weight.device, dtype=old_fc.weight.dtype
        )
        with torch.no_grad():
            new_fc.weight[:, :old_out].copy_(old_fc.weight)
            new_fc.weight[:, old_out:].zero_()
            new_fc.bias.copy_(old_fc.bias)
        model.fc = new_fc
    elif isinstance(model.fc, nn.Sequential):
        # Expect (fc1, ReLU, fc2) from head widen
        if len(model.fc) != 3 or not isinstance(model.fc[0], nn.Linear) or not isinstance(
            model.fc[2], nn.Linear
        ):
            raise ValueError("Unexpected ResNet head structure; cannot widen fc input safely")
        fc1: nn.Linear = model.fc[0]
        fc2: nn.Linear = model.fc[2]
        new_fc1 = nn.Linear(new_out, int(fc1.out_features)).to(
            device=fc1.weight.device, dtype=fc1.weight.dtype
        )
        with torch.no_grad():
            overlap = min(int(fc1.in_features), new_out)
            new_fc1.weight[:, :overlap].copy_(fc1.weight[:, :overlap])
            new_fc1.weight[:, overlap:].zero_()
            new_fc1.bias.copy_(fc1.bias)
        # Keep fc2 unchanged (it already maps widened hidden to classes); extra input dims are OK.
        model.fc = nn.Sequential(new_fc1, model.fc[1], fc2)
    else:
        raise ValueError("Unexpected ResNet fc type")

