from __future__ import annotations

import torch.nn as nn

from models.resnet_cifar import BasicBlock, ResNetCifar


def insert_resnet_block_layer3(model: ResNetCifar, *, position: str = "end") -> None:
    """
    ResNet-safe "deepen" op: insert one identity-initialized BasicBlock into layer3.

    Identity initialization:
    - Set the last BN scale (bn2.weight) to 0 so residual branch starts as 0.
    - Because inputs to blocks in this implementation are already post-ReLU, ReLU(x)=x and
      the block is function-preserving at init (up to numerical noise).
    """
    if not isinstance(model, ResNetCifar):
        raise TypeError("insert_resnet_block_layer3 expects a ResNetCifar model")
    if not isinstance(model.layer3, nn.Sequential) or len(model.layer3) < 1:
        raise ValueError("model.layer3 must be a non-empty nn.Sequential")

    last: BasicBlock = model.layer3[-1]  # type: ignore[assignment]
    # Infer channel width from the last block's conv2 output channels.
    out_ch = int(last.bn2.num_features)
    blk = BasicBlock(out_ch, out_ch, stride=1).to(
        device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
    )

    # Zero-init the last BN so residual branch contributes 0 initially.
    if hasattr(blk, "bn2") and isinstance(blk.bn2, nn.BatchNorm2d):
        nn.init.zeros_(blk.bn2.weight)

    blocks = list(model.layer3)
    if position == "end":
        blocks.append(blk)
    elif position == "start":
        blocks.insert(0, blk)
    else:
        raise ValueError("position must be 'start' or 'end'")
    model.layer3 = nn.Sequential(*blocks)

