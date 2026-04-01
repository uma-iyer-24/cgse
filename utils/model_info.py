from __future__ import annotations

import torch.nn as nn
from typing import Optional, Tuple


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def first_linear_node_id(model) -> Optional[str]:
    for nid in model.execution_order:
        if isinstance(model.nodes[nid], nn.Linear):
            return nid
    return None


def linear_layer_shapes(model, node_id: str) -> Tuple[int, int]:
    layer = model.nodes[node_id]
    if not isinstance(layer, nn.Linear):
        raise ValueError(f"{node_id} is not Linear")
    return layer.in_features, layer.out_features
