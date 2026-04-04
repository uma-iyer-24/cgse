import torch.nn as nn
import torch


def edge_split(model, target_node_id=None):

    # auto-pick first Linear
    if target_node_id is None:
        for nid in model.execution_order:
            if isinstance(model.nodes[nid], nn.Linear):
                target_node_id = nid
                break

    idx = model.execution_order.index(target_node_id)
    old_layer = model.nodes[target_node_id]

    if not isinstance(old_layer, nn.Linear):
        raise ValueError("Only Linear layers can be split")

    # Fan-in to this linear (matches activations after Flatten / prior layers in a valid graph).
    in_f = old_layer.in_features

    # identity layer with correct dimension (same device/dtype as target for MPS/CUDA)
    new_layer = nn.Linear(in_f, in_f).to(
        device=old_layer.weight.device, dtype=old_layer.weight.dtype
    )
    new_id = f"{target_node_id}_split_{len(model.execution_order)}"

    with torch.no_grad():
        nn.init.eye_(new_layer.weight)
        new_layer.bias.zero_()

    model.nodes[new_id] = new_layer
    model.execution_order.insert(idx, new_id)

    return model
