import torch
import torch.nn as nn


def _new_linear(in_features: int, out_features: int, like: nn.Linear) -> nn.Linear:
    """Build Linear on the same device/dtype as an existing layer (avoids CPU tensors on MPS/CUDA)."""
    m = nn.Linear(in_features, out_features)
    return m.to(device=like.weight.device, dtype=like.weight.dtype)


def edge_widen(model, target_node_id=None, delta=4):

    # auto pick first Linear
    if target_node_id is None:
        for nid in model.execution_order:
            if isinstance(model.nodes[nid], nn.Linear):
                target_node_id = nid
                break

    order = model.execution_order
    idx = order.index(target_node_id)
    layer = model.nodes[target_node_id]

    if not isinstance(layer, nn.Linear):
        raise ValueError("Only Linear widening supported")

    old_out = layer.out_features
    new_out = old_out + delta

    # ---- widen target layer ----
    new_layer = _new_linear(layer.in_features, new_out, layer)

    with torch.no_grad():
        new_layer.weight[:old_out] = layer.weight
        new_layer.bias[:old_out] = layer.bias

    model.nodes[target_node_id] = new_layer

    # ---- propagate to ALL downstream Linear layers ----
    current_in = new_out

    for next_id in order[idx+1:]:

        next_layer = model.nodes[next_id]

        if isinstance(next_layer, nn.Linear):

            old_next = next_layer

            resized = _new_linear(current_in, old_next.out_features, old_next)

            with torch.no_grad():
                # copy overlapping weights only
                overlap = min(old_next.in_features, current_in)
                resized.weight[:, :overlap] = old_next.weight[:, :overlap]
                resized.bias.copy_(old_next.bias)

            model.nodes[next_id] = resized

            # update for next hop
            current_in = resized.out_features

    return model
