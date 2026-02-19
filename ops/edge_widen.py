import torch
import torch.nn as nn


def widen_linear(layer, delta):
    in_f = layer.in_features
    out_f = layer.out_features

    new_layer = nn.Linear(in_f, out_f + delta)

    with torch.no_grad():
        new_layer.weight[:out_f] = layer.weight
        new_layer.bias[:out_f] = layer.bias

    return new_layer


def edge_widen(model, target_node_id, delta=4):
    """
    Graph mutation operator:
    widens a Linear node and propagates shape to next Linear.
    """

    if target_node_id not in model.execution_order:
        raise ValueError(f"{target_node_id} not found")

    layer = model.nodes[target_node_id]

    if not isinstance(layer, nn.Linear):
        raise ValueError("Phase-1 only supports widening Linear layers")

    old_out = layer.out_features
    new_layer = widen_linear(layer, delta)
    model.nodes[target_node_id] = new_layer

    # propagate to next linear
    idx = model.execution_order.index(target_node_id)

    for next_id in model.execution_order[idx+1:]:
        next_layer = model.nodes[next_id]

        if isinstance(next_layer, nn.Linear):

            new_next = nn.Linear(old_out + delta, next_layer.out_features)

            with torch.no_grad():
                new_next.weight[:, :old_out] = next_layer.weight
                new_next.bias = next_layer.bias

            model.nodes[next_id] = new_next
            break

    return model

