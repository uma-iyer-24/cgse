import torch.nn as nn
import torch

def split_edge(model, target_node_id):
    idx = model.execution_order.index(target_node_id)
    old_layer = model.nodes[target_node_id]

    # Phase 1 constraints
    if not isinstance(old_layer, nn.Linear):
        raise ValueError("Only Linear layers can be split")

    if idx == len(model.execution_order) - 1:
        raise ValueError("Cannot split output layer in Phase 1")

    in_f = old_layer.in_features

    # New identity-preserving layer
    new_layer = nn.Linear(in_f, in_f)
    new_id = f"{target_node_id}_split"

    with torch.no_grad():
        nn.init.eye_(new_layer.weight)
        new_layer.bias.zero_()

    # Insert BEFORE old layer
    model.nodes[new_id] = new_layer
    model.execution_order.insert(idx, new_id)
