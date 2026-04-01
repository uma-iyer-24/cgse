import torch.nn as nn
import torch


def _infer_input_dim(model, node_index, default=16):
    """
    Walk graph up to node_index to compute feature size.
    Assumes sequential execution graph.
    """

    dim = default

    for nid in model.execution_order[:node_index]:
        layer = model.nodes[nid]

        if isinstance(layer, nn.Linear):
            dim = layer.out_features

    return dim


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

    if idx == len(model.execution_order) - 1:
        raise ValueError("Cannot split output layer in Phase 1")

    # ✅ infer TRUE incoming dimension
    in_f = _infer_input_dim(model, idx)

    # identity layer with correct dimension
    new_layer = nn.Linear(in_f, in_f)
    new_id = f"{target_node_id}_split_{len(model.execution_order)}"

    with torch.no_grad():
        nn.init.eye_(new_layer.weight)
        new_layer.bias.zero_()

    model.nodes[new_id] = new_layer
    model.execution_order.insert(idx, new_id)

    return model
