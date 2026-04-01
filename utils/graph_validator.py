import torch.nn as nn

def validate_graph(model, input_dim=16):
    """
    Checks structural integrity of GraphModule.
    Ensures dimensions match across execution order.
    """

    x_dim = input_dim

    for node_id in model.execution_order:

        if node_id not in model.nodes:
            raise ValueError(f"[Validator] Missing node: {node_id}")

        layer = model.nodes[node_id]

        if isinstance(layer, nn.Linear):

            if layer.in_features != x_dim:
                raise ValueError(
                    f"[Validator] Shape mismatch at {node_id}: "
                    f"expected input {x_dim}, got {layer.in_features}"
                )

            x_dim = layer.out_features

    return True
