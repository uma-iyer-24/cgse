import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.nn as nn

from models.student import StudentNet
from ops.edge_split import edge_split
from ops.edge_widen import edge_widen


def layer_label(name, layer):
    """Return readable label for each layer"""
    if isinstance(layer, nn.Linear):
        return f"{name}:{layer.in_features}->{layer.out_features}"
    else:
        return name


def ascii_graph(model, title):
    """Print architecture as ASCII chain"""

    print(f"\n=== {title} ===")

    parts = []

    for node_id in model.execution_order:
        layer = model.nodes[node_id]
        parts.append(f"[{layer_label(node_id, layer)}]")

    print(" -> ".join(parts))


def run():

    # Build base model
    model = StudentNet(input_dim=32, hidden_dim=16, output_dim=4)

    ascii_graph(model, "INITIAL")

    # Mutation 1
    model = edge_split(model)
    ascii_graph(model, "AFTER SPLIT")

    # Mutation 2
    model = edge_widen(model)
    ascii_graph(model, "AFTER WIDEN")


if __name__ == "__main__":
    run()
