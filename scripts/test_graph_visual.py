import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.student import StudentNet
from ops.edge_split import edge_split
from ops.edge_widen import edge_widen


def print_graph(model, title):
    print(f"\n--- {title} ---")
    for node_id in model.execution_order:
        layer = model.nodes[node_id]
        print(f"{node_id} -> {layer.__class__.__name__}")


def run():

    # build small model
    model = StudentNet(input_dim=32, hidden_dim=16, output_dim=4)

    print_graph(model, "INITIAL GRAPH")

    # apply split
    model = edge_split(model)
    print_graph(model, "AFTER SPLIT")

    # apply widen
    model = edge_widen(model)
    print_graph(model, "AFTER WIDEN")


if __name__ == "__main__":
    run()
