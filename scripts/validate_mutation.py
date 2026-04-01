import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.graph_validator import validate_graph
import sys
from pathlib import Path

# add repo root to python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch

from models.graph import GraphModule
from ops.edge_split import edge_split
from ops.edge_widen import edge_widen


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run():
    torch.manual_seed(42)

    print("\n--- Creating base model ---")
    model = GraphModule(input_dim=16, hidden_dim=32, output_dim=4)
    base_params = count_params(model)
    print("Base params:", base_params)

    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))

    print("\n--- Forward before mutation ---")
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()

    print("Forward/backward OK")

    print("\n--- Applying edge_split ---")
    model2 = edge_split(model)
    validate_graph(model2)
    split_params = count_params(model2)
    print("Params after split:", split_params)

    print("\n--- Applying edge_widen ---")
    model3 = edge_widen(model2)
    validate_graph(model3)
    widen_params = count_params(model3)
    print("Params after widen:", widen_params)

    print("\n--- Forward after mutation ---")
    out = model3(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()

    print("Forward/backward after mutation OK")

    print("\nSUCCESS: mutation pipeline validated.")


if __name__ == "__main__":
    run()

