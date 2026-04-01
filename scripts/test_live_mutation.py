import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from models.graph import GraphModule
from ops.edge_split import edge_split
from ops.edge_widen import edge_widen


def build_model():
    m = GraphModule()
    m.add_node("l1", nn.Linear(16, 32))
    m.add_node("l2", nn.ReLU())
    m.add_node("l3", nn.Linear(32, 4))
    return m


def train_step(model, opt, x, y):
    opt.zero_grad()
    out = model(x)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    opt.step()
    return loss.item()


def run():

    print("\n--- Building model ---")
    model = build_model()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(8, 16)
    y = torch.randn(8, 4)

    print("\n--- Training BEFORE mutation ---")
    for i in range(3):
        loss = train_step(model, optimizer, x, y)
        print("step", i, "loss", loss)

    print("\n--- APPLYING LIVE MUTATION ---")

    model = edge_split(model)
    model = edge_widen(model)

    from utils.optimizer_utils import refresh_optimizer
    optimizer = refresh_optimizer(optimizer, model)


    # 🔴 CRUCIAL PART OF 1.7
    # refresh optimizer so new params are tracked
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n--- Training AFTER mutation ---")
    for i in range(3):
        loss = train_step(model, optimizer, x, y)
        print("step", i, "loss", loss)

    print("\nSUCCESS: live mutation training works")


if __name__ == "__main__":
    run()

