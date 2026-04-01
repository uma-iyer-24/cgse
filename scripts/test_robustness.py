import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import random

from models.graph import GraphModule
from ops.edge_split import edge_split
from ops.edge_widen import edge_widen
from utils.optimizer_utils import refresh_optimizer


def build_model():
    m = GraphModule()
    m.add_node("l1", nn.Linear(16, 32))
    m.add_node("l2", nn.ReLU())
    m.add_node("l3", nn.Linear(32, 4))
    return m


def has_nan(model):
    for p in model.parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            return True
    return False


def train_steps(model, optimizer, steps=3):
    x = torch.randn(8,16)
    y = torch.randn(8,4)

    for _ in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = ((out-y)**2).mean()
        loss.backward()
        optimizer.step()

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("NaN loss detected")

    return loss.item()


def mutation_sequence(model, optimizer, seq):
    for op in seq:
        if op=="split":
            model = edge_split(model)
        elif op=="widen":
            model = edge_widen(model)

        optimizer = refresh_optimizer(optimizer, model)

        if has_nan(model):
            raise RuntimeError("NaN parameters detected")

        train_steps(model, optimizer, 1)

    return model


def run():

    print("\n--- TEST 1: repeated mutations ---")

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # random mutation sequence
    ops = ["split","widen"]
    seq = [random.choice(ops) for _ in range(10)]
    print("sequence:", seq)

    mutation_sequence(model, optimizer, seq)

    print("OK: repeated mutations stable")


    print("\n--- TEST 2: deterministic replay ---")

    torch.manual_seed(0)
    random.seed(0)

    seq = [random.choice(ops) for _ in range(6)]

    torch.manual_seed(42)
    m1 = build_model()
    opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
    mutation_sequence(m1, opt1, seq)
    p1 = sum(p.sum().item() for p in m1.parameters())

    torch.manual_seed(42)
    m2 = build_model()
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    mutation_sequence(m2, opt2, seq)
    p2 = sum(p.sum().item() for p in m2.parameters())

    diff = abs(p1-p2)

    print("param checksum diff:", diff)

    if diff > 1e-6:
        raise RuntimeError("Deterministic replay FAILED")

    print("OK: deterministic replay works")


    print("\nSUCCESS: Phase-1 robustness validated.")


if __name__=="__main__":
    run()

