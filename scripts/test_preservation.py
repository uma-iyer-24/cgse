import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.graph import GraphModule
from ops.edge_split import edge_split

x = torch.randn(8,16)

model = GraphModule(16,32,4)
out1 = model(x)

model2 = edge_split(model)

out2 = model2(x)

diff = (out1 - out2).abs().mean().item()

print("Output difference:", diff)
