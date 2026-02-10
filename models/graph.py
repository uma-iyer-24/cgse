import torch.nn as nn

class Node:
    def __init__(self, node_id, module):
        self.id = node_id
        self.module = module

class GraphModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes = nn.ModuleDict()
        self.execution_order = []

    def add_node(self, node_id, module):
        self.nodes[node_id] = module
        self.execution_order.append(node_id)

    def forward(self, x):
        for node_id in self.execution_order:
            x = self.nodes[node_id](x)
        return x
