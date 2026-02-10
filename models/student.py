import torch.nn as nn
from models.graph import GraphModule

class StudentNet(GraphModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.add_node("linear1", nn.Linear(input_dim, hidden_dim))
        self.add_node("relu1", nn.ReLU())
        self.add_node("linear2", nn.Linear(hidden_dim, output_dim))
