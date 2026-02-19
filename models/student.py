import torch.nn as nn
from models.graph import GraphModule

class StudentNet(GraphModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.add_node("linear1", nn.Linear(input_dim, hidden_dim))
        self.add_node("relu1", nn.ReLU())
        self.add_node("linear2", nn.Linear(hidden_dim, output_dim))

    def deepen_after(self, node_id):
        """
        Insert identity-initialized Linear layer after node_id
        """
    
        node = self.nodes[node_id]
    
        if not isinstance(node, nn.Linear):
            raise ValueError("Can only deepen after Linear")
    
        dim = node.out_features
    
        new_layer = nn.Linear(dim, dim)
    
        # initialize as identity
        nn.init.eye_(new_layer.weight)
        nn.init.zeros_(new_layer.bias)
    
        new_name = f"{node_id}_deep"
    
        self.insert_after(node_id, new_name, new_layer)
