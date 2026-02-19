import torch
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

    def get_node(self, node_id):
        return self.nodes[node_id]

    def forward(self, x):
        for node_id in self.execution_order:
            x = self.nodes[node_id](x)
        return x

    def describe(self):
        print("\n=== Graph Execution Order ===")
        for i, node_id in enumerate(self.execution_order):
            layer = self.nodes[node_id]
            shape = ""
            if hasattr(layer, "in_features"):
                shape = f"({layer.in_features} → {layer.out_features})"
            print(f"{i}: {node_id} | {layer.__class__.__name__} {shape}")

    def validate(self):
        prev_out = None
        for node_id in self.execution_order:
            layer = self.nodes[node_id]
            if isinstance(layer, nn.Linear):
                if prev_out is not None and layer.in_features != prev_out:
                    raise ValueError(
                        f"Shape mismatch at {node_id}: "
                        f"expected {layer.in_features}, got {prev_out}"
                    )
                prev_out = layer.out_features

    def widen_node(self, node_id, extra_out):
        import torch.nn as nn
        import torch
    
        node = self.nodes[node_id]
    
        if not isinstance(node, nn.Linear):
            raise ValueError("Only Linear widening supported")
    
        old_out = node.out_features
        new_out = old_out + extra_out
    
        # create widened layer
        new_layer = nn.Linear(node.in_features, new_out)
    
        with torch.no_grad():
            new_layer.weight[:old_out] = node.weight
            new_layer.bias[:old_out] = node.bias
    
        self.nodes[node_id] = new_layer
    
        # ---- PROPAGATE TO NEXT LINEAR ----
        order = self.execution_order
        idx = order.index(node_id)
    
        for next_id in order[idx+1:]:
            next_node = self.nodes[next_id]
    
            if isinstance(next_node, nn.Linear):
    
                new_next = nn.Linear(new_out, next_node.out_features)
    
                with torch.no_grad():
                    new_next.weight[:, :old_out] = next_node.weight
                    new_next.bias = next_node.bias
    
                self.nodes[next_id] = new_next
                break

    def insert_after(self, target_id, new_id, new_module):
        if target_id not in self.execution_order:
            raise ValueError(f"{target_id} not in graph")
    
        if new_id in self.nodes:
            raise ValueError(f"{new_id} already exists")
    
        # register module
        self.nodes[new_id] = new_module
    
        # find index
        idx = self.execution_order.index(target_id)
    
        # insert after target
        self.execution_order.insert(idx + 1, new_id)
