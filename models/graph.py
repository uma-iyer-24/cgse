import torch
import torch.nn as nn

class Node:
    def __init__(self, node_id, module):
        self.id = node_id
        self.module = module

class GraphModule(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None):
        super().__init__()
        self.nodes = nn.ModuleDict()
        self.execution_order = []

        # optional auto-build simple MLP if dims provided
        if input_dim is not None:
            self.add_node("fc1", nn.Linear(input_dim, hidden_dim))
            self.add_node("relu1", nn.ReLU())
            self.add_node("fc2", nn.Linear(hidden_dim, output_dim))


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
        node = self.nodes[node_id]
    
        if not isinstance(node, nn.Linear):
            raise ValueError("Only Linear widening supported")
    
        old_out = node.out_features
        new_out = old_out + extra_out

        # --- widen selected layer (match device/dtype for MPS/CUDA) ---
        new_layer = nn.Linear(node.in_features, new_out).to(
            device=node.weight.device, dtype=node.weight.dtype
        )

        with torch.no_grad():
            new_layer.weight[:old_out] = node.weight
            new_layer.bias[:old_out] = node.bias

        self.nodes[node_id] = new_layer
    
        # --- propagate to ALL downstream Linear layers ---
        order = self.execution_order
        idx = order.index(node_id)
    
        prev_out = new_out
    
        for next_id in order[idx+1:]:
    
            layer = self.nodes[next_id]
    
            if isinstance(layer, nn.Linear):

                new_next = nn.Linear(prev_out, layer.out_features).to(
                    device=layer.weight.device, dtype=layer.weight.dtype
                )

                with torch.no_grad():
                    copy_in = min(layer.in_features, prev_out)
                    new_next.weight[:, :copy_in] = layer.weight[:, :copy_in]
                    new_next.bias.copy_(layer.bias)

                self.nodes[next_id] = new_next
    
                prev_out = new_next.out_features
    
            elif isinstance(layer, nn.ReLU):
                # activations keep dimension unchanged
                continue


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
