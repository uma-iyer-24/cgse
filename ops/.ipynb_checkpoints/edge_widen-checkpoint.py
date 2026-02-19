import torch
import torch.nn as nn

def widen_linear(layer, delta):
    in_f = layer.in_features
    out_f = layer.out_features

    new_layer = nn.Linear(in_f, out_f + delta)

    with torch.no_grad():
        new_layer.weight[:out_f] = layer.weight
        new_layer.bias[:out_f] = layer.bias

    return new_layer
