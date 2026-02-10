import torch
import torch.nn.functional as F

def train_one_epoch(model, optimizer, device):
    model.train()

    x = torch.randn(64, 10).to(device)
    y = torch.randint(0, 2, (64,)).to(device)

    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    return loss.item()
