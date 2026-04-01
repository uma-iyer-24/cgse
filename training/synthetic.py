import torch
from torch.utils.data import DataLoader, TensorDataset


def build_synthetic_loaders(cfg: dict):
    """Tiny random data for Phase-0-style smoke tests."""
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    batch = tcfg["batch_size"]
    n = max(batch * 20, 256)
    x = torch.randn(n, mcfg["input_dim"])
    y = torch.randint(0, mcfg["output_dim"], (n,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    return loader, loader
