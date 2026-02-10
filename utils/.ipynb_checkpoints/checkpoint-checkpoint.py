import torch
from pathlib import Path

def save_checkpoint(model, optimizer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
