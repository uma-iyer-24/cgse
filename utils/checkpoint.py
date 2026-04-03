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


def load_model_weights(model, path: str) -> None:
    """Load `model` weights from a file saved by `save_checkpoint` (expects key `model`)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path.resolve()}")
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
