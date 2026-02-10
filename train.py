import torch
import yaml
from models.student import StudentNet
from training.loop import train_one_epoch
from utils.checkpoint import save_checkpoint

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config("configs/base.yaml")

    device = torch.device(cfg["device"])
    model = StudentNet(**cfg["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["lr"]
    )

    for epoch in range(cfg["training"]["epochs"]):
        loss = train_one_epoch(model, optimizer, device)
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

    save_checkpoint(model, optimizer, "checkpoints/phase0.pt")

if __name__ == "__main__":
    main()
