import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from models.cifar_student import CifarGraphNet
from models.student import StudentNet
from ops.edge_widen import edge_widen
from training.data import build_cifar10_loaders
from training.loop import evaluate, train_one_epoch
from training.synthetic import build_synthetic_loaders
from utils.checkpoint import save_checkpoint
from utils.graph_validator import validate_forward
from utils.model_info import (
    count_trainable_parameters,
    first_linear_node_id,
    linear_layer_shapes,
)
from utils.mutation_log import append_mutation_jsonl
from utils.optimizer_utils import refresh_optimizer
from utils.repro import set_seed


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


_METRIC_FIELDS = [
    "utc_ts",
    "experiment",
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "num_parameters",
    "mutation_applied_yet",
]


def _append_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_METRIC_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="CGSE training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase2_cifar.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override config device (e.g. cpu, mps, cuda, auto)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    device = resolve_device(args.device or cfg.get("device", "cpu"))

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    model_name = cfg.get("model", {}).get("name", "mlp")

    if model_name == "cifar_cnn":
        train_loader, test_loader = build_cifar10_loaders(cfg)
        model = CifarGraphNet(num_classes=int(cfg["model"]["num_classes"])).to(device)
        sample = next(iter(train_loader))[0][:2].to(device)
        validate_forward(model, sample)
    else:
        train_loader, test_loader = build_synthetic_loaders(cfg)
        model = StudentNet(**cfg["model"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    mutation_cfg = cfg.get("mutation") or {}
    mutation_enabled = bool(mutation_cfg.get("enabled", False))
    once_after = mutation_cfg.get("once_after_epoch")
    widen_delta = int(mutation_cfg.get("widen_delta", 32))
    mutation_applied = False
    mutation_log_jsonl = mutation_cfg.get("log_jsonl")

    log_csv = train_cfg.get("log_csv")
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_name = cfg["experiment"]["name"]

    epochs = int(train_cfg["epochs"])
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, device, train_loader
        )
        val_loss, val_acc = evaluate(model, device, test_loader)

        if (
            mutation_enabled
            and not mutation_applied
            and once_after is not None
            and epoch == int(once_after)
        ):
            params_before = count_trainable_parameters(model)
            target_id = first_linear_node_id(model)
            if target_id is None:
                raise RuntimeError("mutation enabled but no Linear layer found in graph")
            in_f, out_before = linear_layer_shapes(model, target_id)
            edge_widen(model, target_node_id=target_id, delta=widen_delta)
            optimizer = refresh_optimizer(optimizer, model)
            _, out_after = linear_layer_shapes(model, target_id)
            params_after = count_trainable_parameters(model)
            mutation_applied = True
            print(
                f"[mutation] Applied edge_widen({target_id}, delta={widen_delta}) after epoch {epoch}; "
                f"params {params_before} -> {params_after}; optimizer refreshed."
            )
            if mutation_log_jsonl:
                append_mutation_jsonl(
                    mutation_log_jsonl,
                    {
                        "event": "mutation",
                        "op": "edge_widen",
                        "run_id": f"{experiment_name}_{run_ts}",
                        "experiment": experiment_name,
                        "epoch_completed": epoch,
                        "target_node_id": target_id,
                        "delta": widen_delta,
                        "target_linear_in": in_f,
                        "target_linear_out_before": out_before,
                        "target_linear_out_after": out_after,
                        "num_parameters_before": params_before,
                        "num_parameters_after": params_after,
                    },
                )

        line = (
            f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )
        if mutation_applied:
            line += " | mutation=on"
        print(line)

        n_params = count_trainable_parameters(model)
        if log_csv:
            _append_metrics_csv(
                Path(log_csv),
                {
                    "utc_ts": run_ts,
                    "experiment": experiment_name,
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                    "num_parameters": str(n_params),
                    "mutation_applied_yet": str(mutation_applied),
                },
            )

    ck_path = Path("checkpoints") / f"{cfg['experiment']['name']}.pt"
    save_checkpoint(model, optimizer, str(ck_path))
    print(f"Saved checkpoint to {ck_path}")


if __name__ == "__main__":
    main()
