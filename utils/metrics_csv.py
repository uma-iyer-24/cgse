"""Shared per-epoch metrics CSV schema for train.py and evolution_train.py."""

import csv
from pathlib import Path

METRIC_FIELDS = [
    "utc_ts",
    "experiment",
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "num_parameters",
    "mutation_applied_yet",
    "critic_score",
]


def append_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
