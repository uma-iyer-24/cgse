#!/usr/bin/env python3
"""
Aggregate Tier 1 / Tier 1b CSV metrics into web/data/results.json and copy figures.

Usage (from repo root):
  python scripts/build_results_site.py
Then open web/index.html in a browser, or: cd web && python -m http.server 8765
"""
from __future__ import annotations

import csv
import json
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DATA = WEB / "data"
ASSETS = WEB / "assets" / "figures"
TIER1 = ROOT / "runs" / "tier1" / "metrics"
TIER1B = ROOT / "runs" / "tier1b" / "metrics"
FIG_SRC = ROOT / "paper_documentation" / "figures"

# filename stem (without _seedN) -> display label
TIER1_ARMS = [
    ("phase2_cifar_full_metrics", "Fixed (no mutation)"),
    ("phase2_cifar_full_mutate_metrics", "Scheduled widen"),
    ("phase3_cifar_kd_metrics", "Teacher + KD"),
    ("baseline_sear_ch_teacher_mutate_metrics", "Teacher + KD + widen"),
    ("phase2_cifar_full_cgse_metrics", "CGSE (critic)"),
]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    val_accs = [float(r["val_acc"]) for r in rows]
    train_accs = [float(r["train_acc"]) for r in rows]
    epochs = [int(r["epoch"]) for r in rows]
    params = int(float(rows[-1]["num_parameters"]))
    return {
        "final_val_acc": val_accs[-1],
        "best_val_acc": max(val_accs),
        "final_train_acc": train_accs[-1],
        "final_params": params,
        "num_epochs": len(rows),
        "curve": [
            {"epoch": int(r["epoch"]), "val_acc": float(r["val_acc"]), "train_acc": float(r["train_acc"])}
            for r in rows
        ],
    }


def _collect_tier1_arm(stem: str, label: str) -> dict | None:
    seeds = [41, 42, 43]
    per_seed = {}
    for s in seeds:
        p = TIER1 / f"{stem}_seed{s}.csv"
        if not p.exists():
            continue
        per_seed[str(s)] = _summarize_rows(_read_csv(p))
    if not per_seed:
        return None
    bests = [v["best_val_acc"] for v in per_seed.values()]
    finals = [v["final_val_acc"] for v in per_seed.values()]
    std_best = statistics.stdev(bests) if len(bests) > 1 else 0.0
    std_final = statistics.stdev(finals) if len(finals) > 1 else 0.0
    arm = {
        "id": stem.replace("_metrics", ""),
        "label": label,
        "seeds": sorted(int(k) for k in per_seed),
        "per_seed": per_seed,
        "best_val_acc_mean": statistics.mean(bests),
        "best_val_acc_std": std_best,
        "final_val_acc_mean": statistics.mean(finals),
        "final_val_acc_std": std_final,
    }
    return arm


def _collect_tier1b() -> dict:
    out = {"schedule": None, "critic": None}
    for key, stem in [
        ("schedule", "evolution_tier1b_schedule_metrics"),
        ("critic", "evolution_tier1b_critic_metrics"),
    ]:
        seeds_found = []
        per_seed = {}
        for p in sorted(TIER1B.glob(f"{stem}_seed*.csv")):
            # stem_seed41.csv
            part = p.stem.replace(f"{stem}_", "")
            if not part.startswith("seed"):
                continue
            s = part.replace("seed", "")
            rows = _read_csv(p)
            if not rows:
                continue
            per_seed[s] = _summarize_rows(rows)
            seeds_found.append(int(s))
        if per_seed:
            out[key] = {
                "stem": stem,
                "seeds": sorted(seeds_found),
                "per_seed": per_seed,
            }
    return out


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    tier1_arms = []
    for stem, label in TIER1_ARMS:
        arm = _collect_tier1_arm(stem, label)
        if arm:
            tier1_arms.append(arm)

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": "cgse",
        "tier1": {"arms": tier1_arms},
        "tier1b": _collect_tier1b(),
    }

    json_text = json.dumps(payload, indent=2)
    (DATA / "results.json").write_text(json_text, encoding="utf-8")
    print(f"Wrote {DATA / 'results.json'}")

    js_payload = "window.CGSE_RESULTS = " + json.dumps(payload) + ";\n"
    (WEB / "generated-config.js").write_text(js_payload, encoding="utf-8")
    print(f"Wrote {WEB / 'generated-config.js'}")

    if FIG_SRC.is_dir():
        for png in FIG_SRC.glob("*.png"):
            shutil.copy2(png, ASSETS / png.name)
            print(f"Copied figure {png.name}")


if __name__ == "__main__":
    main()
