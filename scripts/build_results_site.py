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
TIER2 = ROOT / "runs" / "tier2" / "metrics"
FIG_SRC = ROOT / "paper_documentation" / "figures"

# Tier 1b full runs: epochs 0..49 → 50 rows
TIER1B_EXPECTED_EPOCHS = 50

# filename stem (without _seedN) -> display label
TIER1_ARMS = [
    ("phase2_cifar_full_metrics", "Fixed (no mutation)"),
    ("phase2_cifar_full_mutate_metrics", "Scheduled widen"),
    ("phase3_cifar_kd_metrics", "Teacher + KD"),
    ("baseline_sear_ch_teacher_mutate_metrics", "Teacher + KD + widen"),
    ("phase2_cifar_full_cgse_metrics", "CGSE (critic)"),
]

TIER2DEV_ROWS = [
    ("tier2dev_teacher_resnet56_metrics", "Teacher (ResNet-56)"),
    ("tier2dev_student_resnet20_ce_metrics", "Student CE (ResNet-20)"),
    ("tier2dev_student_resnet20_kd_metrics", "Student KD (ResNet-20)"),
    ("tier2dev_student_resnet20_cgse_multiop_metrics", "CGSE multi-op (ResNet-20)"),
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
    last = rows[-1]
    def _f(key: str, default: float = 0.0) -> float:
        try:
            v = last.get(key, "")
            return float(v) if v not in ("", None) else default
        except Exception:
            return default

    def _i(key: str) -> int:
        try:
            v = last.get(key, "")
            return int(float(v)) if v not in ("", None) else 0
        except Exception:
            return 0

    return {
        "final_val_acc": val_accs[-1],
        "best_val_acc": max(val_accs),
        "final_train_acc": train_accs[-1],
        "final_params": params,
        "num_epochs": len(rows),
        # Efficiency fields (if present)
        "wall_seconds": _f("wall_seconds", 0.0),
        "teacher_forwards": _i("teacher_forwards"),
        "train_steps": _i("train_steps"),
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


def _tier1b_csv_complete(rows: list[dict]) -> bool:
    if len(rows) < TIER1B_EXPECTED_EPOCHS:
        return False
    epochs = [int(r["epoch"]) for r in rows]
    return max(epochs) >= TIER1B_EXPECTED_EPOCHS - 1


def _collect_tier1b() -> dict:
    arms_out: dict[str, dict] = {}
    status_bits: list[str] = []

    for key, stem, label in [
        ("schedule", "evolution_tier1b_schedule_metrics", "Fixed schedule"),
        ("critic", "evolution_tier1b_critic_metrics", "Discrete critic"),
    ]:
        all_seeds: list[int] = []
        complete: dict[str, dict] = {}
        incomplete: dict[str, dict] = {}
        for p in sorted(TIER1B.glob(f"{stem}_seed*.csv")):
            part = p.stem.replace(f"{stem}_", "")
            if not part.startswith("seed"):
                continue
            s = part.replace("seed", "")
            rows = _read_csv(p)
            if not rows:
                continue
            summ = _summarize_rows(rows)
            summ["complete"] = _tier1b_csv_complete(rows)
            all_seeds.append(int(s))
            if summ["complete"]:
                complete[s] = summ
            else:
                incomplete[s] = summ

        if not all_seeds:
            arms_out[key] = None
            continue

        bests = [v["best_val_acc"] for v in complete.values()]
        finals = [v["final_val_acc"] for v in complete.values()]
        agg = None
        if bests:
            agg = {
                "n_complete": len(bests),
                "best_val_acc_mean": statistics.mean(bests),
                "best_val_acc_std": statistics.stdev(bests) if len(bests) > 1 else 0.0,
                "final_val_acc_mean": statistics.mean(finals),
                "final_val_acc_std": statistics.stdev(finals) if len(finals) > 1 else 0.0,
            }

        arms_out[key] = {
            "label": label,
            "stem": stem,
            "seeds_all": sorted(all_seeds),
            "seeds_complete": sorted(int(s) for s in complete),
            "seeds_incomplete": sorted(int(s) for s in incomplete),
            "complete": complete,
            "incomplete": incomplete,
            "aggregate": agg,
        }

        inc = arms_out[key]["seeds_incomplete"]
        if inc:
            status_bits.append(
                f"{label}: incomplete seeds {inc} (CSV has fewer than {TIER1B_EXPECTED_EPOCHS} epochs)"
            )

    has_any = arms_out.get("schedule") is not None or arms_out.get("critic") is not None
    if not has_any:
        return {
            "arms": arms_out,
            "status_html": "Tier 1b: no seed CSVs found. Run scripts/run_tier1b.sh.",
        }

    if not status_bits:
        sc = arms_out.get("schedule") or {}
        cr = arms_out.get("critic") or {}
        nc = len(sc.get("seeds_complete") or [])
        n2 = len(cr.get("seeds_complete") or [])
        if nc and n2:
            status_bits.append(
                f"Tier 1b: {nc} complete schedule run(s), {n2} complete critic run(s)."
            )
        elif nc or n2:
            status_bits.append("Tier 1b: partial data — see table below.")

    return {
        "arms": arms_out,
        "status_html": " ".join(status_bits) if status_bits else "Tier 1b: see table.",
    }


def _collect_tier2dev() -> dict:
    # Dev sweeps are usually 1 seed.
    seeds = [41, 42, 43]
    rows_out = []
    for stem, label in TIER2DEV_ROWS:
        per_seed = {}
        for s in seeds:
            p = TIER2 / f"{stem}_seed{s}.csv"
            if not p.exists():
                continue
            per_seed[str(s)] = _summarize_rows(_read_csv(p))
        if not per_seed:
            continue
        # choose the lowest seed for display by default
        seed_pick = sorted(int(k) for k in per_seed.keys())[0]
        rows_out.append(
            {
                "id": stem.replace("_metrics", ""),
                "label": label,
                "seeds": sorted(int(k) for k in per_seed.keys()),
                "per_seed": per_seed,
                "default_seed": seed_pick,
            }
        )
    return {"rows": rows_out}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    tier1_arms = []
    for stem, label in TIER1_ARMS:
        arm = _collect_tier1_arm(stem, label)
        if arm:
            tier1_arms.append(arm)

    t1b = _collect_tier1b()
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": "cgse",
        "tier1": {"arms": tier1_arms},
        "tier1b": t1b["arms"],
        "tier1b_status": t1b["status_html"],
        "tier2dev": _collect_tier2dev(),
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
