# CGSE — Implementation & experiment log

**Purpose.** Single, paper-ready record of *what was built*, *why*, *how to reproduce it*, and *what training runs showed*. This document is updated as implementation proceeds.

**Audience.** Future you, collaborators, and the Methods / Experiments / Appendix sections of the paper.

**Conventions.**

- Dates use **ISO 8601** where relevant (`2026-04-02`).
- Code paths are relative to the repository root unless stated otherwise.
- Training commands are copy-pasteable; metrics and logs are under **[`runs/`](../runs/)** (repo root); excerpts are summarized here.
- When the **layout of the code** changes (new packages, renamed modules, new entry points), update **[`CGSE-codebase-guide.md`](CGSE-codebase-guide.md)** in the same breath so it stays the single file-by-file reference.

---

## Table of contents

1. [Project snapshot (for the paper)](#1-project-snapshot-for-the-paper) — includes [plain-language Phase 2 summary](#plain-language-phase-2)
2. [Repository map (implementation-relevant)](#2-repository-map-implementation-relevant)
3. [Phase status](#3-phase-status)
4. [Chronological implementation log](#4-chronological-implementation-log)
5. [Training runs registry](#5-training-runs-registry)
6. [Design decisions (persistent)](#6-design-decisions-persistent)
7. [Paper hooks (open items)](#7-paper-hooks-open-items)

---

## 1. Project snapshot (for the paper)

**Working name:** CGSE (context in `project-doc.pdf` — critic-guided or controller-guided structural evolution vs teacher-only baselines).

**Core idea.** Train a **student** network whose **graph structure can change during training** (e.g. widen linear layers, deepen via identity-initialized inserts). Long-term goal: a **controller** (CGSE) that scores bottlenecks from training signals and picks mutations under a budget, comparable to a **teacher-guided** SEArch-style baseline and **random** mutation controls.

**This codebase currently emphasizes:** Phase 1-style **safe mutations**, **graph validation**, **optimizer refresh** after structural edits, and Phase 2 **CIFAR-10 student training** with **train/val metrics**, optional **CSV logs**, and optional **in-loop widening** (`edge_widen` + optimizer refresh).

### Plain-language Phase 2

**Before:** The project could change a small **toy network** (random fake data, a few numbers in and out) and had scripts to test **widening** and **splitting** layers. That was the graph-surgery lab work.

**What we added (Phase 2):** We connected that idea to **real image data** so training looks like a normal small deep-learning experiment.

1. **Real dataset** — The trainer can load **CIFAR-10** (small 32×32 images, 10 classes). You can use the **full set** or only **part of it** so runs finish quickly while you experiment.
2. **A real student model** — **`CifarGraphNet`** is a **small convolutional network** (a few conv layers, then two **fully connected** layers). It is still built as a **mutable graph** so the same **widen** machinery can target those **linear** layers later.
3. **Honest training loop** — Each epoch reports **training loss/accuracy** and **test-set loss/accuracy** (not just one fake batch).
4. **Config-driven runs** — Settings live in YAML: how long to train, batch size, subset size, device, and so on. **`train.py`** reads that file. The default is the Phase 2 CIFAR setup; the old **random-data MLP** is still available via **`configs/base.yaml`**.
5. **Optional mutation during training** — In the config you can say: **after epoch X, widen the first linear layer once** and **refresh the optimizer** so training continues with the new parameters. That connects the old **mutation scripts** to **mutation inside real training**.
6. **Logging for the paper** — Metrics append to a **CSV** (including **parameter count** each epoch). Each **mutation** can append one **JSON line** (`mutation.log_jsonl`) with layer id, widths before/after, and param totals—good for timelines and figures. Artifacts live in **`runs/`** at the repo root (see [`runs/README.md`](../runs/README.md)); this file records **what changed and why**.
7. **Housekeeping** — **`requirements.txt`** lists dependencies; **`data/`** is git-ignored so large downloads are not committed.

**In one sentence:** We moved from “test structural edits on a toy net” to **train a small CIFAR student for real, log metrics, and optionally apply one widen mid-training**—with a written trail for Methods and results.

---

## 2. Repository map (implementation-relevant)

| Path | Role |
|------|------|
| `models/graph.py` | `GraphModule`, execution order, widen / insert helpers. |
| `models/student.py` | `StudentNet` (MLP-style student), `deepen_after`. |
| `ops/edge_split.py`, `ops/edge_widen.py` | Structural mutation operators (deepen/split, widen). |
| `utils/graph_validator.py` | Post-mutation shape / integrity checks. |
| `utils/optimizer_utils.py` | `refresh_optimizer` after new parameters appear. |
| `training/loop.py` | `train_one_epoch`, `evaluate` (loss + accuracy). |
| `training/data.py` | CIFAR-10 loaders, optional train/test subsets. |
| `training/synthetic.py` | Synthetic tensor dataset for Phase-0 MLP smoke tests. |
| `models/cifar_student.py` | `CifarGraphNet` — small CNN as sequential `GraphModule`. |
| `train.py` | CLI (`--config`, `--device`); Phase 2 + synthetic paths. |
| `configs/base.yaml` | Synthetic MLP defaults (`model` without `name` → MLP). |
| `configs/phase2_cifar.yaml` | Phase 2 defaults (CIFAR subset, MPS, optional mutation). |
| `configs/phase2_smoke.yaml` | Fast CPU smoke run. |
| `configs/phase2_cifar_full.yaml` | Full CIFAR-10 train/test, longer epochs (paper baseline). |
| `utils/model_info.py` | Trainable param count, first Linear id, shape helpers. |
| `utils/mutation_log.py` | Append **JSONL** mutation events. |
| `requirements.txt` | `torch`, `torchvision`, `PyYAML`. |
| `scripts/` | Validation and stress tests (`validate_mutation`, `test_*`). |
| `paper_documentation/` | This log, PDFs, long-form project docs (no training artifacts). |
| `runs/` | Training CSV / JSONL / `.log` outputs (repo root). |

---

## 3. Phase status

| Phase | Theme | Status | Notes |
|-------|--------|--------|--------|
| 0 | Env, config, minimal train, checkpointing | **Partial** | Synthetic loop exists; resume + config-in-checkpoint TBD. |
| 1 | Mutable graph, mutations, validation, live training | **Largely done** | Core ops + tests; API consolidation optional. |
| 2 | CIFAR (or subset), student baseline, mutations in-loop, logging | **In progress** | CIFAR-10 loaders, `CifarGraphNet`, metrics + CSV; optional `mutation` block in YAML. |
| 3+ | Teacher, KD, CGSE controller | **Not started** | Documented in `project-doc.pdf` / phase plan. |

---

## 4. Chronological implementation log

### 2026-04-02 — Documentation scaffold

- **Added** this file and `paper_documentation/README.md`, plus `runs/README.md` for artifact layout (under repo root).
- **Rationale.** Separate long-form PDFs (`project-doc.pdf`, `phase-plan-overview.pdf`) from an **append-only implementation record** suitable for Methods/Reproducibility.

### 2026-04-02 — Phase 2 (student training on CIFAR-10)

- **Data.** `training/data.py`: torchvision **CIFAR-10**, standard normalization and train-time augmentation (crop + flip), optional `subset_train` / `subset_test` for fast iteration.
- **Model.** `models/cifar_student.py`: **`CifarGraphNet`** — Conv-BN-ReLU-Pool stack + `Flatten` + two linear layers, registered as a sequential `GraphModule` (mutations target **Linear** nodes, same as Phase 1 ops).
- **Training.** `training/loop.py`: full-epoch **cross-entropy**, **train accuracy**, **test-set loss/accuracy** each epoch. `train.py`: `--config` (default `configs/phase2_cifar.yaml`), `--device` override, `utils/repro.set_seed`.
- **Logging.** Optional **`training.log_csv`**: fixed columns per epoch (`utc_ts`, `experiment`, losses, accuracies, **`num_parameters`**, `mutation_applied_yet`).
- **Mutation (Phase 2.4 slice).** YAML block `mutation`: `enabled`, `once_after_epoch`, `widen_delta`, optional **`log_jsonl`** — after the given epoch, **`edge_widen`** on the resolved **first Linear** (explicit `target_node_id` path) + **`refresh_optimizer`**. JSONL records `target_node_id`, `delta`, linear in/out before/after, `num_parameters_before` / `num_parameters_after`, `run_id`, `epoch_completed`.
- **Artifacts.** Checkpoints under `checkpoints/<experiment.name>.pt`. `.gitignore` includes `data/` for local CIFAR downloads.
- **Configs.** `configs/phase2_cifar.yaml` (default training), `configs/phase2_cifar_full.yaml` (full data, longer run), `configs/phase2_smoke.yaml`, `configs/phase2_smoke_mutate.yaml` (mutation + JSONL).

### 2026-04-02 — Clean slate + sentinel `paper_documentation/runs` (file)

- Stopped all **`train.py`** jobs; removed partial **overnight / duplicate** artifacts from repo-root **`runs/`** (kept **`runs/README.md`** only).
- Replaced recurring **`paper_documentation/runs/`** directory with a **plain file** `paper_documentation/runs` so tools cannot create that path as a folder.

### 2026-04-02 — `runs/` at repo root

- **Change.** Training artifacts (CSV, JSONL, `.log`) moved from `paper_documentation/runs/` to **`runs/`** at the repository root so **paper narrative** stays separate from **machine-generated outputs**.
- **Configs.** All `training.log_csv` and `mutation.log_jsonl` paths now use `runs/...`.
- **Guard (hard).** The path **`paper_documentation/runs`** is a **committed plain file**, not a directory, so `mkdir …/paper_documentation/runs` or “save to `paper_documentation/runs/foo`” **fails** instead of spawning a second layout. **`utils/run_paths.normalize_run_artifact_path`** + **`train.py`** still rewrite legacy **`paper_documentation/runs/...` strings in YAML** to **`runs/...`**. Shell redirects and editors must target **`runs/...`** at repo root.

### 2026-04-02 — Phase 2 logging hardening (mutation JSONL + full-data config)

- **`utils/mutation_log.append_mutation_jsonl`** — one JSON object per structural edit (append-only).
- **`utils/model_info`** — `count_trainable_parameters`, `first_linear_node_id`, `linear_layer_shapes` (used so logs name the actual widened layer).
- **`train.py`** — CSV schema stabilized; mutation path logs **params** before/after widen; optional **`mutation.log_jsonl`**.
- **`configs/phase2_cifar_full.yaml`** — `subset_train` / `subset_test` **null**, 50 epochs default (tune as needed for the paper).

### 2026-04-02 — Phase 2 baseline result (full CIFAR-10, no mutation)

- **What we trained.** `CifarGraphNet` (small CIFAR CNN implemented as a sequential `GraphModule`) on **full CIFAR-10** (50k train / 10k test) for **50 epochs**.
- **Supervision.** Standard cross-entropy only (**no teacher / no distillation**).
- **Mutations.** **Disabled** (`mutation.enabled: false`) → this run is the **fixed-architecture baseline** we will compare against later (mutation on, teacher-guided, CGSE).
- **Key metrics (epoch 49).** `train_acc ≈ 0.8663`, `test_acc (val_acc) ≈ 0.8454`, `num_parameters = 620,810`.
- **Artifacts.**
  - CSV: `runs/phase2_cifar_full_metrics.csv`
  - Console log: `runs/train_phase2_cifar_full.log`
  - Checkpoint saved locally: `checkpoints/cgse_phase2_cifar_full.pt` (large binary; intentionally not committed)

**Interpretation.** This establishes that the Phase 2 training pipeline is correct on real data and provides a reproducible **baseline accuracy curve**; any later “self-evolution” gains must be measured relative to this run under matched data/compute settings.

### 2026-04-02 — Phase 2 mutation ablation config (full CIFAR)

- **Added** `configs/phase2_cifar_full_mutate.yaml`: same data/hyperparams/seed as the baseline, but **`mutation.enabled: true`** with **`once_after_epoch: 10`**, **`widen_delta: 32`**, **`edge_widen` on `fc1`**, optimizer refresh, and artifacts:
  - `runs/phase2_cifar_full_mutate_metrics.csv`
  - `runs/phase2_cifar_full_mutate_mutations.jsonl`
  - checkpoint `checkpoints/cgse_phase2_cifar_full_mutate.pt` (local, not committed)
- **Compare** final accuracy and training stability vs `runs/phase2_cifar_full_metrics.csv` (mutation off).

---

## 5. Training runs registry

Use one row per meaningful run (baseline, ablation, or production experiment). Paste a short excerpt in this doc; store full output in `runs/`.

| Run ID | Date | Config / command | Notes | Log file |
|--------|------|-------------------|-------|----------|
| phase2_cifar_full_baseline | 2026-04-02 | `python train.py --config configs/phase2_cifar_full.yaml` | Full CIFAR-10, 50 epochs, mutation off. Final `val_acc ≈ 0.8454`. | `runs/train_phase2_cifar_full.log` |
| phase2_cifar_full_mutate | — | `python train.py --config configs/phase2_cifar_full_mutate.yaml` | Same as baseline + one widen after epoch 10. **Run locally**; then add console log path and final metrics here. | `runs/phase2_cifar_full_mutate_metrics.csv` (CSV); capture `train_*.log` if you redirect stdout |

### Excerpt template (copy for each run)

```text
# Run: <id>
# Command: python train.py --config ...
# Commit / dirty: <git describe>
# Hardware: e.g. MPS / CUDA / CPU

<paste 10–40 representative lines: epoch, loss, accuracy, mutation events>
```

---

## 6. Design decisions (persistent)

| ID | Decision | Alternatives | Rationale | Date |
|----|----------|--------------|-----------|------|
| D0 | Implementation log in Markdown under `paper_documentation/` | Wiki only; scattered READMEs | Version-controlled, diffable, paper-friendly | 2026-04-02 |
| D1 | Phase 2 student = **small CIFAR CNN** as `GraphModule` (not MLP on flattened pixels) | MLP-only | Matches image benchmark; linear tail compatible with widen/split ops. | 2026-04-02 |
| D2 | **Test set** for “val” metrics | Held-out slice of train | Standard CIFAR protocol; simpler reproducibility. | 2026-04-02 |
| D3 | `train.py` default config = **`configs/phase2_cifar.yaml`** | `base.yaml` | Phase 2 is the active research track; synthetic remains via `--config configs/base.yaml`. | 2026-04-02 |
| D4 | Mutation events as **JSONL** (append-only) | Only CSV | CSV is per-epoch; JSONL captures discrete **structural** events with layer metadata for plots. | 2026-04-02 |

---

## 7. Paper hooks (open items)

Items to fill as experiments land:

- [x] **Dataset:** CIFAR-10, default recipe in `training/data.py` (normalize + train augment); subset knobs in YAML.
- [x] **Student architecture:** `CifarGraphNet` (document param count in paper from `sum(p.numel() for p in model.parameters())`).
- [ ] **Mutation schedule:** still **epoch-triggered** in config; replace with **signal-driven** when CGSE scoring lands.
- [x] **Structured mutation log:** JSONL schema via `mutation.log_jsonl` (see `utils/mutation_log.py`).
- [x] **Full-data baseline:** `configs/phase2_cifar_full.yaml` completed; artifacts in repo-root **`runs/`**.
- [ ] Comparison table: fixed vs random mutation vs teacher (Phase 3) vs CGSE (Phase 7).
- [ ] Seeds, wall-clock, and hardware for each reported result.

---

*End of current log. New sessions append new dated subsections under §4 and new rows under §5.*
