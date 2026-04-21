# CGSE — Implementation & experiment log

**Purpose.** Single, paper-ready record of *what was built*, *why*, *how to reproduce it*, and *what training runs showed*. This document is updated as implementation proceeds.

**Visual overview for newcomers:** [root `README.md`](../README.md) (primary goal, diagrams) and [`paper_documentation/figures/`](figures/) (PNG figures for slides).

**Audience.** Future you, collaborators, and the Methods / Experiments / Appendix sections of the paper.

**Conventions.**

- Dates use **ISO 8601** where relevant (`2026-04-02`).
- Code paths are relative to the repository root unless stated otherwise.
- Training commands are copy-pasteable; metrics and logs are under **[`runs/`](../runs/)** (repo root); excerpts are summarized here.
- When the **layout of the code** changes (new packages, renamed modules, new entry points), update **[`CGSE-codebase-guide.md`](CGSE-codebase-guide.md)** in the same breath so it stays the single file-by-file reference.
- For a **full narrative** of phases, steps, rationale, and file mapping, see **[`CGSE-detailed-phase-walkthrough.md`](CGSE-detailed-phase-walkthrough.md)**.

---

## Table of contents

1. [Project snapshot (for the paper)](#1-project-snapshot-for-the-paper) — includes [plain-language Phase 2 summary](#plain-language-phase-2); see also **[detailed phase walkthrough](CGSE-detailed-phase-walkthrough.md)**
2. [Repository map (implementation-relevant)](#2-repository-map-implementation-relevant)
3. [Phase status](#3-phase-status)
4. [Chronological implementation log](#4-chronological-implementation-log)
5. [Training runs registry](#5-training-runs-registry)
6. [Design decisions (persistent)](#6-design-decisions-persistent)
7. [Paper hooks (open items)](#7-paper-hooks-open-items)

---

## 1. Project snapshot (for the paper)

**Working name:** CGSE (context in `project-doc.pdf` — critic-guided or controller-guided structural evolution vs teacher-only baselines).

**Core idea.** Train a **student** whose **graph** can change during training. **Primary prior system to beat / compare against:** **SEArch** (Liang, Xiang & Li, *Neurocomputing* 2025, DOI 10.1016/j.neucom.2025.130980) — teacher-guided self-evolution with attention, modification scores, and edge splitting on CIFAR/ImageNet. **CGSE** keeps **iterative structural search** but **replaces the teacher with a critic** for evolution guidance (full argument in **`project-doc.pdf`**). **Experimental contrast in code:** (1) **teacher + KD** (+ optional mutate) = our **SEArch-style control**; (2) **critic, no teacher** = **CGSE** (`critic.enabled`, **`phase2_cifar_full_cgse.yaml`**). See **`SEArch-baseline-and-CGSE-evaluation-plan.md`** for parity tiers and tests.

**This codebase currently emphasizes:** Phase 1 **mutations** + validation + optimizer refresh; Phase 2 **CIFAR** + logging + scheduled or **critic-gated** mutations; Phase 3 **teacher + KD** + **`baseline_sear_ch_teacher_mutate`**; **CGSE** critic + REINFORCE in **`train.py`**.

### Plain-language Phase 2

**Before:** The project could change a small **toy network** (random fake data, a few numbers in and out) and had scripts to test **widening** and **splitting** layers. That was the graph-surgery lab work.

**What we added (Phase 2):** We connected that idea to **real image data** so training looks like a normal small deep-learning experiment.

1. **Real dataset** — The trainer can load **CIFAR-10** (small 32×32 images, 10 classes). You can use the **full set** or only **part of it** so runs finish quickly while you experiment.
2. **A real student model** — **`CifarGraphNet`** is a **small convolutional network** (a few conv layers, then two **fully connected** layers). It is still built as a **mutable graph** so the same **widen** machinery can target those **linear** layers later.
3. **Honest training loop** — Each epoch reports **training loss/accuracy** and **test-set loss/accuracy** (not just one fake batch).
4. **Config-driven runs** — Settings live in YAML: how long to train, batch size, subset size, device, and so on. **`train.py`** reads that file. The default is the Phase 2 CIFAR setup; the old **random-data MLP** is still available via **`configs/synthetic/base.yaml`**.
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
| `training/loop.py` | `train_one_epoch`, `evaluate` (loss + accuracy); optional **KD** vs frozen teacher. |
| `training/data.py` | CIFAR-10 loaders, optional train/test subsets. |
| `training/synthetic.py` | Synthetic tensor dataset for Phase-0 MLP smoke tests. |
| `models/cifar_student.py` | `CifarGraphNet` — small CNN as sequential `GraphModule`. |
| `train.py` | CLI (`--config`, `--device`); CIFAR + synthetic; optional **teacher/KD** (Phase 3). |
| `utils/checkpoint.py` | `save_checkpoint`, `load_model_weights`. |
| `configs/synthetic/base.yaml` | Synthetic MLP defaults (`model` without `name` → MLP). |
| `configs/cifar/phase2_cifar.yaml` | Phase 2 defaults (CIFAR subset, MPS, optional mutation). |
| `configs/cifar/smoke/phase2_smoke.yaml` | Fast CPU smoke run. |
| `configs/cifar/phase2_cifar_full.yaml` | Full CIFAR-10 train/test, longer epochs (paper baseline). |
| `configs/cifar/phase3_cifar_kd.yaml` | Teacher + KD, fixed arch (SEArch-style signal, no mutation). |
| `configs/cifar/smoke/phase3_cifar_kd_smoke.yaml` | Tiny subset KD smoke (needs teacher `.pt`). |
| `configs/cifar/baseline_sear_ch_teacher_mutate.yaml` | **Control:** teacher + KD + widen (same schedule as full mutate). |
| `critics/critic.py` | **`StructuralCritic`** — CGSE module (replaces teacher for structural decisions; training TBD). |
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
| 3 | SEArch-style **teacher + KD** (control) | **Started** | `teacher` YAML, `load_model_weights`, CE+KD in `training/loop.py`; `phase3_cifar_kd*.yaml`, **`baseline_sear_ch_teacher_mutate.yaml`**. |
| 4 | **CGSE critic** (replaces teacher for mutations) | **Started** | **`train.py`**: critic window, ε-greedy + Bernoulli mutate, REINFORCE on Δval after widen; **`critics/state_features.py`**; configs **`phase2_cifar_full_cgse.yaml`**, **`phase2_cifar_cgse_smoke.yaml`**. |

---

## 4. Chronological implementation log

### 2026-04-02 — Documentation scaffold

- **Added** this file and `paper_documentation/README.md`, plus `runs/README.md` for artifact layout (under repo root).
- **Rationale.** Separate long-form PDFs (`project-doc.pdf`, `phase-plan-overview.pdf`) from an **append-only implementation record** suitable for Methods/Reproducibility.

### 2026-04-02 — Phase 2 (student training on CIFAR-10)

- **Data.** `training/data.py`: torchvision **CIFAR-10**, standard normalization and train-time augmentation (crop + flip), optional `subset_train` / `subset_test` for fast iteration.
- **Model.** `models/cifar_student.py`: **`CifarGraphNet`** — Conv-BN-ReLU-Pool stack + `Flatten` + two linear layers, registered as a sequential `GraphModule` (mutations target **Linear** nodes, same as Phase 1 ops).
- **Training.** `training/loop.py`: full-epoch **cross-entropy**, **train accuracy**, **test-set loss/accuracy** each epoch. `train.py`: `--config` (default `configs/cifar/phase2_cifar.yaml`), `--device` override, `utils/repro.set_seed`.
- **Logging.** Optional **`training.log_csv`**: fixed columns per epoch (`utc_ts`, `experiment`, losses, accuracies, **`num_parameters`**, `mutation_applied_yet`).
- **Mutation (Phase 2.4 slice).** YAML block `mutation`: `enabled`, `once_after_epoch`, `widen_delta`, optional **`log_jsonl`** — after the given epoch, **`edge_widen`** on the resolved **first Linear** (explicit `target_node_id` path) + **`refresh_optimizer`**. JSONL records `target_node_id`, `delta`, linear in/out before/after, `num_parameters_before` / `num_parameters_after`, `run_id`, `epoch_completed`.
- **Artifacts.** Checkpoints under `checkpoints/<experiment.name>.pt`. `.gitignore` includes `data/` for local CIFAR downloads.
- **Configs.** `configs/cifar/phase2_cifar.yaml` (default training), `configs/cifar/phase2_cifar_full.yaml` (full data, longer run), `configs/cifar/smoke/phase2_smoke.yaml`, `configs/cifar/smoke/phase2_smoke_mutate.yaml` (mutation + JSONL).

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
- **`configs/cifar/phase2_cifar_full.yaml`** — `subset_train` / `subset_test` **null**, 50 epochs default (tune as needed for the paper).

### 2026-04-02 — Phase 2 baseline result (full CIFAR-10, no mutation)

- **What we trained.** `CifarGraphNet` (small CIFAR CNN implemented as a sequential `GraphModule`) on **full CIFAR-10** (50k train / 10k test) for **50 epochs**.
- **Supervision.** Standard cross-entropy only (**no teacher / no distillation**).
- **Mutations.** **Disabled** (`mutation.enabled: false`) → this run is the **fixed-architecture baseline** we will compare against later (mutation on, teacher-guided, CGSE).
- **Key metrics (epoch 49).** `train_acc ≈ 0.8663`, `test_acc (val_acc) ≈ 0.8454`, `num_parameters = 620,810`.
- **Artifacts.**
  - CSV: `runs/metrics/phase2_cifar_full_metrics.csv`
  - Console log: `runs/logs/train_phase2_cifar_full.log`
  - Checkpoint saved locally: `checkpoints/cgse_phase2_cifar_full.pt` (large binary; intentionally not committed)

**Interpretation.** This establishes that the Phase 2 training pipeline is correct on real data and provides a reproducible **baseline accuracy curve**; any later “self-evolution” gains must be measured relative to this run under matched data/compute settings.

### 2026-04-04 — Mutation ops: new `Linear` modules match device/dtype (MPS/CUDA fix)

- **Bug.** `nn.Linear(...)` defaults to **CPU**. After **`edge_widen`** on a model on **MPS** (or CUDA), the widened and resized layers stayed on CPU → `RuntimeError: Tensor for argument weight is on cpu but expected on mps` on the next forward pass (see `runs/logs/train_phase2_cifar_full_mutate.log`).
- **Fix.** **`ops/edge_widen.py`** builds replacement layers with **`.to(device=…, dtype=…)`** from the target/downstream linears; downstream bias copy uses **`copy_`** instead of replacing the `Parameter`. The same pattern is applied in **`models/graph.py`** (`widen_node`) and **`ops/edge_split.py`** (inserted identity linear).

### 2026-04-04 — Phase 3 (SEArch-style teacher + KD) and CGSE scope

- **Narrow comparison.** Repo targets **teacher-guided KD** (with optional mutation) as the **control** vs **CGSE**, where a **critic replaces the teacher** for **mutation** decisions only (label CE unchanged). Other manuscript brainstorms (multi-objective arbitration, hybrids, etc.) are **out of scope** here.
- **`utils/checkpoint.load_model_weights`** — load `model` state dict from checkpoints written by `save_checkpoint`.
- **`training/loop.py`** — `kd_distillation_loss`; **`train_one_epoch`** accepts optional frozen **`teacher`**, **`kd_temperature`**, **`kd_alpha`**; training loss = \((1-\alpha)\) CE + \(\alpha\) \(T^2\) KL(soft student \(\|\) soft teacher).
- **`train.py`** — if **`teacher.enabled`** and **`model.name: cifar_cnn`**, builds a second **`CifarGraphNet`**, loads **`teacher.checkpoint`**, freezes parameters, runs KD during training. Val metrics remain standard CE / accuracy on the **student**.
- **Configs.** `configs/cifar/phase3_cifar_kd.yaml` (teacher + KD, **no** mutation); `configs/cifar/smoke/phase3_cifar_kd_smoke.yaml` (subset); **`configs/cifar/baseline_sear_ch_teacher_mutate.yaml`** (teacher + KD + widen — **SEArch-style control** vs future critic). **Prerequisite:** compatible **`CifarGraphNet`** `.pt` for **`teacher.checkpoint`**.
- **`critics/critic.py`**, **`critics/__init__.py`** — **`StructuralCritic`** stub for CGSE (**next:** train + gate mutations in **`train.py`**).
- **README** — experimental design table (teacher arm vs CGSE arm).

### 2026-04-05 — Tier 1b sweep status + Results prose + Pages workflow

- **Tier 1b:** **Schedule seed 42** was **still training** (partial CSV, fewer than 50 epochs); **critic seeds 42–43** had **not** started yet (script order: schedule then critic per seed). **Seed 41** complete for **both** arms — see registry rows and **`draft-results-for-paper.md`**. When the background sweep finishes, run **`python scripts/build_results_site.py`** and commit new `runs/tier1b/**` artifacts.
- **`draft-results-for-paper.md`** — copy-paste **Results** paragraphs (Tier 1 summary stats, Tier 1b seed-41 detail, reproducibility table stub).
- **`web/`** — Tier 1b **status banner** + **table** (complete vs incomplete seeds); chart uses **first seed complete for both arms**.
- **GitHub Pages:** workflow **`.github/workflows/deploy-results-site.yml`** (enable Pages → GitHub Actions in repo settings).

### Run environment (fill for paper)

| Field | Notes |
|-------|--------|
| Device | Config **`device: auto`** → CUDA if available, else MPS, else CPU. **Record actual device** from your training console when publishing. |
| Tier 1 sweep | `scripts/run_tier1.sh` with `DEVICE=auto` (see `runs/tier1/logs/tier1_master.log`). |
| Tier 1b | `scripts/run_tier1b.sh`; ~**1 h per 50-epoch job** per arm on a typical **MPS/CUDA** laptop (measure locally). |

### 2026-04-04 — Results preview site + Tier 1 aggregate

- **`web/`** — static **results preview**: Tier 1 table + bar chart + per-arm learning curves (Chart.js), Tier 1b overlay when CSVs exist, embedded concept PNGs. Build: `python scripts/build_results_site.py`. Documented in **`web/README.md`** and linked from root **`README.md`**.
- **Tier 1 (seeds 41–43)** — mean **best val_acc** over seeds summarized in registry row **tier1_five_arm_grid** and in **`CGSE-experiments-and-results-guide.md` §14.
- **`origin/main`** — pushed from **`phase1-graph`** so a default non-feature branch exists for future PRs (GitHub default branch can be switched to **`main`** in repo settings).

### 2026-04-04 — Tier 1b seeds 42–43 (follow-up)

- **`SEEDS="42 43" bash scripts/run_tier1b.sh`** — same as seed 41 protocol; artifacts will land as `runs/tier1b/metrics/evolution_tier1b_*_metrics_seed{42,43}.csv` and logs under `runs/tier1b/logs/`. Re-run **`build_results_site.py`** after completion to refresh the preview.

### 2026-04-04 — Tier 1b full CIFAR (schedule vs critic, seed 41)

- **Command.** `SEEDS=41 bash scripts/run_tier1b.sh` (see **`runs/README.md`**). Default script also runs seeds **42–43**; re-run with `SEEDS="41 42 43"` for the full sweep.
- **Schedule arm** (`configs/evolution/evolution_tier1b_schedule.yaml`): fixed **widen_conv3 → widen_fc1 → split_before_fc2**; metrics `runs/tier1b/metrics/evolution_tier1b_schedule_metrics_seed41.csv`; log `runs/tier1b/logs/tier1b_schedule_seed41.log`.
- **Critic arm** (`configs/evolution/evolution_tier1b_critic.yaml`): discrete critic, same stage budget; mutations `runs/tier1b/mutations/evolution_tier1b_critic_mutations_seed41.jsonl`; metrics `runs/tier1b/metrics/evolution_tier1b_critic_metrics_seed41.csv`; log `runs/tier1b/logs/tier1b_critic_seed41.log`.
- **Checkpoints** under `checkpoints/tier1b/` match `.gitignore` (`checkpoints/*/cgse_*.pt`); not committed.

### 2026-04-04 — Tier 1b roadmap (docs only; parallel with Tier 1 training)

- **Added** **[`SEArch-baseline-and-CGSE-evaluation-plan.md`](SEArch-baseline-and-CGSE-evaluation-plan.md) §7** — SEArch-like **multi-stage** protocol, **budget**, **2–3 operators** (widen / deepen / conv widen), **critic v2** over **(site × op)**, implementation checklist §7.4.
- **Cross-links** from root **`README.md`**, **`runs/README.md`** (develop on a **git branch** while **`nohup` Tier 1** runs; running `train.py` does not hot-reload code), **`paper_documentation/README.md`**, **`CGSE-codebase-guide.md`**, paper hooks §7.
- **Rationale.** Tier 1 (single widen, five arms) stays the first results table; Tier 1b is the **next** code + experiment wave without cancelling in-flight sweeps.

### 2026-04-04 — CGSE critic wired in `train.py` (mutation gating + REINFORCE)

- **`critics/state_features.py`** — **`STATE_DIM=8`**, **`build_critic_state`** (train/val losses and accuracies, epoch progress, param scale, deltas, bias).
- **`train.py`** — YAML **`critic:`** (`enabled`, `hidden_dim`, `lr`, `epsilon`, `window_start_epoch`, `window_end_epoch`, `force_mutate_end_of_window`). **Cannot** combine **`critic.enabled`** with **`teacher.enabled`**. When critic is on, **`once_after_epoch`** is ignored for timing; **`edge_widen`** runs when Bernoulli(σ(logit)) samples mutate (ε-greedy exploration) or **`force_mutate_end_of_window`** fires. **Policy gradient:** next epoch, \(-\log\sigma(\text{logit}) \cdot (\text{val\_acc}_{t+1}-\text{val\_acc}_t)\)** unless the action was random explore or forced. JSONL mutation rows include **`gate`: `critic`** or **`schedule`**. Checkpoints: **`checkpoints/<name>_critic.pt`**. CSV column **`critic_score`**.
- **Configs.** **`configs/cifar/phase2_cifar_full_cgse.yaml`** (critic window **epoch 10 only**, parity with **`phase2_cifar_full_mutate`** timing), **`configs/cifar/smoke/phase2_cifar_cgse_smoke.yaml`**.
- **`train.py --seed N`** — overrides RNG; appends **`_seedN`** to **`experiment.name`** and to **`log_csv`** / **`mutation.log_jsonl`** stems; checkpoints use the suffixed name. Tier 1 recipe: **`runs/README.md`**.

### 2026-04-02 — Phase 2 mutation ablation config (full CIFAR)

- **Added** `configs/cifar/phase2_cifar_full_mutate.yaml`: same data/hyperparams/seed as the baseline, but **`mutation.enabled: true`** with **`once_after_epoch: 10`**, **`widen_delta: 32`**, **`edge_widen` on `fc1`**, optimizer refresh, and artifacts:
  - `runs/metrics/phase2_cifar_full_mutate_metrics.csv`
  - `runs/mutations/phase2_cifar_full_mutate_mutations.jsonl`
  - checkpoint `checkpoints/cgse_phase2_cifar_full_mutate.pt` (local, not committed)
- **Compare** final accuracy and training stability vs `runs/metrics/phase2_cifar_full_metrics.csv` (mutation off).

---

## 5. Training runs registry

Use one row per meaningful run (baseline, ablation, or production experiment). Paste a short excerpt in this doc; store full output in `runs/`.

### Landing work / pull requests

As of **2026-04-04**, GitHub **`origin`** exposes a single branch, **`phase1-graph`** (default). There is no separate **`main`** yet, so there is nothing to open a PR *into*. When you add **`main`** (or another release branch) and set it as the default, open **`phase1-graph` → `main`** and merge there.

### Registry table

| Run ID | Date | Config / command | Notes | Log file |
|--------|------|-------------------|-------|----------|
| phase2_cifar_full_baseline | 2026-04-02 | `python train.py --config configs/cifar/phase2_cifar_full.yaml` | Full CIFAR-10, 50 epochs, mutation off. Final `val_acc ≈ 0.8454`. | `runs/tier1/logs/train_phase2_cifar_full.log` |
| phase2_cifar_full_mutate | 2026-04-03 | `python train.py --config configs/cifar/phase2_cifar_full_mutate.yaml` | Same as baseline + one widen after epoch 10 (device fix 2026-04-04). CSV records **epochs 0–44** (45 epochs); last row `val_acc ≈ 0.8379`, `num_parameters = 686698` after widen. Re-run for full 50 if needed. | `runs/tier1/metrics/phase2_cifar_full_mutate_metrics.csv`; `runs/tier1/logs/train_phase2_cifar_full_mutate.log` |
| phase3_cifar_kd | — | `python train.py --config configs/cifar/phase3_cifar_kd.yaml` | Teacher + KD, fixed arch. Fill metrics after run. | `runs/tier1/metrics/phase3_cifar_kd_metrics_seed41.csv` (multi-seed); see `runs/README.md` |
| baseline_sear_ch_teacher_mutate | — | `python train.py --config configs/cifar/baseline_sear_ch_teacher_mutate.yaml` | **SEArch control:** teacher + KD + widen after epoch 10. | `runs/tier1/metrics/baseline_sear_ch_teacher_mutate_metrics_seed41.csv` (multi-seed) |
| evolution_tier1b_schedule_seed41 | 2026-04-04 | `SEEDS=41 bash scripts/run_tier1b.sh` (schedule arm) | Full CIFAR-10, 5×10 epochs, fixed schedule: **widen_conv3** → **widen_fc1** → **split_before_fc2**. Final epoch `val_acc ≈ 0.8133`; best in CSV epoch 47 `val_acc ≈ 0.8417`; `num_parameters = 935914`. Checkpoints under `checkpoints/tier1b/` (gitignored). | `runs/tier1b/metrics/evolution_tier1b_schedule_metrics_seed41.csv`; `runs/tier1b/logs/tier1b_schedule_seed41.log` |
| evolution_tier1b_critic_seed41 | 2026-04-04 | same script (critic arm) | Same budget as schedule; **discrete critic** picks ops (ε=0.25). Ops applied: **split_before_fc2** @ 9 → **widen_conv3** @ 19 → **widen_fc1** @ 29 (see JSONL). Final `val_acc ≈ 0.8399`; `num_parameters = 926346`. | `runs/tier1b/metrics/evolution_tier1b_critic_metrics_seed41.csv`; `runs/tier1b/logs/tier1b_critic_seed41.log` |
| tier1_five_arm_grid | 2026-04-04 | `scripts/run_tier1.sh` / `train.py --seed N` | **Five arms × seeds 41–43** (full CIFAR-10, matched hyperparameters). **Mean best val_acc** over seeds: Fixed **0.8448 ± 0.0014**; Scheduled widen **0.8438 ± 0.0016**; Teacher + KD **0.8487 ± 0.0021**; Teacher + KD + widen **0.8495 ± 0.0034**; CGSE **0.8468 ± 0.0041**. Tangible summary: regenerate `web/` via `python scripts/build_results_site.py` and open `web/index.html`. | `runs/tier1/metrics/*_metrics_seed{41,42,43}.csv` |
| tier2_paper_seed42 | 2026-04-19 | `SEEDS="42" DEVICE=auto bash scripts/run_tier2.sh` | Tier 2 paper-ish parity track (ResNet-56 teacher, ResNet-20 student, SGD+multistep, 50 epochs). Highlights (best val_acc): teacher **92.38%**; student CE **91.18%**; student KD **91.66%** (**teacher_forwards=19550**); CGSE multi-op (teacher-free) **90.85%** (**teacher_forwards=0**). | `runs_paper/tier2/metrics/*.csv`; logs under `runs_paper/tier2/logs/` |

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
| D3 | `train.py` default config = **`configs/cifar/phase2_cifar.yaml`** | `base.yaml` | Phase 2 is the active research track; synthetic remains via `--config configs/synthetic/base.yaml`. | 2026-04-02 |
| D4 | Mutation events as **JSONL** (append-only) | Only CSV | CSV is per-epoch; JSONL captures discrete **structural** events with layer metadata for plots. | 2026-04-02 |
| D5 | Phase 3 teacher = **same class** as student (`CifarGraphNet`), load `.pt` | Wider separate teacher CNN | Reuses baseline checkpoint; KD logits align without extra bridge layers. | 2026-04-04 |
| D6 | **Implementation scope = teacher vs critic only** | Implement every manuscript variant (AMOG, hybrids, …) | Keeps the scientific story one-dimensional: SEArch-style **teacher KD** vs **CGSE critic** replacing the teacher for mutations. | 2026-04-04 |

---

## 7. Paper hooks (open items)

Items to fill as experiments land:

- [x] **Dataset:** CIFAR-10, default recipe in `training/data.py` (normalize + train augment); subset knobs in YAML.
- [x] **Student architecture:** `CifarGraphNet` (document param count in paper from `sum(p.numel() for p in model.parameters())`).
- [x] **Mutation schedule:** YAML **`once_after_epoch`** (schedule) or **`critic:`** window + policy (CGSE); richer “signals” / targets still open.
- [x] **Structured mutation log:** JSONL schema via `mutation.log_jsonl` (see `utils/mutation_log.py`).
- [x] **Full-data baseline:** `configs/cifar/phase2_cifar_full.yaml` completed; artifacts in repo-root **`runs/`**.
- [ ] **Prior art citation:** Liang et al. (2025) SEArch (Neurocomputing) + honest **operator/budget** gap vs our `CifarGraphNet` + widen/split (see **`SEArch-baseline-and-CGSE-evaluation-plan.md`**).
- [x] **Tier-1 experiment grid:** fixed / mutate-only / teacher+KD / teacher+KD+mutate / **CGSE** (critic); **seeds 41–43** complete; see registry row **tier1_five_arm_grid** and `web/` preview.
- [ ] **Tier-2 (optional parity):** ResNet-56 teacher, ~0.27M student, SGD long retrain — align with SEArch Table 5 setting where feasible.
- [ ] **Tier 1b / §7 plan (partial):** **Seed 41** schedule + critic **complete**; **seeds 42–43** in progress or pending (`scripts/run_tier1b.sh`). **Next:** finish multi-seed, then optional **teacher** arm with same candidate set — see **`SEArch-baseline-and-CGSE-evaluation-plan.md` §7**.
- [x] **Pytest:** `tests/test_graph_ops.py` — KD formula, checkpoint round-trip, `edge_widen` / `edge_split` on CPU (+ CUDA/MPS when available), one-batch teacher+KD smoke (plan §4.1).
- [ ] Seeds, wall-clock, and hardware for each reported result.

---

*End of current log. New sessions append new dated subsections under §4 and new rows under §5.*
