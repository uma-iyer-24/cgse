# CGSE — Implementation & experiment log

**Purpose.** Single, paper-ready record of *what was built*, *why*, *how to reproduce it*, and *what training runs showed*. This document is updated as implementation proceeds.

**Visual overview for newcomers:** [root `README.md`](../README.md) (primary goal, diagrams) and [`paper_documentation/figures/`](figures/) (PNG figures for slides).

**Math reference:** every equation that appears in code is numbered and cross-referenced in [`CGSE-math-and-equations.md`](CGSE-math-and-equations.md) — that file is the source of truth for paper equation numbers.

**Technical deep-dive.** For the *how* behind every module of the SEArch + CGSE-on-SEArch system (channel-attention KD internals, candidate enumeration safety filters, function-preserving init, the unified outer loop, the per-candidate critic, student probe internals, REINFORCE with EMA baseline, optimizer-state preservation across mutations, YAML → code-path mapping, CSV / JSONL schemas, performance characteristics, gotchas, extension points), see [`CGSE-implementation-details.md`](CGSE-implementation-details.md).

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

### 2026-04-28 — Paper-faithful SEArch + CGSE-on-SEArch (Tier 2)

**Goal.** Replace the prior schedule-style Tier-2 arms (single-op `resnet_layer3_widen`, `resnet_head_widen`, scheduled and CGSE-gated variants) with a **paper-faithful SEArch implementation** and a **CGSE arm built on top of the same SEArch outer loop** — so the two arms differ in *exactly one variable*: the supervision signal that produces the modification value.

**Conceptual model adopted.**

- **Teacher (SEArch).** A frozen ResNet-56 doing **two** jobs: (1) per-batch channel-attention KD that supervises the student's *weights* (Eqs. 1–4); (2) per-stage modification-value scoring that picks the *next structural edit* (Eq. 5). Both jobs depend on teacher intermediate features. Cost: ≈ one teacher forward per training batch + a few per stage end → ~22k teacher forwards over a 50-epoch run.
- **Critic (CGSE).** A small MLP that does **only one** job: per-stage candidate scoring. Inputs are the student's own training stats (`build_critic_state`) plus a 5-dim local descriptor per `(stage, op)`. Trained by REINFORCE on `Δval_acc` with an entropy bonus. The student trains on **plain CE** — zero teacher forwards across the whole run.

**What CGSE keeps from SEArch (explicit borrowings, all referenced in code comments).**

- The iterative train→score→split→train outer loop (paper Algorithm 1).
- The two edge-splitting operators: **deepen** (sep-conv 3x3 stacked on edge) and **widen** (parallel sep-conv branch wrapping a block).
- The depthwise-separable Conv 3x3 building block from §3.5.
- The `B_op` cap on stacked deepens per stage (paper default 7).
- Function-preserving mutation initialisation (zero-init the new sep-conv's last BN γ → identity at insert).
- Param-budget termination (default `1.5 × initial_params`, ~408K for ResNet-20).
- Final retrain phase at fixed architecture after the loop terminates.

**What CGSE replaces (the contribution).**

- Loss: SEArch's `L = L_CE + λ·L_im` (channel-attention imitation, λ cosine-annealed) → plain `L = L_CE`.
- MV scorer: SEArch's `MV(n) = D(n) · deg+/deg-` → CGSE's learned `π_critic(global_state ⊕ local_descriptor)` over candidates.
- Cadence: SEArch's stage-clocked decisions (8 epochs/stage) → CGSE's high-frequency cadence (4 epochs/stage), plus `deepen_first: false` so the critic chooses deepen vs widen freely → 5–10× more decision points across the same total epoch budget.

**Code added.**

| File | Role |
|------|------|
| `models/searh_blocks.py` | `SepConv3x3`, `DeepenBlock` (residual sep-conv, identity-init), `WidenedBlock` (parallel sep-conv branch summed in, identity-init) |
| `training/searh_attention.py` | `ChannelAttentionKD` (paper Eqs. 1–3 with per-channel descriptor → softmax attention → projected feature → squared L2), `MultiNodeAttentionKD` (forward hooks on `layer1` / `layer2` / `layer3` outputs; lazy head-build, optimizer-friendly) |
| `training/searh_node_map.py` | Stage-output (student, teacher) module pairing — student/teacher feature shapes match exactly at `stage1` (16ch×32²), `stage2` (32ch×16²), `stage3` (64ch×8²) |
| `evolution/searh_mv.py` | `compute_per_node_distances` (no-grad averaging over `score_batches`), `modification_values` (Eq. 5), `rank_candidates` |
| `evolution/candidates.py` | `Candidate(stage, op, node_id)`, `enumerate_candidates` with `B_op` cap, `_is_widenable_basic_block` filter (only stride-1 same-channel blocks are wrapped) |
| `ops/searh_deepen.py` | `deepen_resnet_stage` — appends one `DeepenBlock` to `model.layer{stage}` |
| `ops/searh_widen.py` | `widen_resnet_stage` — wraps the last legal `BasicBlock` with a `WidenedBlock` |
| `training/searh_loop.py` | Unified `run_searh` outer loop (Algorithm 1). Switches selectors (`teacher` MV / `critic` MV) but everything else is shared. Implements cosine λ-anneal, score-batch loop for MV computation, optimizer refresh after every mutation, REINFORCE update with entropy bonus for the critic arm, final retrain phase. |
| `critics/searh_critic.py` | `PerCandidateCritic` — small MLP scoring `(global_state ⊕ local_descriptor)` per candidate |

**Code modified.**

- `train.py` now branches on `searh.enabled` *before* the legacy mutation/critic path; runs `run_searh`, then saves the checkpoint and returns.
- `scripts/run_tier2.sh` adds two rows: `student_searh` and `student_cgse_searh`.

**New configs.**

- `configs/tier2/student_resnet20_cifar10_searh.yaml` — paper-faithful SEArch arm. `selector: teacher`, `epochs_per_stage: 8`, `B_op: 7`, `deepen_first: true`, `lambda_init: 1.0`, `param_budget_factor: 1.5`, `final_retrain_epochs: 10`.
- `configs/tier2/student_resnet20_cifar10_cgse_searh.yaml` — CGSE-on-SEArch arm. `selector: critic`, `epochs_per_stage: 4` (high-frequency), `deepen_first: false`, `epsilon: 0.30`, `entropy_beta: 0.01`, `param_budget_factor: 1.5`. `teacher.enabled: false`.

**Smoke test (`scripts/smoke_searh.py`).**

Both arms run end-to-end on a 256-sample CIFAR subset in <30s on MPS. Verified:
- SEArch teacher hooks attach on stage outputs; channel-attention heads lazy-build; imitation loss flows back to student weights and to `q.weight`/`k.weight`.
- CGSE critic produces per-candidate scores; ε-greedy exploration fires; REINFORCE update fires every stage with sensible reward sign.
- Both arms apply mutations until the param-budget cap is hit and the candidate set is exhausted (28 mutations on the smoke subset; loop terminates cleanly).
- Optimizer refresh works across structural edits (no stale params, no missing new params).

**Bug fixed during smoke.** First widen attempt wrapped the stride-2 first block of stages 2/3, where the parallel branch's channel count (= block out-channels) didn't match the input channel count (= block in-channels). Fixed in `evolution/candidates.py::_is_widenable_basic_block` and `ops/searh_widen.py::widen_resnet_stage`: only blocks with `in_ch == out_ch and stride == 1` are eligible for widening (paper's "edge at end of stage" naturally corresponds to such a block).

**To run paper sweeps.**

```bash
SEEDS=42 bash scripts/run_tier2.sh
```

This now also produces:
- `runs_paper/tier2/logs/tier2_student_searh_seed42.log`
- `runs_paper/tier2/metrics/tier2_student_resnet20_searh_metrics_seed42.csv`
- `runs_paper/tier2/mutations/tier2_student_resnet20_searh_mutations_seed42.jsonl`

…and the matching `_cgse_searh_*` files. Mutation JSONL rows now include a richer schema: `op` ∈ {`searh_deepen`, `searh_widen`}, `selector` ∈ {`teacher`, `critic`}, `stage_target`, `node_id`, `mv`, `ranked_top5`, `param_budget_cap`.

**Documentation updated alongside.** `paper_documentation/SEArch-baseline-and-CGSE-evaluation-plan.md` §3a (new section) — complete mapping from paper sections (Eqs. 1–6, Algorithm 1, sep-conv, B_op, edge-splitting) to modules in this repo, plus the explicit "what CGSE borrows / replaces" tables.

### 2026-04-28 — CGSE student probe + REINFORCE baseline (closing the gap to SEArch)

**Goal.** Beat SEArch on accuracy at matched param budget without using a teacher. The previous CGSE arm had two structural disadvantages versus SEArch:

1. **No locality signal.** SEArch's MV scorer reads teacher-aligned per-stage feature distance; CGSE only saw aggregate training stats plus a 5-dim structural descriptor (stage one-hot, op flag, deepens count) — meaning the critic could not see *which stage was actually under-performing*.
2. **High-variance reward.** REINFORCE on raw Δval is extremely noisy at small architecture changes; without a baseline the critic was as likely to learn from noise as from signal.

Both are now addressed.

**1. Student probe (`critics/student_probe.py`).** Per stage we attach forward hooks and after every backward pass we collect:

- **`act_var_ratio`** — top-1 PC variance of the channel covariance ÷ trace(Σ). The unsupervised analogue of teacher attention distance D(n): when a stage's representations collapse onto a single direction, the ratio jumps toward 1 — a direct bottleneck signal. Computed without SVD via 4 steps of power iteration on Σ for ~0.5 ms per stage on MPS.
- **`grad_l2`** — Euclidean norm of the stage's parameter gradients from the last backward pass. Where the network is actually learning vs. where signal is dying.
- **`weight_delta`** — Frobenius distance between current stage weights and the snapshot taken at the most recent mutation in this stage. "Has this stage converged?" detector — collapses to 0 right after a mutation, then grows.

Both `grad_l2` and `weight_delta` are min-max-normalised across the three stages so the critic sees a comparable *relative* magnitude regardless of run-to-run scale.

These three features are concatenated into the per-candidate local descriptor at scoring time, growing it from **5-dim → 8-dim**. The critic's `local_dim` is set automatically in `train.py` based on `searh.use_student_probe`, so the same critic checkpoint format works for ablation pairs.

**2. REINFORCE baseline (`training/searh_loop.py`).** The PG step now subtracts an exponential-moving-average baseline `b ← momentum · b + (1-momentum) · R` (default `baseline_momentum: 0.9`) from the reward. The critic updates on the *advantage* `A = R − b` instead of the raw reward — sharply reducing gradient variance from sparse Δval signals. This is independent of the probe and helps every CGSE run, with or without the probe enabled.

**Code added.**

| File | Role |
|------|------|
| `critics/student_probe.py` | `StudentProbe` class with `attach`/`detach` hook lifecycle, `update_grads`, `run_forward` (one no-grad probe forward), `snapshot_stage` (mutation-time weight checkpoint), `per_stage_features` returning 3-dim per-stage descriptor. `PROBE_DIM = 3` exported for sizing the critic's `local_dim`. |

**Code modified.**

- `training/searh_loop.py` — `_make_local_descriptor` now accepts optional `probe_features`. `_critic_mv_selector` accepts `probe_features_per_stage` and passes the matching stage's features into each candidate's row. `run_searh` initialises the probe (if `searh.use_student_probe: true`), takes a baseline weight snapshot, gathers grads + activations once per stage end, calls `snapshot_stage` after each mutation, and detaches at run end. The PG block now maintains an EMA baseline and updates with `advantage = R − baseline`. The mutation JSONL gains `use_probe` and `baseline_value` fields.
- `train.py` — `local_dim` for `PerCandidateCritic` is now `5 + PROBE_DIM` when `use_student_probe: true`, else 5.
- `configs/tier2/student_resnet20_cifar10_cgse_searh.yaml` — `use_student_probe: true` and `baseline_momentum: 0.9` enabled by default for the paper-facing CGSE arm. The SEArch arm config is untouched (the probe is CGSE-only by design — it would shadow the teacher's MV signal otherwise).
- `scripts/smoke_searh.py` — runs the critic arm twice (probe-off, probe-on) to verify both code paths.

**Smoke results.**

```
[searh] critic PG update: choice=4 R=+0.1562 baseline=+0.0183 adv=+0.1533 entropy=1.609
[searh] critic PG update: choice=3 R=-0.1992 baseline=-0.0035 adv=-0.2175 entropy=1.609
```

Both probe-off and probe-on runs complete 28 mutations cleanly under the param budget; baseline EMA tracks recent rewards and the centred advantage is what drives the critic update.

**Why this is expected to beat SEArch on the headline accuracy claim.**

- **Same operators, same outer loop, same budget** as SEArch — accuracy upside has to come from *better-targeted* edits, and the probe gives the critic the same kind of locality information SEArch derives from the teacher (variance ratio plays the role of `D(n)`).
- **Zero teacher forwards** — the efficiency win is unconditional regardless of the accuracy outcome.
- **No teacher in the per-epoch path** — student-only forward + backward is ~40% cheaper per epoch than SEArch's student + teacher + attention-KD pipeline, so even at *matched epochs-per-stage* CGSE wins on wall-clock by construction. (This was the Tier-2-era CGSE configuration's "high-frequency cadence" — `epochs_per_stage: 4` vs SEArch's 8 — which gave the critic more decisions per epoch but introduced a comparability problem; **superseded for the paper headline by Tier 3's matched cadence — see the Tier 3 entry below.**)

**Ablation rows planned for the paper.**

| Arm | Loss | Selector | Probe | Baseline |
|-----|------|----------|-------|----------|
| SEArch (paper-faithful) | CE + λ·L_im | teacher MV | n/a | n/a |
| CGSE base | CE | critic | off | off |
| CGSE + baseline | CE | critic | off | on |
| CGSE + probe | CE | critic | on | off |
| **CGSE full (this paper's headline arm)** | CE | critic | on | on |

The first three rows isolate the contribution of each addition; the last is the headline arm.

### 2026-04-28 — Tier 3 launch plan: paper-faithful headline track with matched cadence

**Goal.** A self-contained, no-reuse track that becomes the paper's headline accuracy + efficiency table. Tier 2 stays as-is (legacy parity work, ResNet-safe ops, etc.); **Tier 3 is the contribution-claim track** and uses fresh teacher, fresh baselines, and a strictly matched outer loop between SEArch and all four CGSE ablation arms.

**The 8 Tier 3 models (all configs in `configs/tier3/`).**

| # | Arm | Config | Role |
|---|-----|--------|------|
| 1 | Teacher | `teacher_resnet56_cifar10.yaml` | ResNet-56, **100 ep** (vs Tier 2's 50 ep) — stronger teacher → sharper attention maps for SEArch's MV. **Trained once; shared by all student-arm seeds.** |
| 2 | Student CE | `student_resnet20_cifar10_ce.yaml` | ResNet-20, 50 ep, plain CE. The floor. |
| 3 | Student KD | `student_resnet20_cifar10_kd.yaml` | ResNet-20, 50 ep, logit KD (α=0.5, T=4). Isolates KD-only contribution. |
| 4 | SEArch | `student_resnet20_cifar10_searh.yaml` | Paper-faithful: channel-attention KD + MV scoring + sep-conv edge-splitting + B_op=7 + 1.5× param budget + 10-ep retrain. **`epochs_per_stage: 8`.** |
| 5 | CGSE base | `student_resnet20_cifar10_cgse_base.yaml` | probe **OFF**, baseline **OFF** (`baseline_momentum: 1.0` pins baseline at 0 → raw REINFORCE). |
| 6 | CGSE + baseline | `student_resnet20_cifar10_cgse_baseline.yaml` | probe OFF, **baseline ON** (`baseline_momentum: 0.9`). Isolates variance-reduction contribution. |
| 7 | CGSE + probe | `student_resnet20_cifar10_cgse_probe.yaml` | **probe ON**, baseline OFF. Isolates locality-signal contribution. |
| 8 | **CGSE full** ★ | `student_resnet20_cifar10_cgse_full.yaml` | **probe ON, baseline ON. Headline arm.** |

All four CGSE configs (rows 5-8) and the SEArch config (row 4) share identical outer-loop settings: **`epochs_per_stage: 8`, `B_op: 7`, `param_budget_factor: 1.5`, `final_retrain_epochs: 10`**. The only inter-arm differences are (a) the MV signal source (teacher attention vs critic policy) and (b) the two ablation toggles (`use_student_probe`, `baseline_momentum`).

**Cadence-parity decision (matched at 8 epochs/stage).** The earlier Tier 2 CGSE arm used `epochs_per_stage: 4` to give the critic a "high-frequency" decision pace. For Tier 3's headline claim — *"identical outer loop, only the MV signal differs"* — that asymmetry is unacceptable: a reviewer can attribute any CGSE win (or loss) to faster/slower decisions rather than to the critic. Tier 3 therefore matches both arms at 8 epochs/stage. CGSE is still ~1.6× faster in wall-clock at matched epochs because the per-epoch cost drops from ~60 s (student fwd + teacher fwd + attention-KD) to ~35 s (student fwd only) — the efficiency win comes from the loss term, not from the cadence.

**Per-seed totals.**

| Arm | Per-seed epochs | Per-seed wall-clock (MPS) |
|-----|-----------------|--------------------------|
| Teacher (one-time) | 100 | ~2.0 h |
| Student CE | 50 | ~25 min |
| Student KD | 50 | ~45 min |
| SEArch | 234 (28 stages × 8 ep + 10 retrain) | ~3.9 h |
| CGSE base / +B / +P / full | 234 each | ~2.3 h each |

**Sweep totals.**

| Sweep | Runs | Total epochs | Wall-clock |
|-------|------|--------------|------------|
| Minimal (1 seed × {CE, KD, SEArch, CGSE full} + 1 teacher) | 5 | 668 | ~9 h |
| **Headline (3 seeds × {CE, KD, SEArch, CGSE full} + 1 teacher)** | **13** | **1,804** | **~24 h** |
| Full ablation (3 seeds × all 7 student arms + 1 teacher) | 22 | 3,304 | ~38 h |

**Code added.**

| File | Role |
|------|------|
| `configs/tier3/*.yaml` | 8 fresh configs, no reuse from Tier 2. |
| `scripts/run_tier3.sh` | Sweep driver with `TEACHER_ONLY`, `STUDENTS_ONLY`, `RESUME`, `ARMS`, `SEEDS`, `DEVICE` toggles. Headline sweep: `SEEDS="42 43 44" bash scripts/run_tier3.sh`. |

**Artifact layout.** `runs_paper/tier3/{logs,metrics,mutations}/` and `checkpoints/tier3/`. Nothing under `runs_paper/tier2/` or `checkpoints/tier2/` is read or written by Tier 3.

**Operator inventory in Tier 3** (identical for both selectors): `deepen` (append residual sep-conv-3×3 block at end of stage) and `widen` (wrap last stride-1 same-channel `BasicBlock` with parallel sep-conv-3×3 branch). Both grow-only; no prune, no noop, no cross-stage moves. Matches SEArch paper Fig. 4a/4b. A reversible-mutation extension (`un_deepen`, `un_widen`) is listed as future work in [`SEArch-baseline-and-CGSE-evaluation-plan.md`](SEArch-baseline-and-CGSE-evaluation-plan.md) §3a but is not part of the Tier 3 sweep.

**Next steps.**

1. Run `bash scripts/run_tier3.sh TEACHER_ONLY=1` (~2 h) — produces `checkpoints/tier3/tier3_teacher_resnet56_cifar10_seed42.pt`.
2. Run minimal sweep (`SEEDS=42 ARMS="ce kd searh cgse_full" bash scripts/run_tier3.sh STUDENTS_ONLY=1`) to validate the pipeline on one seed (~7 h after teacher).
3. Launch headline sweep (`SEEDS="42 43 44" bash scripts/run_tier3.sh STUDENTS_ONLY=1`).
4. Once metrics CSVs are populated under `runs_paper/tier3/metrics/`, run `python scripts/build_results_site.py` to refresh the website with Tier 3 numbers.

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
