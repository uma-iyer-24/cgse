# CGSE codebase guide

**Purpose.** A **file-by-file** map of this repository so you can read it alongside the code and understand **what lives where**, **how data flows**, and **how pieces depend on each other**. This is the canonical “onboarding + deep dive” document for the implementation (complementary to `project-doc.pdf` for research narrative, `CGSE-implementation-log.md` for experiment history, and **[`CGSE-detailed-phase-walkthrough.md`](CGSE-detailed-phase-walkthrough.md)** for a long-form phase-by-phase story with rationale).

**How to maintain this document (important).**

- When you **add, rename, or delete** a module, script, or config: **update this guide in the same change** (or immediately after).  
- When you add a **new concept** (e.g. teacher, critic): add a short **architecture** subsection and new rows in the **file tables**.  
- Keep **entry points** (`train.py`, scripts) accurate so a reader can trace execution from top to bottom.

---

## Table of contents

1. [Big picture](#1-big-picture)
2. [Repository tree (top level)](#2-repository-tree-top-level)
3. [Execution paths](#3-execution-paths)
4. [Configs](#4-configs)
5. [Models](#5-models)
6. [Structural operations (`ops/`)](#6-structural-operations-ops)
7. [Training](#7-training)
8. [Utilities (`utils/`)](#8-utilities-utils)
9. [Scripts (`scripts/`)](#9-scripts-scripts)
10. [Tests & placeholders](#10-tests--placeholders)
11. [Generated & ignored artifacts](#11-generated--ignored-artifacts)
12. [Dependency graph (mental model)](#12-dependency-graph-mental-model)

---

## 1. Big picture

CGSE’s code is organized around three ideas:

1. **Mutable graph** — The student is a `torch.nn.Module` whose layers are stored in a **`ModuleDict`** with a fixed **execution order** (sequential forward). That makes “where to edit” explicit.
2. **Structural ops** — Small, testable functions (**widen**, **split/deepen**) that change width/depth while trying to **preserve behavior** (weight copy / identity init) and **fix downstream shapes**.
3. **Training loop** — Standard supervised learning; optionally **mutate** once or repeatedly, then **refresh the optimizer** so new parameters are trained.

Phase 2 adds **CIFAR-10** and **logging**. Phase 3 adds a **frozen teacher + KD** path (**SEArch-style control**). **CGSE** uses **`StructuralCritic`** (+ **`state_features`**) in **`train.py`** for mutation gating (no teacher); other multi-objective / hybrid ideas from early notes are **not** in scope for this repo.

**Planned backbone progression.** The current student in Phase 2 is a **small CIFAR CNN** (`CifarGraphNet`) because it is simple, fast, and makes mutation plumbing easy to validate. For paper-quality comparisons and later phases, the intended direction is to move the student backbone toward a **ResNet-style model** (still on CIFAR first), while keeping the same mutation/controller interfaces. When that switch happens, update this document’s **Models** and **Execution paths** sections accordingly.

---

## 2. Repository tree (top level)

```
cgse/
├── train.py                 # Main training CLI
├── requirements.txt         # Python deps (torch, torchvision, PyYAML)
├── configs/
│   ├── cifar/               # CIFAR YAML; `smoke/` = quick subset runs
│   ├── evolution/           # Tier 1b; `smoke/` = short dev runs
│   └── synthetic/           # base.yaml (MLP smoke)
├── models/                  # GraphModule, StudentNet, CifarGraphNet
├── ops/                     # edge_widen, edge_split (mutations)
├── training/                # Data loaders, train/eval loop, synthetic data
├── utils/                   # checkpoint, seeds, validators, mutation logging, optimizer refresh
├── scripts/                 # Standalone mutation / robustness demos (not pytest)
├── critics/                 # StructuralCritic (CGSE; replaces teacher for structural decisions)
├── paper_documentation/     # Paper PDFs, implementation log, this guide
├── runs/                    # tier1|tier1b|smoke|other × metrics|logs|mutations
├── checkpoints/             # Saved .pt (some patterns gitignored)
└── data/                    # Local datasets (gitignored)
```

---

## 3. Execution paths

### 3.1 Primary: CIFAR Phase 2 training

1. **`train.py`** loads YAML (`--config`, optional `--device` override).
2. **`training/data.build_cifar10_loaders`** builds train/test `DataLoader`s (optional subsets).
3. **`models/cifar_student.CifarGraphNet`** is constructed and moved to device; **`utils/graph_validator.validate_forward`** runs a tiny batch sanity check.
4. **Optimizer** (Adam by default) wraps `model.parameters()`.
5. Each **epoch**: **`training/loop.train_one_epoch`** → **`training/loop.evaluate`** (test set as “val”). If YAML **`teacher.enabled`**, training loss mixes CE with **KD** against a frozen **`CifarGraphNet`** loaded from **`teacher.checkpoint`**.
6. Optional **mutation**: YAML **`once_after_epoch`** (scheduled) **or** **`critic.enabled`** window + Bernoulli/ε-greedy gate → **`ops/edge_widen`**, **`refresh_optimizer`**, optional REINFORCE step on **`StructuralCritic`** (next-epoch Δval).
7. **Logging**: optional CSV (`training.log_csv`, includes **`critic_score`** when critic on), optional mutation JSONL (`mutation.log_jsonl`), console prints.
8. **Checkpoint**: **`save_checkpoint`** writes `checkpoints/<experiment.name>.pt`; critic weights → **`checkpoints/<experiment.name>_critic.pt`** when **`critic.enabled`**.

### 3.2 Legacy / smoke: synthetic MLP

If `model.name` is absent or not `cifar_cnn`, **`train.py`** uses **`models/student.StudentNet`** and **`training/synthetic.build_synthetic_loaders`** — random tensors, useful for quick pipeline checks (`configs/synthetic/base.yaml`).

### 3.3 Phase 1 validation scripts

Scripts under **`scripts/`** import **`models.graph.GraphModule`**, **`ops.edge_split`**, **`ops.edge_widen`**, **`utils.optimizer_utils.refresh_optimizer`**, etc., to test **preservation**, **live mutation**, **robustness** without full CIFAR training.

---

## 4. Configs

| File | Role |
|------|------|
| **`configs/synthetic/base.yaml`** | Phase-0 style: synthetic data, small MLP (`StudentNet`), `device`, few epochs. No `model.name` → MLP path in `train.py`. |
| **`configs/cifar/phase2_cifar.yaml`** | Default Phase 2: CIFAR subset, `CifarGraphNet`, `training.log_csv`, optional `mutation` (often off for baseline). |
| **`configs/cifar/phase2_cifar_full.yaml`** | Full 50k/10k CIFAR, more epochs — paper baseline runs. |
| **`configs/cifar/phase2_cifar_full_mutate.yaml`** | Same as full baseline + **one** `edge_widen` after epoch 10; separate CSV/JSONL/checkpoint name. |
| **`configs/cifar/phase2_cifar_full_cgse.yaml`** | Full CIFAR, **no teacher**; **`critic:`** gates one widen inside a window + REINFORCE. |
| **`configs/cifar/smoke/phase2_cifar_cgse_smoke.yaml`** | Small subset/epochs for fast CGSE smoke. |
| **`configs/cifar/smoke/phase2_smoke.yaml`** | Tiny subset, CPU-friendly smoke test. |
| **`configs/cifar/smoke/phase2_smoke_mutate.yaml`** | Smoke + one widen + mutation JSONL path. |
| **`configs/cifar/phase3_cifar_kd.yaml`** | Full CIFAR + **KD**: frozen teacher from `teacher.checkpoint` (same `CifarGraphNet` arch). |
| **`configs/cifar/smoke/phase3_cifar_kd_smoke.yaml`** | Small subset, few epochs, CPU; same teacher block. |
| **`configs/cifar/baseline_sear_ch_teacher_mutate.yaml`** | **SEArch control:** teacher + KD + one widen (same schedule as full mutate). |
| **`configs/evolution/evolution_tier1b_schedule.yaml`** | Tier 1b: fixed multi-op schedule, full CIFAR. |
| **`configs/evolution/evolution_tier1b_critic.yaml`** | Tier 1b: discrete critic over legal ops. |
| **`configs/evolution/smoke/evolution_tier1b_smoke.yaml`**, **`configs/evolution/smoke/evolution_tier1b_critic_smoke.yaml`** | Small subset / few epochs for pipeline checks. |

**Cross-cutting YAML sections:**

- **`experiment.name`** — Used for checkpoint filename and run identity in logs.
- **`data`** — `root`, `num_workers`, `subset_train`, `subset_test` (see `training/data.py`).
- **`training`** — `epochs`, `batch_size`, `lr`, `weight_decay`, `seed`, `log_csv`.
- **`model`** — For CIFAR: `name: cifar_cnn`, `num_classes`. For MLP: `input_dim`, `hidden_dim`, `output_dim`.
- **`device`** — `cpu`, `cuda`, `mps`, or `auto` (overridable by CLI).
- **`teacher`** (Phase 3, CIFAR only) — `enabled`, `checkpoint`, `temperature`, `alpha` (KD weight on softened logits).
- **`mutation`** — `enabled`, `once_after_epoch`, `widen_delta`, `log_jsonl`.

---

## 5. Models

### 5.1 `models/graph.py`

**`GraphModule`** (`torch.nn.Module`):

- **`nodes`**: `nn.ModuleDict` — layer id → module.
- **`execution_order`**: list of ids — **forward** runs modules in this order.
- **`add_node`**, **`get_node`** — register and lookup.
- **`forward(x)`** — strictly sequential: each node receives the previous tensor output (supports Conv → … → Flatten → Linear chains).
- **`describe()`** — prints human-readable layer list (debugging).
- **`validate()`** — linear-chain dimension walk (legacy helper for MLP-heavy graphs; CNNs also work once execution hits the first `Linear`).
- **`widen_node(node_id, extra_out)`** — widen one `Linear` and resize downstream `Linear` layers (in-place mutation).
- **`insert_after(target_id, new_id, new_module)`** — deepen / insert (used by student deepen API).

**`Node`** — lightweight id + module holder (optional; core path uses `ModuleDict` directly).

### 5.2 `models/student.py`

**`StudentNet(GraphModule)`** — two-layer MLP: `linear1` → `relu1` → `linear2`.

- **`deepen_after(node_id)`** — insert **identity-initialized** `Linear(dim, dim)` after a `Linear` node (Net2Net-style deepen).

### 5.3 `models/cifar_student.py`

**`CifarGraphNet(GraphModule)`** — small **CNN** for CIFAR-10: several **Conv → BN → ReLU → Pool** blocks, **`Flatten`**, then **`fc1` → `relu_fc` → `fc2`**. Mutations in the current codebase typically target **`fc1`** (first `Linear` in order) when using `edge_widen` with auto target.

---

## 6. Structural operations (`ops/`)

| File | Role |
|------|------|
| **`ops/edge_widen.py`** | **`edge_widen(model, target_node_id=None, delta=...)`** — widen a `Linear`’s output by `delta`, copy existing weights into the top rows, then **resize every downstream `Linear`** to match new input width. Returns `model`. If `target_node_id` is `None`, picks the **first** `Linear` in `execution_order`. |
| **`ops/edge_split.py`** | **`edge_split(model, target_node_id=None)`** — insert an **identity** `Linear(in_f, in_f)` **before** the target linear in the order (deepen path). Forbids splitting the **last** layer (output). Fan-in **`in_f`** is the target linear’s **`in_features`** (correct after `Flatten` / conv stacks). |
| **`ops/__init__.py`** | Package marker (may be empty). |

**Invariant:** After these ops, **`utils/graph_validator.validate_graph`** (linear-only walk) or **`validate_forward`** (full forward) should be used depending on architecture.

---

## 7. Training

| File | Role |
|------|------|
| **`training/loop.py`** | **`train_one_epoch(..., teacher=None, kd_temperature, kd_alpha)`** — CE, or \((1-\alpha)\) CE + \(\alpha\) KD vs frozen teacher. **`evaluate`** — CE + accuracy (student only). |
| **`training/data.py`** | **`build_cifar10_loaders(cfg)`** — torchvision **CIFAR-10**, train augment, test eval transform, optional **`Subset`**. |
| **`training/synthetic.py`** | **`build_synthetic_loaders(cfg)`** — `TensorDataset` of random features/labels for MLP smoke tests. |

---

## 8. Utilities (`utils/`)

| File | Role |
|------|------|
| **`utils/checkpoint.py`** | **`save_checkpoint`**, **`load_model_weights`** (restore `model` from a file saved by `save_checkpoint`). |
| **`utils/repro.py`** | **`set_seed(seed)`** — Python / NumPy / PyTorch seeds. |
| **`utils/graph_validator.py`** | **`validate_graph(model, input_dim=...)`** — walks `Linear` layers in order (legacy MLP checks). **`validate_forward(model, x)`** — one forward pass (good for CNNs). |
| **`utils/optimizer_utils.py`** | **`refresh_optimizer(old_opt, model)`** — new optimizer over `model.parameters()`, copies optimizer state for **tensor-identical** parameters, fresh state for new tensors. |
| **`utils/model_info.py`** | **`count_trainable_parameters`**, **`first_linear_node_id`**, **`linear_layer_shapes`** — introspection for logging and mutation targeting. |
| **`utils/mutation_log.py`** | **`append_mutation_jsonl(path, dict)`** — append one JSON object per line (mutation events). |
| **`utils/run_paths.py`** | **`normalize_run_artifact_path`** — redirects legacy `paper_documentation/runs/...` strings in config to `runs/...` (warns). |
| **`paper_documentation/runs`** | **Sentinel file** (not a directory). Prevents accidental creation of `paper_documentation/runs/` as a folder; see file contents. |

---

## 9. Scripts (`scripts/`)

Runnable demos / stress tests (invoke with `python scripts/<name>.py` from repo root or with `sys.path` hacks as in each file):

| Script | Purpose |
|--------|---------|
| **`validate_mutation.py`** | Forward/backward before/after split + widen; uses **`validate_graph`**. |
| **`test_preservation.py`** | Compare outputs before/after **`edge_split`** (expect ~0 diff for identity insert). |
| **`test_live_mutation.py`** | Train a few steps, mutate, refresh optimizer, train again. |
| **`test_robustness.py`** | Random sequences of split/widen + **`refresh_optimizer`**; deterministic replay check. |
| **`test_graph_visual.py`**, **`test_graph_ascii.py`** | Visual / ASCII printouts of graph structure after mutations (human inspection). |

---

## 10. Tests & placeholders

| Path | Status |
|------|--------|
| **`tests/test_graph_ops.py`** | **Pytest:** KD formula, checkpoint round-trip, **`edge_widen` / `edge_split`**, teacher+KD smoke, critic state (`pytest tests/test_graph_ops.py`). |

---

## 11. Generated & ignored artifacts

| Path | Notes |
|------|------|
| **`data/`** | CIFAR download cache — **gitignored** (see root `.gitignore`). |
| **`checkpoints/`** | Training outputs; **`checkpoints/cgse_*.pt`** pattern gitignored; older **`phase0.pt`** may still be tracked from earlier commits. |
| **`runs/`** (repo root) | **`metrics/`**, **`logs/`**, **`mutations/`**; top-level **symlinks** keep legacy paths working. See **`runs/README.md`**. |
| **`__pycache__/`, `.ipynb_checkpoints/`** | Should not be committed — in `.gitignore`. |

---

## 12. Dependency graph (mental model)

```mermaid
flowchart TD
  subgraph configs [YAML under configs/]
    Y[YAML]
  end
  train[train.py]
  data[training/data.py / synthetic.py]
  loop[training/loop.py]
  cifar[models/cifar_student.py]
  student[models/student.py]
  graph[models/graph.py]
  ops[ops/edge_widen.py / edge_split.py]
  optu[utils/optimizer_utils.py]
  log[utils/mutation_log.py / model_info.py]
  ckpt[utils/checkpoint.py]

  Y --> train
  train --> data
  train --> cifar
  train --> student
  cifar --> graph
  student --> graph
  train --> loop
  train --> ops
  ops --> graph
  train --> optu
  train --> log
  train --> ckpt
```

**Rule of thumb:** **`models/graph.py`** + **`ops/*`** define *what can change*; **`training/*`** + **`train.py`** define *how we learn*; **`utils/*`** define *reproducibility and observability*.

**Planned (Tier 1b).** Multi-stage training, **param/FLOP budget**, **multiple** mutations, **discrete (site × operator)** critic — spec and checklist in **[`SEArch-baseline-and-CGSE-evaluation-plan.md`](SEArch-baseline-and-CGSE-evaluation-plan.md) §7**. Update this guide when `train.py` gains stage loops and new ops.

---

## Document history

| Date | Change |
|------|--------|
| 2026-04-02 | Initial guide: Phase 1–2 layout, configs, scripts, placeholders. |
| 2026-04-02 | **`runs/`** at **repo root** only; legacy YAML paths rewritten in code. |
| 2026-04-02 | **`paper_documentation/runs`** is a **sentinel file** blocking a duplicate `runs/` directory under docs. |
| 2026-04-04 | Cross-link **[`CGSE-detailed-phase-walkthrough.md`](CGSE-detailed-phase-walkthrough.md)** from the purpose blurb (narrative + rationale companion). |
| 2026-04-04 | Phase 3: **`teacher`** YAML, KD in **`training/loop.py`**, **`load_model_weights`**, configs **`phase3_cifar_kd*.yaml`**. |
| 2026-04-04 | Scope: **teacher vs critic** only; **`baseline_sear_ch_teacher_mutate.yaml`**, **`StructuralCritic`** in **`critics/`**. |
| 2026-04-04 | **Tier 1b** roadmap: pointer to **[`SEArch-baseline-and-CGSE-evaluation-plan.md`](SEArch-baseline-and-CGSE-evaluation-plan.md) §7** (multi-stage, multi-op, critic v2). |

*Append a row whenever this guide is meaningfully updated.*
