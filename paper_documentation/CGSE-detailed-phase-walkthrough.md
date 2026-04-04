# CGSE — Detailed phase-by-phase walkthrough (English)

**Purpose.** This document is a **long-form, narrative explanation** of what the CGSE codebase implements today: **what each phase is trying to achieve**, **how we implemented it step by step**, **why we made specific choices**, and **which files contain each piece**. It complements the shorter tables in [`CGSE-implementation-log.md`](CGSE-implementation-log.md) (experiment history) and [`CGSE-codebase-guide.md`](CGSE-codebase-guide.md) (file-by-file map).

**Audience.** Anyone who wants a **readable story** of the implementation—not only where code lives, but **the reasoning chain** behind it.

**How to use it.** Read phases in order (0 → 1 → 2). Within each phase, subsections are numbered steps. The **“Files”** bullets at the end of each step point to the concrete locations in the repository.

---

## Table of contents

1. [Background: what CGSE is building toward](#1-background-what-cgse-is-building-toward)
2. [Phase 0 — Environment, config, and a minimal training path](#2-phase-0--environment-config-and-a-minimal-training-path)
3. [Phase 1 — Mutable graph, structural mutations, and safety rails](#3-phase-1--mutable-graph-structural-mutations-and-safety-rails)
4. [Phase 2 — Real data (CIFAR-10), a graph-based CNN student, logging, and in-training mutation](#4-phase-2--real-data-cifar-10-a-graph-based-cnn-student-logging-and-in-training-mutation)
5. [Cross-cutting work: reproducibility, artifacts layout, and housekeeping](#5-cross-cutting-work-reproducibility-artifacts-layout-and-housekeeping)
6. [What is deliberately not done yet (later phases)](#6-what-is-deliberately-not-done-yet-later-phases)
7. [Quick reference: config → behavior → outputs](#7-quick-reference-config--behavior--outputs)

---

## 1. Background: what CGSE is building toward

**Research direction (high level).** The repo implements **structurally mutable** students on CIFAR-10 and builds toward **one** clear comparison: **SEArch-style control** uses a **frozen teacher** (knowledge distillation) as the extra signal while the student may **mutate**; **CGSE** **removes the teacher** and uses a **structural critic** for **mutation** decisions (classification still uses label CE). Other ideas from early brainstorming are **not** in scope here.

**What this repository implements today.** The code is intentionally **staged**:

- **Phase 0–1:** **Mutable graph**, **widen/split** ops, **validation**, **optimizer refresh**.
- **Phase 2:** **CIFAR-10**, **`CifarGraphNet`**, metrics, CSV/JSONL, **YAML-timed** mutation.
- **Phase 3:** **Teacher + KD** (`phase3_cifar_kd.yaml`) and **`baseline_sear_ch_teacher_mutate.yaml`** (teacher + KD + mutate — the **control** arm).
- **Next (CGSE):** train **`StructuralCritic`** and **gate** mutations from it in **`train.py`** (no teacher).

---

## 2. Phase 0 — Environment, config, and a minimal training path

**Goal.** Establish a **reproducible Python project**: dependencies, a **single entry point** for training, **YAML-driven** hyperparameters, and a **tiny dataset** that does not require downloads—so CI and quick sanity checks are cheap.

### Step 2.1 — Dependencies and project layout

**What we did.** Declared core dependencies in **`requirements.txt`**: PyTorch, torchvision (for later CIFAR), and PyYAML for configs.

**Why.** Pinning expectations in one file avoids “works on my machine” drift; collaborators and paper reviewers can recreate the environment from a short install step.

**Files.**

- **`requirements.txt`** — `torch`, `torchvision`, `PyYAML`.

### Step 2.2 — Configuration as the single source of truth for experiments

**What we did.** Experiment settings (epochs, batch size, learning rate, model sizes, device, data paths) live in **YAML files** under **`configs/`**, not hard-coded in Python. **`train.py`** loads the chosen file via **`--config`**.

**Why.** Papers need **exact reproducibility** and **ablation tables**; YAML diffs are readable, and you can keep many experiments as small files without branching code.

**Files.**

- **`configs/synthetic/base.yaml`** — Default **synthetic / Phase-0 style** run: random tensor “dataset,” small MLP dimensions, few epochs.
- **`train.py`** — **`load_config`**, argument parser with **`--config`** (default points at Phase 2 CIFAR config today) and optional **`--device`** override.

### Step 2.3 — Synthetic data for the MLP student

**What we did.** Implemented **`build_synthetic_loaders`** that builds a **`TensorDataset`** of random feature vectors and integer labels, wrapped in **`DataLoader`s**. This path is used when the model is **not** the CIFAR CNN (see Step 4.2 for the branch in **`train.py`**).

**Why.** Before touching CIFAR, you want to verify **training loops, checkpointing, and mutation hooks** without network I/O or large downloads.

**Files.**

- **`training/synthetic.py`** — Synthetic loaders from config fields.

### Step 2.4 — Minimal MLP student on the graph base

**What we did.** **`StudentNet`** subclasses **`GraphModule`** and registers a short chain: linear → ReLU → linear. This is the **original student** for Phase-0/1 experiments.

**Why.** An MLP is **small and predictable** for testing **`edge_widen`** and **`edge_split`** logic; convolutions add shape bookkeeping that you add only once the graph abstractions are trusted.

**Files.**

- **`models/student.py`** — **`StudentNet`**, **`deepen_after`** (Net2Net-style deepen on the graph).
- **`models/graph.py`** — **`GraphModule`**, **`execution_order`**, **`add_node`**, forward pass, widen/insert helpers (shared by MLP and CNN students).

### Step 2.5 — Training step: one epoch and evaluation

**What we did.** Factored **one full pass** over a loader (**`train_one_epoch`**) and **evaluation** (**`evaluate`**) into **`training/loop.py`**, using cross-entropy and reporting **loss and accuracy**.

**Why.** Keeps **`train.py`** focused on orchestration (config, device, mutation, logging, checkpoint) while the math of a standard supervised step stays in one reusable module.

**Files.**

- **`training/loop.py`** — **`train_one_epoch`**, **`evaluate`**.

### Step 2.6 — Checkpoints and seeds (partial Phase 0)

**What we did.** **`utils/checkpoint.py`** saves model (and optimizer) state to disk; **`utils/repro.py`** sets Python, NumPy, and PyTorch seeds from the config.

**Why.** Reproducibility and resume experiments; full “config embedded in checkpoint” is still **optional / future** work (see implementation log **Phase status**).

**Files.**

- **`utils/checkpoint.py`** — **`save_checkpoint`**.
- **`utils/repro.py`** — **`set_seed`**.

**Phase 0 status note.** Synthetic training + config + loop + checkpoint **exist**; some roadmap items (e.g. richer checkpoint metadata, full resume story) remain **partial**—documented in [`CGSE-implementation-log.md`](CGSE-implementation-log.md) §3.

---

## 3. Phase 1 — Mutable graph, structural mutations, and safety rails

**Goal.** Represent the student as an **explicit ordered graph** of layers, apply **structural mutations** (widen linear layers, split/deepen edges) **in place**, and ensure the model **still runs and trains**—including **refreshing the optimizer** when new parameters are created.

### Step 3.1 — Graph module and execution order

**What we did.** **`GraphModule`** stores layers in an **`nn.ModuleDict`** keyed by **string ids** and runs them in a fixed **`execution_order`**. The **forward** pass feeds each module the output of the previous one—supporting Conv → … → Flatten → Linear chains for Phase 2.

**Why.** Mutations need a **stable address** for “which layer to edit” and a **deterministic forward** so validators and tests are meaningful. A plain **`nn.Sequential`** could work for simple cases, but explicit ids align with **operator APIs** and logging (**`target_node_id`** in JSONL).

**Files.**

- **`models/graph.py`** — **`GraphModule`**, **`add_node`**, **`forward`**, **`widen_node`**, **`insert_after`**, **`describe`**, **`validate`**.

### Step 3.2 — Widen operator (more units on a linear edge)

**What we did.** **`edge_widen`** in **`ops/edge_widen.py`** increases the **output dimension** of a target **`nn.Linear`** by **`delta`**, **copies** existing weights into the preserved part of the new weight matrix, and **resizes downstream Linear layers** so input dimensions stay consistent. If **`target_node_id`** is omitted, the op can select the **first Linear** in **`execution_order`** (used by the training script’s default mutation path).

**Why.** Widening is a standard **Net2Net-style** move: expand capacity while **preserving** the function approximately (new units start from zeros in the copied block pattern used here). It is a building block before **learned** widening policies.

**Files.**

- **`ops/edge_widen.py`** — **`edge_widen`**.

### Step 3.3 — Split / deepen operator (insert identity path)

**What we did.** **`edge_split`** inserts an **identity-initialized** linear (or equivalent deepening) **before** a target linear in the execution order, subject to constraints (e.g. not splitting the final output layer in a way that breaks the head).

**Why.** **Depth** expansion is a second axis of structural change beside **width**; identity init aims for **near-zero change** in outputs at initialization so training can gradually use the new capacity.

**Files.**

- **`ops/edge_split.py`** — **`edge_split`** (identity insert uses the target linear’s **`in_features`**).

### Step 3.4 — Validation after structural edits

**What we did.** **`utils/graph_validator.py`** provides **`validate_graph`** (linear-chain dimension walk, very useful for MLP-heavy graphs) and **`validate_forward`** (run a **real forward** with a tensor batch—**essential for CNNs** where a pure linear walk is not the whole story).

**Why.** Mutations are **easy to get subtly wrong**; a forward pass catches **shape errors** before you waste hours training a broken model.

**Files.**

- **`utils/graph_validator.py`** — **`validate_graph`**, **`validate_forward`**.

### Step 3.5 — Optimizer refresh after new parameters

**What we did.** **`refresh_optimizer`** builds a **new optimizer** over **`model.parameters()`** and copies optimizer state for parameters that **match by tensor identity**, leaving **fresh state** for newly added tensors.

**Why.** PyTorch optimizers hold **per-parameter momentum/variance**; after structural edits, **parameter objects change**; reusing the old optimizer blindly is incorrect. Refresh is the standard pattern in “train → mutate → train” workflows.

**Files.**

- **`utils/optimizer_utils.py`** — **`refresh_optimizer`**.

### Step 3.6 — Standalone scripts to stress-test mutations

**What we did.** Under **`scripts/`**, several runnable scripts **train briefly**, **mutate**, **refresh**, and **compare outputs or robustness**—without requiring CIFAR.

**Why.** **Interactive debugging** and **regression confidence** before coupling mutations to long CIFAR jobs.

**Files.**

- **`scripts/validate_mutation.py`** — Forward/backward around split + widen.
- **`scripts/test_preservation.py`** — Output similarity after identity-style split.
- **`scripts/test_live_mutation.py`** — Train, mutate, refresh, train again.
- **`scripts/test_robustness.py`** — Sequences of mutations + deterministic checks.
- **`scripts/test_graph_visual.py`**, **`scripts/test_graph_ascii.py`** — Human-readable structure dumps.

### Step 3.7 — Structural critic (CGSE)

**What we did.** **`critics/critic.py`** — **`StructuralCritic`** (MLP logit for mutate probability). **`critics/state_features.py`** — **`build_critic_state`** (8-D stats). **`train.py`** — YAML **`critic:`** window, ε-greedy + Bernoulli action, **REINFORCE** on one-step-ahead Δval after a widen; **`teacher.enabled`** and **`critic.enabled`** are mutually exclusive.

**Why.** **CGSE** replaces the teacher with **internal training statistics** for **when** to apply the same **`edge_widen`** operator (classification remains label CE).

**Files.**

- **`critics/critic.py`**, **`critics/state_features.py`**, **`critics/__init__.py`**, **`train.py`**

**Phase 1 status note.** Core **ops + graph + optimizer refresh + scripts** are in place; optional **API consolidation** can still grow; **`tests/test_graph_ops.py`** holds **pytest** coverage.

---

## 4. Phase 2 — Real data (CIFAR-10), a graph-based CNN student, logging, and in-training mutation

**Goal.** Move from “mutation lab on toy data” to a **credible small vision experiment**: **CIFAR-10**, a **CNN student** that is **still a `GraphModule`**, **honest metrics** each epoch, **artifact paths suitable for a paper** (CSV curves, JSONL mutation events), and **optional widen during training** controlled by YAML.

### Step 4.1 — CIFAR-10 data pipeline

**What we did.** **`training/data.py`** defines **`build_cifar10_loaders`**: torchvision **CIFAR-10** download into **`./data`** (gitignored), **standard normalization** (per-channel mean/std commonly used for CIFAR), **train-time augmentation** (random crop with padding, horizontal flip), and **test-time** evaluation transforms without augmentation. Optional **`subset_train`** and **`subset_test`** in YAML take the **first N** examples for **fast smoke runs**.

**Why.** CIFAR-10 is a **widely cited** benchmark; the recipe is **simple to describe in a paper**; subsets let you **debug quickly** on CPU before launching full 50-epoch jobs.

**Files.**

- **`training/data.py`** — **`build_cifar10_loaders`**, **`_cifar10_transforms`**.
- **`.gitignore`** (repo root) — Typically ignores **`data/`** so binaries are not committed.

### Step 4.2 — Branch in `train.py`: CIFAR CNN vs synthetic MLP

**What we did.** If **`model.name`** in YAML is **`cifar_cnn`**, **`train.py`** builds CIFAR loaders and **`CifarGraphNet`**; otherwise it uses **synthetic loaders** and **`StudentNet`**.

**Why.** One entry point **`train.py`** serves both **Phase 0 smoke** and **Phase 2 research**; the **default `--config`** was switched to Phase 2 CIFAR to match the **active** experimental track (see decision **D3** in the implementation log).

**Files.**

- **`train.py`** — Device resolution, seed, model/data branch, training loop, mutation block, CSV/JSONL, checkpoint.

### Step 4.3 — `CifarGraphNet`: small CNN as an ordered graph

**What we did.** **`CifarGraphNet`** stacks three **Conv → BatchNorm → ReLU → MaxPool** blocks, then **`Flatten`**, then **`fc1` → ReLU → `fc2`**. The **flattened** feature size is **128 × 4 × 4 = 2048**, so **`fc1`** is **`Linear(2048, 256)`** and **`fc2`** maps to **`num_classes`** (10 for CIFAR-10).

**Why.**

- **CNN** rather than MLP on pixels: matches **standard practice** for CIFAR and produces **meaningful** accuracy baselines.
- Still a **`GraphModule`**: Phase 1 **`edge_widen`** can target **`fc1`** (first **`nn.Linear`** in **`execution_order`**) without rewriting the mutation code for a special `nn.Sequential` CNN.

**Files.**

- **`models/cifar_student.py`** — **`CifarGraphNet`**.

### Step 4.4 — Forward sanity check on real batches

**What we did.** After constructing the CIFAR model, **`train.py`** grabs a **tiny batch** from the train loader and runs **`validate_forward`**.

**Why.** Catches **wrong image sizes**, **wrong channel counts**, or **broken graph order** before training starts.

**Files.**

- **`train.py`** — Call site.
- **`utils/graph_validator.py`** — **`validate_forward`**.

### Step 4.5 — Per-epoch CSV metrics (paper-friendly curves)

**What we did.** If **`training.log_csv`** is set in YAML, **`train.py`** **appends** one row per epoch with fixed columns: timestamp, experiment name, epoch, train/val loss and accuracy, **trainable parameter count**, and whether mutation has already fired.

**Why.** Spreadsheets and plotting tools **ingest CSV trivially**; **`num_parameters`** makes **capacity changes** visible when mutation is on; **`mutation_applied_yet`** helps align curves **before/after** a structural event.

**Files.**

- **`train.py`** — **`_METRIC_FIELDS`**, **`_append_metrics_csv`**.

### Step 4.6 — YAML-driven in-loop mutation (single widen)

**What we did.** Under **`mutation:`** in YAML: **`enabled`**, **`once_after_epoch`** (0-based **epoch index**—mutation runs **after** that epoch’s train+eval completes), **`widen_delta`**, optional **`log_jsonl`**. When conditions match, **`train.py`** counts parameters, resolves **first Linear id** via **`first_linear_node_id`**, calls **`edge_widen`**, **`refresh_optimizer`**, logs to console, and optionally appends **one JSON object** to the JSONL file.

**Why.**

- **Single scheduled mutation** first: simple **ablation** against **no mutation** (same seed, same hyperparameters).
- **First Linear (`fc1`)** is a **clear, documented** widening target; later work can add **explicit `target_node_id`** or **controller-chosen** targets.

**Files.**

- **`train.py`** — Mutation block inside the epoch loop.
- **`ops/edge_widen.py`** — **`edge_widen`**.
- **`utils/optimizer_utils.py`** — **`refresh_optimizer`**.
- **`utils/model_info.py`** — **`count_trainable_parameters`**, **`first_linear_node_id`**, **`linear_layer_shapes`**.
- **`utils/mutation_log.py`** — **`append_mutation_jsonl`**.

### Step 4.7 — Mutation JSONL (structural event log)

**What we did.** Each mutation writes a **single JSON line** including experiment name, run timestamp id, epoch, target node id, delta, linear shapes before/after, and parameter counts before/after.

**Why.** **CSV** is **per-epoch**; **JSONL** is ideal for **discrete events** (one line per mutation) and **rich nested fields** for append-only logging and later figure generation.

**Files.**

- **`utils/mutation_log.py`**
- **`configs/cifar/smoke/phase2_smoke_mutate.yaml`**, **`configs/cifar/phase2_cifar_full_mutate.yaml`** — Example **`mutation.log_jsonl`** paths.

### Step 4.8 — Config files for different experiment scales

**What we did.** Split YAML configs so you can run **smoke**, **default subset**, **full data baseline**, and **full data + mutation** without editing Python.

| Config file | Intent |
|-------------|--------|
| **`configs/cifar/phase2_cifar.yaml`** | Practical default Phase 2 run (often subset / moderate epochs—tune as needed). |
| **`configs/cifar/phase2_cifar_full.yaml`** | **Full** train/test CIFAR, longer training—**fixed architecture baseline** (`mutation.enabled: false`). |
| **`configs/cifar/phase2_cifar_full_mutate.yaml`** | **Same** as full baseline (same seed and hyperparameters) but **one widen** after epoch 10; **separate** CSV, JSONL, and **`experiment.name`** so artifacts do not overwrite the baseline. |
| **`configs/cifar/smoke/phase2_smoke.yaml`** | Tiny data, CPU-friendly **sanity** run. |
| **`configs/cifar/smoke/phase2_smoke_mutate.yaml`** | Smoke + mutation + JSONL path test. |
| **`configs/cifar/phase3_cifar_kd.yaml`** | Full CIFAR + KD from a saved student checkpoint used as teacher. |
| **`configs/cifar/smoke/phase3_cifar_kd_smoke.yaml`** | Tiny subset KD smoke. |
| **`configs/synthetic/base.yaml`** | Synthetic MLP path (no CIFAR). |

**Why.** **Separation of concerns**: baseline vs ablation is a **config diff**, not a code fork—critical for **Methods** sections and **reproducibility**.

**Files.** All under **`configs/`** as above.

### Step 4.9 — Recorded baseline result (mutation off)

**What we did.** Ran **`configs/cifar/phase2_cifar_full.yaml`** for **50 epochs** on full CIFAR-10, **mutation disabled**, and recorded **final test accuracy ~0.845** and **~620k** parameters in the implementation log, with artifacts under **`runs/`**.

**Why.** Every later claim about “self-evolving” or **mutation impact** needs a **fixed-architecture** reference at **matched** data and training budget.

**Files.**

- **`runs/metrics/phase2_cifar_full_metrics.csv`** — Per-epoch metrics.
- **`runs/logs/train_phase2_cifar_full.log`** — Console capture (if saved).
- **`checkpoints/cgse_phase2_cifar_full.pt`** — Local checkpoint (large; typically **not** committed—see **`.gitignore`** patterns).

**Phase 2 status note.** CIFAR path, logging, and scheduled mutation are **working**.

**Phase 3 (started).** **`teacher`** YAML + **`load_model_weights`**: frozen **`CifarGraphNet`**, **KD** in **`training/loop.py`**; **`baseline_sear_ch_teacher_mutate.yaml`** for **teacher + mutate** control. **CGSE:** **`critic:`** + **`StructuralCritic`** in **`train.py`** (see Step 3.7, **`phase2_cifar_full_cgse.yaml`**).

---

## 5. Cross-cutting work: reproducibility, artifacts layout, and housekeeping

**Goal.** Keep **generated outputs** predictable, **documented**, and **separate** from narrative PDFs; avoid **duplicate** `runs` folders; keep the repo **clean** of caches and accidental binaries.

### Step 5.1 — `runs/` only at repository root

**What we did.** All **`training.log_csv`**, **`mutation.log_jsonl`**, and saved **console logs** are intended to live under **`runs/`** at the **repo root**, documented in **`runs/README.md`**.

**Why.** **`paper_documentation/`** should hold **human-written** paper materials (PDFs, Markdown logs, this walkthrough), not **machine-generated** CSVs that churn every experiment.

**Files.**

- **`runs/README.md`** — Explains artifact types.
- **`paper_documentation/README.md`** — Points to **`runs/`** at root.

### Step 5.2 — Normalizing legacy paths in YAML

**What we did.** **`utils/run_paths.normalize_run_artifact_path`** rewrites any legacy **`paper_documentation/runs/...`** string to **`runs/...`** (with a warning), and **`train.py`** applies this when resolving CSV and JSONL paths.

**Why.** Older configs or copy-paste mistakes should not **silently** recreate the wrong folder layout.

**Files.**

- **`utils/run_paths.py`**
- **`train.py`** — Uses **`normalize_run_artifact_path`** for **`log_csv`** and **`mutation.log_jsonl`**.

### Step 5.3 — Sentinel file `paper_documentation/runs`

**What we did.** **`paper_documentation/runs`** is a **small committed text file**, not a directory, so tools cannot create **`paper_documentation/runs/`** as a folder for logs by accident.

**Why.** A **hard guard** complements the soft rewrite in **`run_paths`**: shells and editors that bypass Python still **fail** instead of fragmenting artifact layout.

**Files.**

- **`paper_documentation/runs`** — Sentinel (read its contents for the short rationale printed there).

### Step 5.4 — Documentation set for the paper and collaborators

**What we did.** Maintained three complementary Markdown documents:

- **[`CGSE-implementation-log.md`](CGSE-implementation-log.md)** — Dated **changelog**, **run registry**, **design decision table**, **paper hooks** checklist.
- **[`CGSE-codebase-guide.md`](CGSE-codebase-guide.md)** — **File-by-file** map and execution diagrams.
- **This file** — **Narrative** phase/step walkthrough with **rationale** and **file pointers**.

**Why.** Different readers need different grains: **history**, **lookup**, **story**.

**Files.**

- **`paper_documentation/CGSE-implementation-log.md`**
- **`paper_documentation/CGSE-codebase-guide.md`**
- **`paper_documentation/CGSE-detailed-phase-walkthrough.md`** (this document)
- **`paper_documentation/README.md`** — Index of PDFs + Markdown docs.

### Step 5.5 — Repository hygiene

**What we did.** Ensured **`__pycache__`**, **`.ipynb_checkpoints`**, and **`data/`** are not treated as source; removed accidental tracking where it occurred; avoided committing large **`checkpoints/cgse_*.pt`** files (pattern in **`.gitignore`**).

**Why.** Git history stays **reviewable**; disk-heavy artifacts stay **local** or in separate storage.

**Files.**

- **`.gitignore`** (repo root)

---

## 6. What is deliberately not done yet (later phases)

- **Richer critic objectives** (multi-step returns, learned baselines, operator choice beyond one **`edge_widen`**).
- **Random mutation** control matched for compute (optional ablation).
- **Richer checkpoint resume** (Phase 0 partial).
- **ResNet-style backbone** (optional stronger student)—see **`CGSE-codebase-guide.md`**.

**Files for future work.**

- **`train.py`**, **`critics/*`** — extend policy / state / rewards as experiments demand.
- PDFs under **`paper_documentation/`** for narrative context only; **implementation scope** is **teacher vs critic**, not every variant in those notes.

---

## 7. Quick reference: config → behavior → outputs

| You want… | Config (example) | Main code path | Typical outputs |
|-----------|------------------|----------------|-----------------|
| Synthetic MLP smoke | **`configs/synthetic/base.yaml`** | **`train.py`** → **`training/synthetic.py`** → **`StudentNet`** | Checkpoint path from **`experiment.name`** |
| Phase 2 CIFAR (default file) | **`configs/cifar/phase2_cifar.yaml`** | **`train.py`** → **`training/data.py`** → **`CifarGraphNet`** | CSV if **`training.log_csv`** set |
| Full baseline (paper) | **`configs/cifar/phase2_cifar_full.yaml`** | Same as above, **`mutation.enabled: false`** | **`runs/metrics/phase2_cifar_full_metrics.csv`**, local **`.pt`** |
| Full + one widen | **`configs/cifar/phase2_cifar_full_mutate.yaml`** | Same + **`edge_widen`** after epoch 10 | **`runs/metrics/*_mutate_metrics.csv`**, **`runs/mutations/*_mutations.jsonl`**, local **`.pt`** |
| Teacher + KD (fixed arch) | **`configs/cifar/phase3_cifar_kd.yaml`** | **`train.py`** + **`training/loop.py`** KD | **`runs/metrics/phase3_cifar_kd_metrics.csv`**, local **`.pt`** |
| SEArch control: teacher + KD + widen | **`configs/cifar/baseline_sear_ch_teacher_mutate.yaml`** | KD + **`edge_widen`** after epoch 10 | **`runs/metrics/baseline_sear_ch_teacher_mutate_*.csv`**, **`runs/mutations/…jsonl`** |
| CGSE: critic-gated widen | **`configs/cifar/phase2_cifar_full_cgse.yaml`** | No teacher; **`critic:`** window + REINFORCE | **`runs/metrics/phase2_cifar_full_cgse_*.csv`**, **`runs/mutations/…jsonl`**, **`_critic.pt`** |
| Fast CPU check | **`configs/cifar/smoke/phase2_smoke.yaml`** | Subsets in **`data`** section | Small CSV / logs |

---

## Document maintenance

When you **add a phase** or **change behavior** (new model, new mutation trigger, new log field):

1. Update **[`CGSE-implementation-log.md`](CGSE-implementation-log.md)** with a **dated** subsection and registry row.
2. Update **[`CGSE-codebase-guide.md`](CGSE-codebase-guide.md)** tables and execution paths.
3. Update **this walkthrough** so the **narrative** stays true—especially §4–§7 and the config table.

*Last aligned with repository layout through Phase 3 teacher path and CGSE critic in `train.py`: 2026-04-04.*
