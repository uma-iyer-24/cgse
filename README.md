# CGSE — research code for structurally mutable neural networks

**CGSE** (working name: *critic- or controller-guided structural evolution*) is a research codebase for training **student networks whose architecture can change during optimization**—for example, widening fully connected layers or inserting depth—while keeping training numerically stable through **graph validation** and **optimizer state handling**.

This repository is intended for **reproducible experiments** and a future publication comparing **learned or rule-based structural policies** against **fixed architectures** and other baselines (e.g. teacher-guided structural search). The long-term goal is a **controller** that proposes mutations from training signals under a budget; **what you can run today** is a **fully implemented mutation substrate** and **supervised CIFAR-10 training** with optional **scheduled** widening and **structured logging**.

If you have **no prior context**, read this file first, then the **[detailed phase walkthrough](paper_documentation/CGSE-detailed-phase-walkthrough.md)** for a step-by-step narrative of design choices and file locations.

---

## Motivation

Standard deep learning fixes a model’s structure before training. **Structural evolution** instead adjusts width or depth **during** training so capacity can grow where it helps. Doing this safely requires:

- An explicit **computational graph** (not an opaque blob of layers) so edits have a well-defined target.
- **Mutation operators** that preserve behavior approximately when possible (e.g. Net2Net-style widening with weight copying).
- **Re-validation** after each edit (shapes, forward pass).
- **Optimizer refresh** when new parameters appear, so momentum and variance estimates are not applied to stale tensors.

CGSE implements that substrate in PyTorch and connects it to a **real vision benchmark** (CIFAR-10) so results are meaningful, not only toy demonstrations.

---

## What is implemented in this repository

| Area | Status | Description |
|------|--------|-------------|
| **Mutable graph student** | Active | `GraphModule`: ordered nodes (`nn.ModuleDict` + execution list), sequential forward. |
| **Structural operators** | Active | `edge_widen` (widen a `Linear`, fix downstream linears); `edge_split` (identity-style deepen). |
| **Safety & training integration** | Active | Forward validation, `refresh_optimizer` after mutations. |
| **CIFAR-10 training** | Active | `CifarGraphNet` (small CNN as a graph); train/test metrics; YAML configs. |
| **Experiment logging** | Active | Per-epoch CSV (`training.log_csv`); optional JSONL per mutation event (`mutation.log_jsonl`). |
| **In-loop mutation (Phase 2)** | Active | Config-driven: e.g. widen once after a chosen epoch, then continue training. |
| **Teacher / distillation / learned controller** | Not in code yet | Described in `paper_documentation/project-doc.pdf` and phase roadmap PDFs. |
| **Pytest suite** | Minimal | Demos and stress scripts under `scripts/`; `tests/` reserved for automated tests. |

See **[Phase status](paper_documentation/CGSE-implementation-log.md#3-phase-status)** in the implementation log for a compact roadmap table.

---

## Repository layout

```
cgse/
├── README.md                 ← This file
├── requirements.txt
├── train.py                  # Main entry: load YAML, train, optional mutation, checkpoint
├── configs/                  # Experiment YAML (CIFAR full baseline, mutate ablation, smoke, synthetic MLP)
├── models/                   # GraphModule, MLP student, CIFAR CNN student
├── ops/                      # edge_widen, edge_split
├── training/                 # CIFAR loaders, train/eval loop, synthetic data
├── utils/                    # seeds, checkpoint, validators, mutation logging, optimizer refresh
├── scripts/                  # Standalone mutation / robustness demos (not pytest)
├── paper_documentation/      # Paper PDFs, implementation log, codebase guide, narrative walkthrough
├── runs/                     # Metrics CSV, JSONL, console logs (see runs/README.md)
├── checkpoints/              # Saved weights: checkpoints/<experiment.name>.pt (large; often gitignored)
└── data/                     # Local dataset cache (gitignored)
```

---

## Installation

**Requirements:** Python 3.10+ recommended, [PyTorch](https://pytorch.org/) with a working backend (**CUDA**, **Apple MPS**, or **CPU**).

```bash
git clone https://github.com/uma-iyer-24/cgse.git
cd cgse
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically on first use (via `torchvision`) into `./data` unless you change `data.root` in the YAML.

---

## Quick start

**Default Phase 2 CIFAR run** (configurable subset, device from YAML):

```bash
python train.py --config configs/phase2_cifar.yaml
```

**Override device** (e.g. CPU):

```bash
python train.py --config configs/phase2_cifar.yaml --device cpu
```

**Full CIFAR-10 train/test, fixed architecture (paper-style baseline):**

```bash
python train.py --config configs/phase2_cifar_full.yaml
```

**Same setup, plus one mid-training widen** (ablation vs baseline; separate CSV/JSONL names in YAML):

```bash
python train.py --config configs/phase2_cifar_full_mutate.yaml
```

**Fast smoke tests** (small subsets):

```bash
python train.py --config configs/phase2_smoke.yaml
python train.py --config configs/phase2_smoke_mutate.yaml
```

**Synthetic MLP only** (no CIFAR; useful for pipeline checks):

```bash
python train.py --config configs/base.yaml
```

---

## Outputs and reproducibility

- **Per-epoch metrics** append to the path in `training.log_csv` (columns include loss, accuracy, parameter count, mutation flag). Typical location: **`runs/*.csv`**.
- **Mutation events** (when enabled) append one JSON object per line to `mutation.log_jsonl` under **`runs/`**.
- **Checkpoints** are written at the end of training to **`checkpoints/<experiment.name>.pt`**. Patterns like `checkpoints/cgse_*.pt` are **gitignored** so binaries do not bloat the repository; keep them locally or in separate artifact storage.
- **Seeds** are set from `training.seed` in each config via `utils/repro.set_seed`.

For artifact conventions and the distinction between **`runs/`** (machine output) and **`paper_documentation/`** (narrative and PDFs), see **`runs/README.md`** and **`paper_documentation/README.md`**.

---

## Documentation for researchers and collaborators

| Document | Purpose |
|----------|---------|
| [`paper_documentation/project-doc.pdf`](paper_documentation/project-doc.pdf) | Full research narrative, positioning, experimental design. |
| [`paper_documentation/phase-plan-overview.pdf`](paper_documentation/phase-plan-overview.pdf) | Phased roadmap (Phases 0–8). |
| [`paper_documentation/CGSE-implementation-log.md`](paper_documentation/CGSE-implementation-log.md) | Living changelog, run registry, design decisions, paper checklist. |
| [`paper_documentation/CGSE-codebase-guide.md`](paper_documentation/CGSE-codebase-guide.md) | File-by-file map and execution paths. |
| [`paper_documentation/CGSE-detailed-phase-walkthrough.md`](paper_documentation/CGSE-detailed-phase-walkthrough.md) | Long-form English walkthrough: goals, steps, rationale, file pointers. |

---

## Validation and tests

- **Scripts** under **`scripts/`** exercise mutations, preservation, live train–mutate–train loops, and robustness sequences. Run from the repo root (see each file’s docstring or header).
- **Automated pytest** coverage is intentionally **limited** today; expanding **`tests/`** is part of the engineering plan alongside larger experiments.

---

## Limitations (read before citing numbers)

- Reported accuracies depend on **config, seed, hardware, and PyTorch version**. Always cite the **exact YAML**, **commit hash**, and **environment** used.
- The current **mutation policy** in Phase 2 is **hand-scheduled in YAML** (e.g. one widen after epoch *k*), not a learned controller.
- The CIFAR student is a **small CNN** chosen for clarity and mutation plumbing; scaling to stronger backbones (e.g. ResNet-style) is a **planned** extension described in the codebase guide.

---

## Contributing and contact

Use issues or pull requests for bug reports and improvements. For **paper-related** questions, refer to **`project-doc.pdf`** and the implementation log so terminology stays aligned with the manuscript.

---

## Citation

If you use this code in research, please cite the **forthcoming or associated publication** once available, and reference this repository URL and commit hash. A `CITATION.cff` file may be added when the canonical citation is finalized.
