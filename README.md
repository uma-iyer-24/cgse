# CGSE — research code for structurally mutable neural networks

**CGSE** (*Critic-Guided Self-Evolution*) is a research codebase for training **student networks whose architecture can change during optimization**—for example, widening fully connected layers or inserting depth—while keeping training numerically stable through **graph validation** and **optimizer state handling**.

### Experimental design (teacher baseline vs CGSE)

We isolate **one** contrast:

| Arm | Non-label guidance | Structural edits |
|-----|--------------------|------------------|
| **SEArch-style control (teacher)** | A **frozen teacher** provides **knowledge distillation** while the student trains. The student may also undergo **scheduled mutations** (e.g. widen). This is the **external-guidance baseline**. |
| **CGSE** | **No teacher.** A **structural critic** (trained on **internal** optimization statistics) **replaces** the teacher for **when/where to mutate**. **Label cross-entropy** still trains the student for classification; the critic does **not** replace class logits. |

Broader brainstorming from early drafts (multi-objective arbitration, predictive architecture selection, staged teacher–critic hybrids, etc.) is **out of scope** for this repository—we only implement toward **teacher control → critic replaces teacher**.

**In code now:** mutations, CIFAR, logging, **`teacher` + KD** in `train.py`, configs **`phase3_cifar_kd.yaml`** (teacher, no mutation) and **`baseline_sear_ch_teacher_mutate.yaml`** (teacher + KD + mutate). **`critics/critic.py`** defines **`StructuralCritic`**; **training it** and **using it to gate mutations** (instead of YAML) is the next step.

Background materials: `paper_documentation/`. Step-by-step implementation story: **[detailed phase walkthrough](paper_documentation/CGSE-detailed-phase-walkthrough.md)**.

---

## Motivation (engineering)

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
| **Frozen teacher + KD (SEArch-style control)** | Active | **`teacher`** in YAML: CE + KD vs a saved **`CifarGraphNet`** (e.g. **`configs/phase3_cifar_kd.yaml`**). |
| **Teacher + KD + mutate (full control baseline)** | Active | **`configs/baseline_sear_ch_teacher_mutate.yaml`** — same mutation schedule as Phase 2 mutate, plus teacher. |
| **CGSE critic (replaces teacher)** | In progress | **`StructuralCritic`** in **`critics/critic.py`**; mutation gating + critic training in **`train.py`** not wired yet. |
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
├── critics/                  # StructuralCritic (CGSE; replaces teacher for mutation decisions)
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

**SEArch-style teacher baselines** (need `checkpoints/cgse_phase2_cifar_full.pt` from the Phase 2 full run):

```bash
python train.py --config configs/phase3_cifar_kd.yaml              # teacher + KD, fixed arch
python train.py --config configs/baseline_sear_ch_teacher_mutate.yaml  # teacher + KD + widen (control for CGSE)
python train.py --config configs/phase3_cifar_kd_smoke.yaml       # tiny subset, CPU
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
| `paper_documentation/project-doc.pdf` | Full research narrative (printable): NAS vs SEArch, CGSE variants, critic formalism, teacher baselines, risks—excerpts summarized above. |
| `paper_documentation/phase-plan-overview.pdf` | Phased roadmap (Phases 0–8). |
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
- The current **mutation policy** in Phase 2 is **hand-scheduled in YAML** (e.g. one widen after epoch *k*), not the **critic-guided** policy described in the research manuscript.
- The manuscript itself records **open critiques** of the CGSE idea: e.g. the critic’s intelligence is still **designed** (choice of statistics, operator set, meta-loss); the critic may collapse to a **heuristic** without causal guarantees; **growth-only** bias if removal is weak; and **optimization health** need not equal **task generalization**. This codebase does **not** settle those questions empirically yet.
- The CIFAR student is a **small CNN** chosen for clarity and mutation plumbing; scaling to stronger backbones (e.g. ResNet-style) is a **planned** extension described in the codebase guide.

---

## Contributing and contact

Use issues or pull requests for bug reports and improvements. For **paper-related** terminology, keep the implementation log and the PDFs under `paper_documentation/` aligned.

---

## Citation

If you use this code in research, please cite the **forthcoming or associated publication** once available, and reference this repository URL and commit hash. A `CITATION.cff` file may be added when the canonical citation is finalized.
