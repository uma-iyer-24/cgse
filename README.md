# CGSE — research code for structurally mutable neural networks

**CGSE** (*Critic-Guided Self-Evolution*) is a research codebase for training **student networks whose architecture can change during optimization**—for example, widening fully connected layers or inserting depth—while keeping training numerically stable through **graph validation** and **optimizer state handling**.

Long-form narrative PDFs live under `paper_documentation/` alongside Markdown logs; the excerpts below summarize the **research intent** distilled from that manuscript.

**What you can run in this repository today** is a **fully implemented mutation substrate** and **supervised CIFAR-10 training** with optional **YAML-scheduled** widening and **structured logging**. The **learned critic** and full **closed-loop CGSE controller** described below are **not trained here yet**; they are the target of later implementation phases (roadmap PDF in the same folder).

If you have **no prior context**, read this file first, then the **[detailed phase walkthrough](paper_documentation/CGSE-detailed-phase-walkthrough.md)** for a step-by-step narrative of design choices and file locations.

---

## Research context: neural architecture, SEArch, the critic, and the CGSE goal

The following quotes and summaries are taken from the project’s internal research manuscript (working document, `paper_documentation/`).

### Neural architecture search (NAS) vs in-training structural change

**Neural architecture search** usually treats **architecture choice** as an outer **search** over candidates (RL, evolution, weight-sharing supernets, etc.): pick a structure, train weights, score, repeat. **Self-evolving architecture (SEArch)** instead stresses **one training trajectory** in which **connectivity or capacity can change online**—closer in spirit to “the model edits itself while it learns.”

The manuscript contrasts NAS-style search with **continuous structural self-evolution** tied to ordinary SGD:

> Unlike neural architecture search, our method performs continuous, low-overhead **architectural self-evolution during training**, guided purely by **internal learning dynamics**.

**CGSE** is positioned as **not** “evaluate many architectures in an outer loop,” but as **closed-loop control**: architecture updates react to **state drawn from the same run** (loss curvature, gradient behavior, representation diagnostics, resource use), rather than only to an external search policy.

### Formal view: critic and structural updates

The document states that **CGSE treats architecture evolution as a closed-loop control system driven by internally learned critics, not external signals**. A critic maps a **state** (training dynamics, representation diagnostics, resource usage, constraint signals) to **scores**; a **structural update rule** maps those scores to **discrete graph edits** (e.g. localized widening or splitting). **Weight updates** follow the ordinary task loss; the critic is trained with a **meta-objective** aimed at **system-level improvement**, not label imitation.

On the distinction from teacher–student guidance:

> The critic **does not say what to predict**, only whether the system is **structurally underperforming**.

Teacher–student is summarized as **external reference, fixed target, imitative signal**; CGSE as **self-referential evaluation of internal dynamics** and **no external oracle** for structural decisions—though the same manuscript later discusses **hybrids** (teacher safety + critic-driven growth) as a pragmatic design.

### Lay intuition: learner vs critic

The manuscript poses the motivating question:

> Can a model tell, by looking at **its own training behavior**, whether its **structure** is wrong?

It then separates **two roles**:

> **CGSE separates two roles:** **Learner:** updates weights as usual. **Critic:** watches how learning is going. The critic doesn’t know labels. It doesn’t know the “right answer.” It only knows things like: learning has stalled, representations are collapsing, gradients are unstable, capacity is underused. When those signals cross certain thresholds, the architecture adapts.

The same section emphasizes:

> **Architecture change is triggered by how learning feels**, not by what the model predicts. That’s the conceptual leap.

### Relation to teacher-as-critic (TCNO) and “condensed” CGSE

The manuscript groups several teacher-centric *-Search* ideas and describes **Teacher as Critic, Not Oracle (TCNO-Search)** as a setting where the teacher **does not supply targets** but **critiques** (e.g. bottleneck severity). **CGSE** is presented as a **condensed** direction that **absorbs** ideas from TCNO and teacher-free self-referential variants, with stated **core novelty**: **no imitation**, **critic-based structural signals**, and **autonomous architecture growth**—still subject in practice to engineering choices (which statistics enter the critic, which operators are allowed).

### Where this repository stands

**In code today**, the **critic module is not trained**; see the placeholder **`critics/critic.py`**. Experiments use **YAML schedules** (e.g. widen once after epoch *k*) so that **safe mutations, metrics, and baselines** exist before the full critic loop is implemented. **Knowledge distillation**, **teacher networks**, and **TGCE-style** staged teacher/critic control appear in the manuscript as **design options**, not as defaults in `train.py`.

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
| **Teacher / distillation / learned critic–controller** | Not in code yet | Described in the manuscript PDFs under `paper_documentation/`; stub **`critics/critic.py`**. |
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
