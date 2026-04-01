# CGSE — Implementation & experiment log

**Purpose.** Single, paper-ready record of *what was built*, *why*, *how to reproduce it*, and *what training runs showed*. This document is updated as implementation proceeds.

**Audience.** Future you, collaborators, and the Methods / Experiments / Appendix sections of the paper.

**Conventions.**

- Dates use **ISO 8601** where relevant (`2026-04-02`).
- Code paths are relative to the repository root unless stated otherwise.
- Training commands are copy-pasteable; full console output is archived under [`runs/`](runs/) and summarized here.

---

## Table of contents

1. [Project snapshot (for the paper)](#1-project-snapshot-for-the-paper)
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

**This codebase currently emphasizes:** Phase 1-style **safe mutations**, **graph validation**, **optimizer refresh** after structural edits, and (upcoming) Phase 2 **real-data student training** with logged runs.

---

## 2. Repository map (implementation-relevant)

| Path | Role |
|------|------|
| `models/graph.py` | `GraphModule`, execution order, widen / insert helpers. |
| `models/student.py` | `StudentNet` (MLP-style student), `deepen_after`. |
| `ops/edge_split.py`, `ops/edge_widen.py` | Structural mutation operators (deepen/split, widen). |
| `utils/graph_validator.py` | Post-mutation shape / integrity checks. |
| `utils/optimizer_utils.py` | `refresh_optimizer` after new parameters appear. |
| `training/loop.py` | Training step(s); to be extended for real dataloaders + metrics. |
| `train.py` | Entry point; config-driven. |
| `configs/base.yaml` | Default hyperparameters and model dims. |
| `scripts/` | Validation and stress tests (`validate_mutation`, `test_*`). |
| `paper_documentation/` | This log, PDFs, and archived run logs under `runs/`. |

---

## 3. Phase status

| Phase | Theme | Status | Notes |
|-------|--------|--------|--------|
| 0 | Env, config, minimal train, checkpointing | **Partial** | Synthetic loop exists; resume + config-in-checkpoint TBD. |
| 1 | Mutable graph, mutations, validation, live training | **Largely done** | Core ops + tests; API consolidation optional. |
| 2 | CIFAR (or subset), student baseline, mutations in-loop, logging | **Next** | Data pipeline + real metrics + run artifacts. |
| 3+ | Teacher, KD, CGSE controller | **Not started** | Documented in `project-doc.pdf` / phase plan. |

---

## 4. Chronological implementation log

### 2026-04-02 — Documentation scaffold

- **Added** this file and `paper_documentation/README.md`, plus `paper_documentation/runs/README.md` for raw log storage.
- **Rationale.** Separate long-form PDFs (`project-doc.pdf`, `phase-plan-overview.pdf`) from an **append-only implementation record** suitable for Methods/Reproducibility.
- **Next (pending your go-ahead).** Phase 2: dataset integration, real training loop, baseline run(s), mutation hook + logs archived under `paper_documentation/runs/`.

---

## 5. Training runs registry

Use one row per meaningful run (baseline, ablation, or production experiment). Paste a short excerpt in this doc; store full output in `runs/`.

| Run ID | Date | Config / command | Notes | Log file |
|--------|------|-------------------|-------|----------|
| — | — | — | No registered runs yet. | — |

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

---

## 7. Paper hooks (open items)

Items to fill as experiments land:

- [ ] Exact **dataset** (CIFAR-10 full vs subset) and **augmentation** recipe.
- [ ] **Student architecture** (MLP vs small CNN) and parameter counts over time.
- [ ] **Mutation schedule** (fixed epoch vs signal-driven) for Phase 2 baseline.
- [ ] Comparison table: fixed vs random mutation vs teacher (Phase 3) vs CGSE (Phase 7).
- [ ] Seeds, wall-clock, and hardware for each reported result.

---

*End of current log. New sessions append new dated subsections under §4 and new rows under §5.*
