# CGSE — Experiments and results guide (paper-oriented)

**Purpose.** This document is a single reference for drafting the **Methods**, **Experimental setup**, **Baselines**, **Statistical reporting**, and **Results** sections of a manuscript on **Critic-Guided Self-Evolution (CGSE)** relative to **SEArch** (Liang, Xiang & Li, *Neurocomputing* 2025, DOI [10.1016/j.neucom.2025.130980](https://doi.org/10.1016/j.neucom.2025.130980)). It consolidates the **tier ladder** (Tier 1 → 1b → 2 → 3), **parity considerations**, **test hierarchy** (from engineering checks to publication experiments), and **artifact conventions** used in this repository.

**Companion documents.** Technical gap analysis and §7 protocol detail: [`SEArch-baseline-and-CGSE-evaluation-plan.md`](SEArch-baseline-and-CGSE-evaluation-plan.md). Run commands and folder layout: [`runs/README.md`](../runs/README.md). Chronological build history and run registry: [`CGSE-implementation-log.md`](CGSE-implementation-log.md). Narrative positioning: [`project-doc.pdf`](project-doc.pdf).

---

## 1. Scientific framing (for the paper introduction)

**Problem.** We study **structural evolution** of a neural network during training: the model’s **graph** (width, depth, selected layers) may change at defined points while optimization continues, under **validation** and **optimizer-state** safety rules.

**Prior system.** **SEArch** performs **teacher-guided** iterative search: a strong **teacher**, **knowledge distillation**, **attention** over features, and **modification scores** guide **where** to **split** edges and grow a **DAG**-style student until a **parameter or FLOP budget** is met, often followed by a **long retrain**.

**Our contribution (empirical claim to support).** **CGSE** keeps the **same class of problem**—iterative growth under a budget—but replaces **teacher-derived structural guidance** with a **critic** that reads **internal training statistics** (and, in Tier 1b, **discrete choices** over legal structural actions). **Classification** remains **label cross-entropy**; the critic does **not** replace the softmax over classes.

**Controlled contrast.** Across rows of the experiment grid, we hold fixed **dataset**, **initial student family**, **optimizer and epoch budget** (per tier), **mutation operator set** (per tier), and **structural budget** where possible; we vary **only** whether **structural** decisions come from a **fixed schedule**, a **teacher + KD** pipeline, or a **critic**.

---

## 2. Honest comparison to SEArch (what we can and cannot claim)

SEArch and this codebase differ along several axes. The paper should **either** state a **method comparison** on our **CifarGraphNet** stack (**Strategy A**), **or** invest in **architectural and training parity** (**Strategy B**). Mixing claims without labeling the strategy invites reviewer pushback.

| Dimension | SEArch (paper) | This repository (typical Tier 1 / 1b) |
|-----------|----------------|----------------------------------------|
| Student topology | DAG, residual separable conv edges | Sequential **`GraphModule`**, **`CifarGraphNet`** |
| Growth operators | Edge splitting, stacked ops per edge | **`edge_widen`** (Linear), **`edge_split`** (depth), **`edge_widen_conv`** (trunk) |
| Teacher signal | Attention + KD over features | Frozen teacher, **logit KD** (no intermediate attention module) |
| Training recipe | SGD, long final retrain (e.g. hundreds of epochs in appendix) | Default **Adam**, configs in YAML (shorter runs unless Tier 2 is implemented) |
| Scope | CIFAR-10/100, ImageNet | **CIFAR-10** primary; Tier 3 = scale-out |

**Publication-safe wording.**

- **Strategy A (faster):** “We implement **self-evolving** students with **teacher vs critic** structural control on a **fixed graph CNN** (`CifarGraphNet`), and cite SEArch as the **closest teacher-guided self-evolving** reference, **without** claiming operator-for-operator reproduction.”
- **Strategy B (stronger parity):** “We additionally align **teacher capacity**, **student parameter budget**, and **optimizer schedule** with the CIFAR-10 KD setting reported in SEArch (e.g. ResNet-56 teacher, ~0.27M student, long SGD), then run the **same Tier 1b-style protocol** on that backbone.”

---

## 3. Experimental tiers (full ladder)

Tiers are **layers of evidence**, not mutually exclusive code paths. Later tiers **build on** earlier ones.

### 3.1 Tier 1 — Minimum viable comparison grid (`CifarGraphNet`, single structural decision)

**Goal.** Isolate **who controls structure** when at most **one** primary widen (or critic-gated widen) is compared at a **matched epoch**.

**Student.** `CifarGraphNet`: small CIFAR-10 CNN as a sequential graph (conv–BN–ReLU–pool ×3; flatten; `fc1`: 2048→256; `fc2`: 256→*K*). Initial parameter count **620,810** before mutations.

**Standard five rows (all on full CIFAR-10 unless noted):**

| Row | Config | Structural control | Teacher |
|-----|--------|--------------------|---------|
| Fixed | `configs/cifar/phase2_cifar_full.yaml` | None | No |
| Scheduled widen | `configs/cifar/phase2_cifar_full_mutate.yaml` | `once_after_epoch` (e.g. 10), `edge_widen` | No |
| Teacher + KD | `configs/cifar/phase3_cifar_kd.yaml` | None | Yes (frozen checkpoint) |
| Teacher + KD + mutate | `configs/cifar/baseline_sear_ch_teacher_mutate.yaml` | Same widen schedule as mutate row | Yes |
| CGSE | `configs/cifar/phase2_cifar_full_cgse.yaml` | **Critic** gates widen in a **window** matched to the mutate row’s decision epoch | No |

**Fairness constraints.** Match **epochs**, **batch size**, **learning rate**, **weight decay**, and **data subset** across rows. The **teacher** row requires a **compatible** `CifarGraphNet` checkpoint (same architecture class as student).

**Execution.** Multi-seed: `python train.py --config <path> --device auto --seed N` (see [`runs/README.md`](../runs/README.md)). Sweep: `scripts/run_tier1.sh`.

### 3.2 Tier 1b — SEArch-like *protocol* on `CifarGraphNet` (multi-stage, multi-operator)

**Goal.** Approximate SEArch’s **outer loop**: **train for T epochs** → **apply at most one (or more) structural edits** from a **legal set** → repeat, under a **fixed stage count** and **parameter trajectory**, comparing **fixed schedule** vs **critic** on the **same candidate set**.

**Representative configs.**

- Schedule: `configs/evolution/evolution_tier1b_schedule.yaml`
- Critic: `configs/evolution/evolution_tier1b_critic.yaml`
- Smoke: `configs/evolution/smoke/evolution_tier1b_smoke.yaml`, `evolution_tier1b_critic_smoke.yaml`

**Execution.** `scripts/run_tier1b.sh` (default seeds 41–43). Artifacts under `runs/tier1b/`.

**Planned extension (paper checklist).** **Teacher arm** with the **same candidate set** and **budget** as the critic (evaluation plan §7.4–7.5).

### 3.3 Tier 2 — Numeric / training parity with SEArch (CIFAR-10 KD table)

**Goal.** Compare **numbers** to settings such as SEArch **Table 5**: e.g. **ResNet-56**-class teacher, **~0.27M** student, **SGD**, **long retrain**. This tier is **specified** in [`SEArch-baseline-and-CGSE-evaluation-plan.md`](SEArch-baseline-and-CGSE-evaluation-plan.md) §4.2 but **not** fully implemented as turnkey configs in the repo at the time of this guide.

**Paper use.** Methods subsection: “**Tier 2 parity experiments** (appendix / future work)” until configs and loop options land.

### 3.4 Tier 3 — Dataset scale-out

**Goal.** CIFAR-100, ImageNet, or other benchmarks as in SEArch—**long-term** engineering.

---

## 4. Which tier for “comparing to SEArch”?

| If the paper claims… | Use primarily |
|----------------------|----------------|
| Same **idea** (teacher-guided vs critic-guided evolution), honest **gap** table | **Tier 1** (+ **Tier 1b** for richer protocol) on **`CifarGraphNet`** |
| Closest **reported accuracy / training budget** to their tables | **Tier 2** (when implemented), ideally **+ Tier 1b protocol** on that backbone |
| Closest **iterative search narrative** without ResNet work | **Tier 1b** |

**Recommendation.** Report **Tier 1** and **Tier 1b** as **primary** CGSE evidence on the **current** stack; position **Tier 2** explicitly as **parity** or **future work** unless completed.

---

## 5. Methods text — templates you can paste and fill

### 5.1 Dataset

“We evaluate on **CIFAR-10** (50,000 training images, 10,000 test images, 10 classes). We follow the default pipeline in our code: **per-channel normalization** to the standard CIFAR means and stds and **training-time augmentation** (e.g. random crop and horizontal flip) as implemented in `training/data.py`. **Metrics** labeled validation use the **official test set** (same protocol as common CIFAR practice and our implementation log).”

*Fill in:* exact augmentation list, whether a val split is ever carved from train (default: no).

### 5.2 Architectures

“**Student.** We use **`CifarGraphNet`**, a VGG-style shallow CNN expressed as a **mutable sequential graph** (`GraphModule`), with **initial parameter count** *N₀* = 620,810. **Teacher** (when enabled) is the **same architecture class**, initialized from a **frozen checkpoint** trained without mutation, following …”

*Fill in:* checkpoint path convention, teacher training config name.

### 5.3 Optimization

“We optimize with **Adam** (unless Tier 2: **SGD** with …), learning rate *η*, weight decay *λ*, batch size *B*, for *E* epochs (Tier 1) or *S* stages × *T* epochs per stage (Tier 1b). **After each structural edit**, we **re-instantiate** the optimizer (or refresh state per `utils/optimizer_utils.py`) so new parameters receive consistent moments.”

*Fill in:* numbers from the YAML actually used in the paper tables.

### 5.4 Structural edits

“We consider **operator set** 𝒪 = {**widen Linear**, **split before Linear**, **widen conv**} at allowed **sites** (Tier 1b). **Tier 1** uses a **single** `edge_widen` on `fc1` at a fixed epoch unless otherwise stated. **CGSE** uses a **critic** with policy … **exploration** ε = … **reward** …”

*Fill in:* cite `critics/`, `training/evolution_train.py`, config blocks.

---

## 6. Outcome metrics and reporting standards

### 6.1 Primary

- **Test accuracy** (our logs call it **`val_acc`**: evaluation on the CIFAR test loader).
- **Final parameter count** after all structural edits (from last CSV row or `num_parameters` column).
- **Number and type of structural edits** (from JSONL or stage logs).

### 6.2 Secondary

- **Training loss / accuracy curves** (CSV per epoch).
- **Wall-clock** and **hardware** (CPU / CUDA / MPS, GPU model).
- **Stability across seeds** (mean ± std for Tier 1 and Tier 1b headline rows).

### 6.3 Statistical protocol

- **≥3 random seeds** for any claim about **superiority** of one arm over another.
- Report **mean ± standard deviation** for final **test accuracy** and, if relevant, **final parameter count** (often deterministic given a fixed schedule; critic arm may vary).
- Fix **seeds** for **data shuffling** and **initialization** via `train.py --seed N` (see `utils/repro.py`).

---

## 7. Test and validation hierarchy (engineering → paper)

A credible paper rests on **layered** checks: unit/integration tests, fast smokes, then full runs.

### 7.1 Level 0 — Environment

- [ ] `pip install -r requirements.txt`
- [ ] PyTorch sees intended device (`cuda` / `mps` / `cpu`).
- [ ] CIFAR-10 available under `./data` (ignored by git).

### 7.2 Level 1 — Automated tests (pytest)

Run from repo root:

```bash
PYTHONPATH=. pytest tests/test_graph_ops.py -q
```

**Coverage (see `tests/test_graph_ops.py`):**

- KD loss matches **\(T^2\) × KL** formulation (`kd_distillation_loss`).
- **Checkpoint** save/load round-trip for `CifarGraphNet`.
- **`edge_widen`**, **`edge_split`**, **`edge_widen_conv`** (where applicable): shapes, forward pass, **device consistency** (CPU + CUDA/MPS if available).
- **Structural critic** smoke: state dim, forward, optional one-batch teacher+KD path as implemented.

*Paper mention:* “We ship automated tests for mutation operators, KD, and checkpoints (`tests/test_graph_ops.py`).”

### 7.3 Level 2 — Smoke configs (minutes)

| Purpose | Example config |
|---------|------------------|
| Short CIFAR train | `configs/cifar/smoke/phase2_smoke.yaml` |
| Mutate smoke | `configs/cifar/smoke/phase2_smoke_mutate.yaml` |
| CGSE smoke | `configs/cifar/smoke/phase2_cifar_cgse_smoke.yaml` |
| KD smoke (needs teacher `.pt`) | `configs/cifar/smoke/phase3_cifar_kd_smoke.yaml` |
| Tier 1b smoke | `configs/evolution/smoke/evolution_tier1b_smoke.yaml`, `evolution_tier1b_critic_smoke.yaml` |

*Criterion:* run completes without exception; CSV and optional JSONL append.

### 7.4 Level 3 — Full Tier 1 grid (hours to days)

- All five rows of §3.1 with **matched** hyperparameters.
- **≥3 seeds** per row for publication.
- Artifacts under `runs/tier1/`; checkpoints under `checkpoints/tier1/` (gitignored per `.gitignore` patterns).

### 7.5 Level 4 — Full Tier 1b (schedule vs critic)

- `scripts/run_tier1b.sh` for seeds 41–43 (or documented subset).
- Compare **final accuracy**, **edit sequences** (JSONL), **parameter trajectory**.

### 7.6 Level 5 — Tier 2 / Tier 3 (when available)

- Document new configs and **deviation** from SEArch appendix (epochs, LR schedule).

---

## 8. Suggested execution order (from project start to camera-ready)

1. **Pytest** (Level 1) on every release candidate.
2. **Smokes** (Level 2) after any change to `train.py`, ops, or evolution loop.
3. **Tier 1 fixed + mutate** — sanity-check accuracy and mutation JSONL.
4. **Tier 1 teacher** — train or obtain teacher checkpoint; run KD and teacher+mutate rows.
5. **Tier 1 CGSE** — critic row with matched window vs mutate row.
6. **Multi-seed Tier 1** — `run_tier1.sh` or scripted loops; fill **Table 1** (main result).
7. **Tier 1b** — schedule vs critic; optional teacher parity when implemented.
8. **Ablations** (appendix): ε, critic LR, window width, stage count *T*, operator ablation (remove one op from candidate set).
9. **Tier 2** (optional major milestone) — then repeat headline comparisons on parity backbone.
10. **Freeze** git commit hash, dependency versions, and **register** every run in [`CGSE-implementation-log.md`](CGSE-implementation-log.md) §5.

---

## 9. Artifacts and reproducibility checklist

For **each** run that might appear in the paper:

| Field | Record |
|-------|--------|
| Git commit | `git rev-parse HEAD` |
| Config path | e.g. `configs/cifar/phase2_cifar_full_cgse.yaml` |
| Command line | full `python train.py ...` |
| Seed | integer |
| Device / GPU | e.g. Apple MPS, NVIDIA … |
| Wall-clock | seconds or hours |
| Metrics CSV | path under `runs/` |
| Mutation JSONL | if structural edits |
| Console log | e.g. `tee` under `runs/tier1/logs/` |
| Checkpoint | path (note: large files may be gitignored) |

**Registry.** Maintain one row per run in **§5 of the implementation log** or a spreadsheet mirrored there.

---

## 10. Results section — suggested figures and tables

### Tables

- **Table 1 (main):** Tier 1 five rows × (mean ± std) test accuracy, final params, number of edits.
- **Table 2:** Tier 1b schedule vs critic × seeds; include **edit sequence** summary or appendix.
- **Table 3 (optional):** Comparison to SEArch **only** if Tier 2 or clearly labeled **non-parity** numbers.

### Figures

- Learning curves (train/val loss and accuracy) for **representative** seed per arm.
- **Parameter count vs epoch** (from CSV) for evolving runs.
- **Timeline** of structural events (from JSONL) for Tier 1b.
- Optional: critic logits / exploration rate over stages (if logged).

---

## 11. Limitations (ready-made bullets)

- **Operator and graph mismatch** vs SEArch’s DAG and separable convolutions.
- **Teacher modeling:** logit KD only vs their **attention** over features.
- **Training budget:** Adam and shorter runs vs their **long SGD** retrain unless Tier 2 matches.
- **Single dataset** in core experiments (CIFAR-10) unless Tier 3 is done.
- **Critic** may require **tuning** (ε, architecture, reward horizon); report sensitivity.

---

## 12. Appendix text — pytest and code availability

“We provide reproducible training code and configs in the supplementary repository. Structural operators are covered by unit tests (`tests/test_graph_ops.py`) verifying tensor shapes, device placement after edits, and distillation loss correctness.”

---

*This guide is meant to evolve: when Tier 2 configs land or the teacher Tier 1b arm is added, update §3 and §8 and bump the date below.*

*Last updated: 2026-04-04.*
