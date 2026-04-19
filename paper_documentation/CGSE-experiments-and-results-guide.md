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

**Goal.** Compare **numbers** to settings such as SEArch **Table 5**: e.g. **ResNet-56**-class teacher, **~0.27M** student, **SGD**, **long retrain**.

**Status (implemented, approximate).** This repo includes a CIFAR ResNet track:

- Teacher: `configs/tier2/teacher_resnet56_cifar10.yaml`
- Student (CE): `configs/tier2/student_resnet20_cifar10_ce.yaml`
- Student (KD): `configs/tier2/student_resnet20_cifar10_kd.yaml` (requires teacher checkpoint)
- Sweep: `scripts/run_tier2.sh`

This Tier 2 track is **recipe/scale parity** (teacher strength, student size, long SGD), not a reproduction of SEArch’s DAG + sep-conv search space.

**Tier 2 additional rows (structural edits on ResNet).** To keep Tier 2 comparable while adding “evolution” capability, we implement ResNet-safe edits:

- `resnet_head_widen`: Net2Net-style widening of the classifier head (function-preserving at init).
- `resnet_insert_block`: deepen by inserting an identity-initialized residual block into `layer3` (function-preserving at init).
- `resnet_layer3_widen`: widen `layer3` channel width by adding zero-initialized channels and updating downstream layers to ignore them (function-preserving at init).

Tier 2 can therefore include:

- **Scheduled** head widen (`student_resnet20_cifar10_sched_headwiden.yaml`)
- **CGSE** head widen (binary gating) (`student_resnet20_cifar10_cgse_headwiden.yaml`)
- **CGSE multi-op** discrete action choice (`student_resnet20_cifar10_cgse_multiop.yaml`), where the critic chooses one of `{noop, resnet_head_widen, resnet_insert_block}` at the decision epoch.
- **CGSE multi-op** discrete action choice (`student_resnet20_cifar10_cgse_multiop.yaml`), where the critic chooses one of `{noop, resnet_head_widen, resnet_layer3_widen, resnet_insert_block}` at the decision epoch.

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

## 4.1 CGSE vs SEArch — where CGSE can be better (paper-ready table)

This table is meant for the **Discussion** or a “Why CGSE?” paragraph in **Experiments**. It highlights *meaningful* comparison axes where CGSE can legitimately outperform SEArch/NAS-style approaches **without** requiring operator-for-operator reproduction.

| Dimension / metric | CGSE (this repo) | SEArch (Liang et al., 2025) | Why it matters / real-world applications |
|---|---|---|---|
| **Teacher requirement** | **Optional**. Core CGSE runs **teacher-free** (`teacher.enabled: false`). | **Required** (teacher-guided evolution + attention/KD signals). | Teacher models may be unavailable (privacy/IP), too costly, or hard to maintain across domains; teacher-free methods are easier to deploy and reproduce. |
| **Teacher compute overhead** (`teacher_forwards`) | **0** for CGSE arms; **>0** only when you intentionally enable KD. Logged as cumulative `teacher_forwards`. | Substantial by design (teacher inference is central to guidance). | In production or low-budget research, teacher inference cost can dominate; reducing it can be a significant practical improvement. |
| **Accuracy per compute** (`val_acc` vs `wall_seconds`) | Report accuracy as a function of **wall time**, including any auxiliary compute. Logged as `epoch_seconds` / `wall_seconds`. | Often reported as accuracy after long training/search; auxiliary compute is not always priced into headline numbers. | Enables “time-to-accuracy” comparisons (e.g., fast iteration, limited GPU-hours) where CGSE may win even if peak accuracy is close. |
| **Teacher-free constraint** | Defined and measurable: `teacher_forwards = 0` while still allowing evolution (critic uses internal signals). | Not applicable (method assumes teacher-derived guidance). | A legitimate win condition: “best achievable accuracy under no-teacher constraint,” relevant to edge devices, regulated data, and teacher scarcity. |
| **Search / evolution efficiency** (edits-to-threshold) | Can report **# edits / stages** to reach a target accuracy (Tier 1b) using JSONL mutation logs and staged curves. | Iterative growth under a budget, but tied to teacher scoring and often followed by long retrain. | Measures how quickly the method finds useful edits; relevant when structural evaluations are expensive. |
| **Robustness to teacher mismatch** | Not dependent on teacher alignment; critic uses internal optimization dynamics. | Can be sensitive to teacher/student/domain mismatch. | Real datasets shift; teacher mismatch can degrade KD-driven guidance. |
| **Operational simplicity** | Single training loop + optional critic; fewer external dependencies. | More moving parts (teacher features/attention, guidance machinery). | Reduced engineering overhead helps reproducibility and reduces failure modes; easier for collaborators to run. |
| **Raw accuracy parity claim** (Tier 2) | **Measured via Tier 2** (ResNet-56 teacher / ResNet-20 student, SGD + schedule). | Paper-reported tables on CIFAR/ImageNet. | This is the “headline table” axis reviewers expect; it’s harder but now well-posed in Tier 2. |

**How we report this in Tier 2.** For Tier 2 configs (`configs/tier2/`) we will report both (i) student test accuracy (`val_acc`) at the end of training and (ii) efficiency/dependency metrics (`teacher_forwards`, `wall_seconds`, accuracy-vs-time curves). That supports claims like “similar accuracy at lower auxiliary compute,” which is a meaningful improvement even when teacher baselines are strong.

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
- **Teacher compute / dependency cost** (Tier 2 and teacher arms): teacher forward passes, and/or wall-clock with and without teacher.
- **Search / evolution efficiency**: number of edits or stages required to reach a target accuracy (“edits-to-threshold”).

### 6.3 Statistical protocol

- **≥3 random seeds** for any claim about **superiority** of one arm over another.
- Report **mean ± standard deviation** for final **test accuracy** and, if relevant, **final parameter count** (often deterministic given a fixed schedule; critic arm may vary).
- Fix **seeds** for **data shuffling** and **initialization** via `train.py --seed N` (see `utils/repro.py`).

### 6.4 Efficiency metrics (where CGSE can beat teacher/NAS legitimately)

These are the most defensible “CGSE wins” metrics because they price in what teacher-guided systems and many NAS pipelines assume (extra models, extra compute):

- **Teacher-free constraint**: accuracy under a rule “no teacher model / no teacher forward passes.” SEArch-style guidance is not applicable here; CGSE remains applicable.
- **Accuracy per compute**: accuracy vs **wall-clock seconds** (or GPU-hours), counting any teacher forward-pass overhead in the teacher/KD baselines.
- **Teacher forward-pass count**: number of teacher forward passes performed during training (proxy for extra inference compute).
- **Edits-to-threshold**: minimum structural edits needed to reach a target accuracy (or area under accuracy-vs-edits curve).

**Implementation in this repo (logged in CSV):**

- `epoch_seconds`: wall time for the epoch (train + eval).
- `wall_seconds`: cumulative wall time since start of run.
- `train_steps`: cumulative number of training batches processed.
- `teacher_forwards`: cumulative number of teacher forward passes (equals `train_steps` when KD teacher is enabled; 0 otherwise).
- `optimizer`, `lr`: for reproducibility (Tier 2 parity uses SGD + schedule).

These fields are appended by `train.py` / `training/evolution_train.py` via `utils/metrics_csv.py`.

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

## 13. Ready-to-paste: honest gap vs SEArch (Related Work / Methods)

*Use verbatim or shorten; cite Liang, Xiang & Li (2025), DOI 10.1016/j.neucom.2025.130980.*

> **Comparison to SEArch.** SEArch (Liang et al., 2025) performs teacher-guided self-evolution on a DAG student with attention-weighted knowledge transfer and iterative edge splitting toward a parameter budget, often followed by a long SGD retrain. Our work targets the same *class* of problem—iterative structural improvement under a budget—but implements a **sequential** `CifarGraphNet` with **Net2Net-style widening**, **depth insertion**, and **conv widening** under graph validation, uses **logit-level** distillation (no intermediate attention module), and defaults to **Adam** with a fixed epoch budget unless we report an explicit Tier-2 parity protocol. We therefore do **not** claim operator-for-operator reproduction of SEArch; we position SEArch as the closest **teacher-guided self-evolving** baseline and isolate a single empirical contrast—**teacher-driven vs critic-driven structural decisions**—on our shared student family and operator set.

---

## 14. Draft results snapshot (Tier 1, seeds 41–43)

*Numbers from `runs/tier1/metrics/` (aggregated 2026-04-04). **Val acc** = accuracy on the CIFAR-10 test split as logged.*

| Arm | Mean best val acc | Std (across seeds) |
|-----|-------------------|---------------------|
| Fixed (no mutation) | 84.48% | 0.14 pp |
| Scheduled widen | 84.38% | 0.16 pp |
| Teacher + KD | 84.87% | 0.21 pp |
| Teacher + KD + widen | 84.95% | 0.34 pp |
| CGSE (critic) | 84.68% | 0.41 pp |

### 14.1 Results prose (Tier 1 + Tier 1b)

**Full paragraphs** suitable for a Results section (including Tier 1b seed-41 narrative and reproducibility stubs) live in **[`draft-results-for-paper.md`](draft-results-for-paper.md)**. Refresh that file when new seeds land; keep this guide as the **methods** checklist.

**One-line takeaway (Tier 1):** Teacher-inclusive arms edge **mean best** val accuracy slightly; **CGSE** stays within ~0.3–0.7 pp on that metric without teacher guidance for **when** to widen.

**Tier 1b (incomplete multi-seed as of 2026-04-05):** **Seed 41** complete for **both** schedule and critic; **schedule seed 42** was still training; critic seeds **42–43** not yet finished — see **`CGSE-implementation-log.md`** and the **status line** on the [`web/`](../web/) preview after `python scripts/build_results_site.py`.

### 14.2 Hardware / wall-clock

Fill the table in **`draft-results-for-paper.md`** with your machine. Sweeps used **`device: auto`** in YAML (CUDA → MPS → CPU). Per-job wall-clock varies; Tier 1b full CIFAR (~50 epochs) was on the order of **~1 hour per arm per seed** on a typical laptop GPU (indicative only — **measure** on your hardware).

---

## 15. Deferred (not required for Tier 1 demo)

- **Tier 2:** ResNet-56 / ~0.27M student / long SGD — see §3.3.
- **Tier 1b teacher arm:** same candidate set as critic under `evolution:` (evaluation plan §7).
- **Further ablations:** ε, critic hidden size, stage count, operator subsets.

---

*This guide is meant to evolve: when Tier 2 configs land or the teacher Tier 1b arm is added, update §3 and §8 and bump the date below.*

*Last updated: 2026-04-05.*
