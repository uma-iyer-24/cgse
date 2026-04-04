# SEArch baseline paper, CGSE positioning, and evaluation parity

This note ties the **reference SEArch system** (the paper you added as `SEArch-MP1-base-paper.pdf`) to **this repository**, states how **CGSE** is meant to compete with it, and lists **tests and experiments** needed for a credible research paper.

**Visual overview (for newcomers and slides):** see the [root `README.md`](../README.md) (primary goal, plain-language section, Mermaid diagrams) and PNG figures under [`paper_documentation/figures/`](figures/) (teacher vs critic; staged evolution pipeline).

---

## 1. Reference: Liang, Xiang & Li (2025) — SEArch

**Full citation (for your manuscript).** Yongqing Liang, Dawei Xiang, Xin Li, *SEArch: A self-evolving framework for network architecture optimization*, **Neurocomputing** 651 (2025) 130980. Open access. DOI: `https://doi.org/10.1016/j.neucom.2025.130980`.

**What SEArch does (conceptual).** A **well-trained teacher** network defines a target capacity/behavior. A **student** starts from a **simple** graph and **iteratively**:

1. **Trains** under **teacher guidance** (knowledge distillation: intermediate features and predictions; they use an **attention** mechanism to transfer teacher knowledge).
2. **Scores** edges/nodes with a **modification value** to find **bottlenecks**.
3. **Evolves** the graph by **edge splitting** (inserting structure—primarily **3×3 residual separable convolutions** on a DAG), repeating until a **parameter or FLOP budget** is met.

So SEArch is **not** classical NAS over a giant supernet in the same way as DARTS; it is **iterative local search** driven by **teacher-derived signals**, combining ideas from **pruning, KD, and architecture growth**.

**Reported benchmarks (from the paper).** **CIFAR-10**, **CIFAR-100**, **ImageNet**; comparisons vs **pruning** and **KD** methods. A headline KD-style row: **ResNet-56**-class teacher, student optimized to **~0.27M parameters**, reported **93.58%** accuracy on CIFAR-10 (their Table 5), under their full search + **long final retrain** protocol (appendix: e.g. **400 epochs** SGD on CIFAR with their LR schedule—different from our default **Adam** short runs).

**Implementation details to remember for parity discussions.** They use **PyTorch**, **sep_conv_3x3**, max **stacked operations per edge** \(B_{op}=7\), **10 epochs per search stage**, then a **long retrain** of the final student. Your codebase currently uses a **different** student (**`CifarGraphNet`**), **Adam**, and **linear `edge_widen`** / **`edge_split`** on a **sequential graph**—scientifically related (self-evolving under guidance) but **not** a line-for-line reproduction of their macro search space.

---

## 2. How CGSE competes with SEArch

**One-sentence claim.** **CGSE keeps the same *type* of problem as SEArch** (iterative structural evolution of a student toward good accuracy under a budget) but **replaces the teacher’s role in bottleneck / evolution guidance** with a **critic** that reads **internal optimization state**, so evolution is **not tied to teacher feature alignment**.

**Elaboration (align with `project-doc.pdf`).** The internal design memo in **`paper_documentation/project-doc.pdf`** develops **Critic-Guided Self-Evolution**: structural updates driven by **learned critiques of training dynamics** rather than **imitation of a teacher**. That is the **direct alternative** to SEArch’s **attention + teacher supervision** for **where to split / grow**.

**Fair comparison structure.**

| Condition | Who guides *structural* changes? | Supervision for *weights* |
|-----------|----------------------------------|---------------------------|
| SEArch (Liang et al.) | Teacher (attention / modification scores) + KD | KD + labels |
| **This repo — teacher baseline** | Hand schedule or teacher-aligned training (configs with `teacher`) | CE + KD |
| **CGSE (target)** | **Critic** (scores from internal stats) | **CE on labels** (critic does **not** replace softmax logits) |

**What “compete” means empirically.** On **matched** datasets, splits, and **budgets** (parameters and/or FLOPs), report **accuracy**, **training cost** (wall-clock, epochs), and **number of structural edits**. Secondary: stability across **seeds**. You do not need to beat every row in their ImageNet table on day one; you **do** need a **clear protocol** and **honest gap analysis** where the student architecture or search operator differs.

---

## 3. Gap between this repository and the SEArch paper (honest)

| SEArch paper | This repo (today) |
|--------------|-------------------|
| DAG student, residual separable conv edges, iterative edge split | Sequential **`GraphModule`**, **`CifarGraphNet`**, **`edge_widen`** / **`edge_split`** |
| Attention module from teacher features | Frozen teacher **logit KD** only (no intermediate attention layer yet) |
| Modification value score for bottleneck edges | YAML epoch schedule or (future) **critic** scalar |
| Multi-stage search + long final retrain | Single training run; shorter epochs in configs |
| CIFAR-10/100, ImageNet, some NeRF | **CIFAR-10** first |

**Implication.** Two publication strategies:

- **Strategy A (faster).** Position CGSE as a **method contribution** on a **fixed student family** (your graph CNN), with **teacher vs critic** as the variable, and cite SEArch as **closest prior self-evolving teacher-guided** work **without claiming operator-for-operator reproduction**.
- **Strategy B (stronger parity).** **Port** their **student/operator stack** (or ResNet-56/20 teacher–student KD setting from Table 5) and match **budgets and epochs** as closely as resources allow, then run **SEArch-teacher vs CGSE-critic** on **that** stack.

Both are valid; Strategy B is more work but answers reviewers who ask “why not their exact setup?”

---

## 4. Tests and experiment checklist (research engineering)

### 4.1 Automated tests (code health)

| Test | Purpose |
|------|---------|
| **`edge_widen` / `edge_split`** on CPU and (if available) MPS/CUDA | Shape + **device** correctness after mutation |
| **KD loss** vs known tensors | Matches \(T^2\) KL formula |
| **`load_model_weights`** round-trip | Checkpoint compatibility |
| **Optional:** one epoch **forward+backward** with `teacher` enabled | Smoke integration |

Today most coverage is **scripts** under `scripts/`; **`tests/test_graph_ops.py`** should gain real **pytest** cases (see implementation log).

### 4.2 Experiment grid (paper-facing)

**Minimum viable comparison (Tier 1 — use current `CifarGraphNet`).**

1. **Fixed student, no mutation** — `phase2_cifar_full.yaml` (already have baseline acc / params).
2. **Mutate on schedule, no teacher** — `phase2_cifar_full_mutate.yaml`.
3. **Teacher + KD, no mutation** — `phase3_cifar_kd.yaml`.
4. **Teacher + KD + mutate (SEArch-style control in our stack)** — `baseline_sear_ch_teacher_mutate.yaml`.
5. **CGSE (critic, no teacher)** — **`configs/cifar/phase2_cifar_full_cgse.yaml`**: critic gates one **`edge_widen`** with **`window_start_epoch` / `window_end_epoch` = 10** (same **decision epoch** as **`once_after_epoch: 10`** in **`phase2_cifar_full_mutate.yaml`**); same **`widen_delta`**. Multi-seed: **`train.py --seed N`** → suffixed CSV/JSONL/checkpoints (see **`runs/README.md`**).

**Record per run:** config path, git commit, seed, hardware, **`runs/*.csv`**, final **val acc**, **params**, wall-clock.

**Tier 2 (parity with Liang et al. CIFAR-10 KD table, approximate).**

- **ResNet-56** teacher, **0.27M** student budget, their **optimizer schedule** (or document deviation).
- Requires **new model configs** and training loop options (SGD, 400-epoch retrain)—not implemented yet.

**Tier 3.** CIFAR-100 and ImageNet as in SEArch—long-term.

**Tier 1b / next-wave (after Tier 1 completes) — SEArch-like *protocol* + CGSE with the same outer loop.** See **§7** for the full specification: **multi-stage** training with a **parameter (or FLOP) budget**, **multiple structural edits** per run, **2–3 mutation types** with **discrete (site, operator)** choices, and a critic that **scores candidate moves** from **internal** signals and **updates from post-edit** training outcomes. This is **not** implemented in `train.py` yet; Tier 1 remains **single widen** for a clean first table.

### 4.3 Statistical rigor

- **≥3 seeds** for final claims where variance matters.
- Report **mean ± std** for accuracy and, if applicable, **final parameter count** (deterministic given fixed schedule).

---

## 5. Where to record results

- **Per-run artifacts:** `runs/` (CSV, JSONL, logs).
- **Narrative + registry:** `paper_documentation/CGSE-implementation-log.md` §5 and §7.
- **Methods text:** cite **Liang et al. (2025)** as the **primary self-evolving teacher-guided baseline**; cite **`project-doc.pdf`** for **CGSE** definition and critic formalism.

---

## 6. File locations

| Item | Location |
|------|----------|
| SEArch PDF (local copy) | `paper_documentation/SEArch-MP1-base-paper.pdf` |
| CGSE narrative / positioning | `paper_documentation/project-doc.pdf` |
| Teacher + KD configs | `configs/cifar/phase3_cifar_kd.yaml`, `configs/cifar/baseline_sear_ch_teacher_mutate.yaml` |
| Critic (v1 + planned extensions) | `critics/critic.py`, `critics/state_features.py`, `train.py` |
| This plan | `paper_documentation/SEArch-baseline-and-CGSE-evaluation-plan.md` |

---

## 7. Next protocol: SEArch-style evolution + multi-operator CGSE (planned)

This section records **what to build and test next** so the **experimental protocol** matches SEArch’s *spirit* (iterated growth under a budget, bottleneck-driven edits) while **CGSE** uses the **same outer loop** with **critic** instead of **teacher modification scores**.

### 7.1 Outer loop (align SEArch control and CGSE)

**SEArch (paper).** Repeat until budget: short **train** → **score** edges/nodes → **apply** one or more **edge splits** (their operator) → repeat; then often a **long retrain** of the final graph.

**Target protocol for “Tier 1b” experiments (both arms).**

1. **Budget:** Stop when **parameter count** or **estimated FLOPs** (or **max number of structural steps** \(K\)) is reached—match **SEArch control** and **CGSE** on the **same** \(K\) and budget caps.
2. **Stages:** Each stage = **\(T\)** epochs of student training (e.g. \(T \approx 10\) as in SEArch’s search stages, tunable). After each stage, **at least one** structural edit may fire (or zero if policy says “hold” and budget allows delay—document the rule).
3. **Teacher arm (our SEArch-style control):** After each stage, **bottleneck scoring** approximates SEArch: e.g. **teacher-aligned** signals (per-layer KD or gradient/activation statistics vs teacher when available) to rank **candidate (site, op)**; pick **argmax** or **stochastic** top-1. If attention over features is added later, that becomes closer to the paper.
4. **CGSE arm:** Same **stages** and **budget**; **no teacher** for structural choice. A **critic** (see §7.3) outputs a **distribution over a finite set** of **legal** mutations **{(location, operator)}** from **internal** training stats (+ optional **local** descriptors of each candidate site). **REINFORCE / bandit / ranking** updates use **post-edit** outcomes (e.g. **Δval** over the **next** stage or next \(m\) epochs), analogous to how Tier 1 uses one-step **Δval** but extended for **credit assignment** over stages.
5. **Final retrain (optional parity knob):** Optional **long** SGD phase on the **final** graph (SEArch-style appendix)—same for both arms when claiming close protocol parity.

**Fair comparison.** Same **student family**, **initial graph**, **budget \(K\)**, **stage length \(T\)**, **operator set**, and **seeds**. Report **accuracy**, **wall-clock**, **number of edits**, and **final params/FLOPs**.

### 7.2 Recommended mutation types (2–3 to start)

All should be **graph-safe** (validator + optimizer refresh after each edit). Prefer operators that **change capacity or depth** in complementary ways.

| Operator | Role | Repo status | Notes |
|----------|------|-------------|--------|
| **A. MLP widen** | Increases **fc** width (classifier capacity) | **`edge_widen`** on `fc1` (or chosen linear) | Already used in Tier 1. |
| **B. MLP deepen** | Inserts **identity-init** linear → extra **nonlinearity** depth without changing I/O shape of the tail | **`edge_split`** before `fc1` or before `fc2` | Implemented in `ops/edge_split.py`; **not** wired in `train.py` yet. Good complement to widen. |
| **C. Conv widen (mid-trunk)** | Widens **Conv2d** channels (e.g. `conv2`/`conv3`) with **Net2Net-style** weight replication | **New op** (not in Tier 1) | Targets **representation** bottleneck **before** flatten; closest analogue to “grow the trunk” without SEArch’s sep-conv DAG. |

**Why these three.** **Widen fc** and **deepen fc** give the critic **orthogonal** knobs (width vs depth in the head). **Conv widen** ties evolution to **spatial feature** capacity, not only the MLP—closer to where conv nets usually bottleneck. A fourth option later: **insert narrow low-rank / bottleneck conv** (more engineering).

**Instantiation points (“sites”).** For a sequential `GraphModule`, a **site** is a **node id** + **allowed operators** at that node (e.g. widen this `Linear`, split before this `Linear`, widen this `Conv2d`). The **candidate set** is the **filtered** list of **(node_id, op)** pairs that **preserve validity** (e.g. do not split the final logits layer in a breaking way—rules already partly in `edge_split`).

### 7.3 Critic: bottleneck-aware, discrete (site × operator)

**Goal.** The critic should **not** only output “mutate now / not”; it should **choose among** **typed** moves at **typed** locations, using **internal** observation of training and of the **open slots** in the student graph.

**Planned design (incremental).**

1. **Candidate enumeration** — After each stage, build the set \(\mathcal{A} = \{(s_i, o_j)\}\) of **legal** (site, operator) pairs (graph rules + budget).
2. **Features** — For each candidate (or globally): **global** vector (as today: losses, accs, epoch, param count, deltas) plus **local** scalars: e.g. **fan-in/fan-out**, **layer index**, **type** (conv vs linear), optional **gradient norm** or **activation norm** **at that module** (requires hooks—phase 2 of critic).
3. **Scoring** — **StructuralCritic v2:** e.g. **shared MLP** over global state + **embedding** of (site, op), or **per-candidate** small net; **softmax** over \(\mathcal{A}\) → sample or **argmax** for the edit.
4. **Learning signal** — **Stage-level reward:** e.g. **best val acc in the next stage** minus **baseline** trend, or **area-under-val-curve** for \(m\) epochs—reduces noise vs single-step **Δval**. **REINFORCE:** minimize \(-\log \pi(a \mid s)\,(R - b)\) with baseline \(b\) (EMA of \(R\) or value head). **Exploration:** ε-greedy or entropy bonus.
5. **Teacher arm parity** — Same \(\mathcal{A}\) and **same edit schedule budget**; replace **π** with **teacher-derived** scores (or **hand-crafted** bottleneck heuristic until attention is built).

### 7.4 Implementation checklist (engineering)

- [ ] **`train.py`:** multi-stage loop (outer **stage**, inner **epochs**); **per-stage** mutation block; **`max_mutations`** or **param cap**.
- [ ] **YAML:** `evolution: { stages, epochs_per_stage, max_param, operators: [widen_fc, split_fc, widen_conv] }`.
- [ ] **Ops:** wire **`edge_split`**; add **`edge_widen_conv`** (or similar) with device-safe layer replacement.
- [ ] **Critic v2:** candidate set input, softmax policy, stage reward, optional per-layer stats via hooks.
- [ ] **Tests:** each new op on CPU/MPS; short 2-stage smoke config.
- [ ] **Experiments:** **SEArch-control** vs **CGSE** on **Tier 1b** grid + Tier 2 ResNet/SGD when ready.

### 7.5 Relation to Tier 2 (ResNet / table parity)

**ResNet-56 + 0.27M student + long SGD** (existing **Tier 2** in §4.2) can be combined with **§7**: the **same multi-stage + multi-op** protocol runs on the **ResNet-style** student once that backbone exists. Order of work is flexible: **(i)** multi-op on `CifarGraphNet` first, or **(ii)** ResNet parity first—document which you ship in each paper revision.

---

*Last updated: 2026-04-04. §7 added: SEArch-like multi-stage protocol + multi-op CGSE roadmap.*
