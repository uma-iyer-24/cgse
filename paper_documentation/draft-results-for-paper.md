# Draft — Results text for the manuscript (copy/edit)

**Purpose.** Paste-ready paragraphs and bullets derived from committed metrics (`runs/tier1/`, `runs/tier1b/`). **Update** after new seeds finish; re-run `python scripts/build_results_site.py` for the static preview.

**Last aligned with repo data:** 2026-04-05 (Tier 1: seeds 41–43; Tier 1b: **complete** schedule + critic for seed **41** only; schedule seed **42** still training).

---

## Results — Tier 1 (five-arm grid)

We trained **`CifarGraphNet`** on **CIFAR-10** under five conditions with **matched** optimization hyperparameters (full training set, 50 epochs, batch size 128, Adam with learning rate and weight decay fixed across arms). **Validation accuracy** refers to **classification accuracy on the official test split**, logged each epoch. **Structural** interventions differ by arm: none; a **single scheduled widen** of the first fully connected layer after epoch 10; **frozen-teacher logit distillation** with and without the same widen; and **CGSE**, where a **scalar critic** gates the same widen at the **same epoch window** without teacher logits.

Across **three independent seeds** (41–43), **mean ± standard deviation** of the **best validation accuracy** achieved during each run was: **84.48 ± 0.14%** (fixed); **84.38 ± 0.16%** (scheduled widen); **84.87 ± 0.21%** (teacher + KD); **84.95 ± 0.34%** (teacher + KD + widen); **84.68 ± 0.41%** (CGSE). Teacher-inclusive rows attained slightly higher **peak** validation accuracy in this grid, while **CGSE** remained **competitive** despite using **no teacher** for **structural** timing—classification is trained only with **label cross-entropy** plus the critic’s policy gradient on validation improvement, as implemented in the repository.

*Optional sentence for limitations forward-reference:* Full **Tier 2** parity with Liang et al. (2025) (ResNet-scale teacher, sub-megabyte student, long SGD retrain) is **out of scope** for this table; we report **Strategy A** (method comparison on our graph CNN) as defined in **`CGSE-experiments-and-results-guide.md`**.

---

## Results — Tier 1b (multi-stage schedule vs critic)

We additionally evaluated a **multi-stage** protocol (**five stages** of ten epochs each, **50 epochs** total) with **three structural edits** drawn from a shared candidate set (**conv widen**, **FC widen**, **split before logits**). The **schedule** arm applies a **fixed** sequence of edits; the **critic** arm samples edits with an **ε-greedy discrete policy** trained with **stage-level feedback** (repository default).

With **seed 41** (complete runs for **both** arms), **best** validation accuracy was **84.17%** (schedule) versus **84.14%** (critic); **final** epoch validation accuracy was **81.33%** (schedule) versus **83.99%** (critic), with **different final parameter counts** (**935,914** vs **926,346**) reflecting **different edit orders** (see mutation JSONL). **Multi-seed statistics** for Tier 1b require completed CSVs for additional seeds; **schedule seed 42** was **in progress** at the time of this draft.

---

## Results — Tier 2 (ResNet parity track; paper runs)

Tier 2 implements an **approximate parity** setup to SEArch’s CIFAR-10 teacher–student table: a **ResNet-56** teacher and a **ResNet-20** student at **~0.27M parameters**, trained with **SGD + multistep LR**. These runs are logged under `runs_paper/tier2/metrics/` and enable apples-to-apples comparisons between **teacher/KD** baselines and **teacher-free** CGSE variants on the same backbone and optimizer schedule.

**Seed 42 (50 epochs, full CIFAR-10):**

| Row | Best val acc | Final val acc | Teacher forwards |
|-----|--------------|---------------|------------------|
| Teacher (ResNet-56) | 92.38% | 92.19% | 0 |
| Student KD (ResNet-20) | 91.66% | 91.59% | 19,550 |
| Student KD (budgeted teacher, ResNet-20) | 91.03% | 91.03% | 4,900 |
| CGSE multi-op (ResNet-20, teacher-free) | 90.85% | 90.62% | 0 |
| CGSE multi-op + budgeted KD (ResNet-20) | 88.49% | 78.18% | 4,900 |

**Note on Student CE rerun (not shown in table):** Student CE (seed 42) was rerun on 2026-04-20 to refresh artifacts; the rerun reproduced the same headline values (**best 91.18%, final 91.10%**, teacher_forwards **0**).

**Comparison to SEArch (Liang et al., 2025).** SEArch reports **93.58%** CIFAR-10 accuracy for a ~0.27M student under their full search + long retrain protocol (see `SEArch-baseline-and-CGSE-evaluation-plan.md`). Under our current Tier 2 settings (50 epochs; no SEArch operator space; no long final retrain), we **do not** exceed that accuracy yet. The strongest current “legitimate win” we can claim from Tier 2 is **teacher-free training** (0 teacher forwards) with competitive accuracy relative to the KD student baseline.

---

## Reproducibility — hardware and wall-clock (fill locally)

Record **once** for the camera-ready version:

| Item | Value (fill in) |
|------|------------------|
| CPU / GPU | e.g. Apple Silicon M*, NVIDIA … |
| PyTorch device | `cuda` / `mps` / `cpu` (from console or `train.py` log) |
| Approx. wall-clock / Tier 1 arm | … |
| Approx. wall-clock / Tier 1b run | ~1 h per 50-epoch job on our dev machine (adjust) |

**Note.** Sweep scripts used `DEVICE=auto` where applicable; PyTorch selects **CUDA**, else **MPS**, else **CPU**.

---

## Figure callouts (suggested)

- **Figure (concept):** Teacher vs critic arms — `paper_documentation/figures/cgse_teacher_vs_critic.png`.
- **Figure (pipeline):** Staged evolution — `cgse_evolution_stages.png`.
- **Figure (quantitative):** Tier 1 mean best val accuracy by arm — export from `web/index.html` bar chart or replot from `runs/tier1/metrics/*.csv`.
