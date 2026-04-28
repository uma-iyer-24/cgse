# Paper & project documentation

This folder holds materials for the CGSE paper and reproducibility.

### Figures and diagrams (visual overview)

| Asset | Purpose |
|-------|---------|
| [`figures/cgse_teacher_vs_critic.png`](figures/cgse_teacher_vs_critic.png) | High-level **teacher / KD vs CGSE / critic** contrast (also embedded in the [root README](../README.md)). |
| [`figures/cgse_evolution_stages.png`](figures/cgse_evolution_stages.png) | **Multi-stage** train → choose edit → apply op → repeat. |

The [repository `README.md`](../README.md) includes **Mermaid** diagrams (train/validate loop, arm comparison, Tier 1b stage flow) that render on GitHub; use the PNGs for slides or PDFs where Mermaid is unavailable.

| Document | Role |
|----------|------|
| `SEArch-MP1-base-paper.pdf` | **Liang et al. (2025)** — *SEArch* (Neurocomputing): teacher-guided self-evolving architecture; **primary baseline system** CGSE competes with. |
| [`SEArch-baseline-and-CGSE-evaluation-plan.md`](SEArch-baseline-and-CGSE-evaluation-plan.md) | Citation, SEArch↔codebase gap analysis, **Tier 1 / 2 / 3**, **§7** multi-stage + multi-op SEArch/CGSE roadmap, pytest/paper checklist. |
| [`CGSE-experiments-and-results-guide.md`](CGSE-experiments-and-results-guide.md) | **Paper-oriented** guide: Methods/Experiments framing, **all tiers**, SEArch comparability (Strategies A/B), **test hierarchy**, execution order, metrics, tables/figures, reproducibility checklist. Includes **§13–§15** (SEArch gap paragraph, Tier 1 number snapshot, deferred items). |
| Results preview (static UI) | Repo root [`web/`](../web/): run `python scripts/build_results_site.py`, open `web/index.html` — charts + concept figures from `runs/*/metrics/*.csv`. Optional **GitHub Pages**: see `web/README.md`. |
| [`draft-results-for-paper.md`](draft-results-for-paper.md) | **Copy-paste Results prose** (Tier 1 + Tier 1b) + reproducibility table stubs; update when seeds finish. |
| Tier 2 configs | `configs/tier2/` — ResNet CIFAR teacher/student parity track (ResNet-56 teacher; ResNet-20 student; SGD + LR schedule). |
| [`phase-plan-overview.pdf`](phase-plan-overview.pdf) | Phased roadmap (Phases 0–8). |
| [`project-doc.pdf`](project-doc.pdf) | CGSE narrative, critic vs teacher, prior art (read alongside the SEArch paper). |
| [`CGSE-implementation-log.md`](CGSE-implementation-log.md) | **Living implementation log**: code changes, decisions, configs, and training run records (for paper Methods / Appendix). Includes a **[plain-language Phase 2](CGSE-implementation-log.md#plain-language-phase-2)** summary (non-technical “what we built and why”). |
| [`CGSE-codebase-guide.md`](CGSE-codebase-guide.md) | **Codebase map (file-by-file)**: architecture, execution paths, module roles, configs, and maintenance notes—read alongside the source for a full in-depth understanding. **Update this when you add or rename code.** |
| [`CGSE-detailed-phase-walkthrough.md`](CGSE-detailed-phase-walkthrough.md) | **Long-form English narrative**: phase-by-phase goals, step-by-step what was built, **why** key choices were made, and **which files** implement each step (companion to the log + codebase guide). |
| [`CGSE-math-and-equations.md`](CGSE-math-and-equations.md) | **Publication-ready mathematical formulation:** every numbered equation the paper needs, with a single consistent notation. Channel-attention KD (Eqs. 1–5), λ-anneal (Eq. 6), arm losses (Eqs. 7–8), modification value (Eqs. 9–10), CGSE student probe (Eqs. 11–17), critic policy (Eqs. 18–21), REINFORCE-with-baseline (Eqs. 22–25), edge-splitting operators (Eqs. 26–28), termination (Eqs. 29–30), Algorithm 1 pseudocode. Cross-referenced from the implementation log and codebase guide. |

Training **metrics, JSONL, and console logs** live only in **[`runs/`](../runs/)** at the **repo root**. The path **`paper_documentation/runs`** is intentionally a **small text file** (not a directory) so nothing can create a nested `runs` folder here by mistake. **Do not delete it** to make a directory with that name.

### Phase 2 training (quick reference)

```bash
pip install -r requirements.txt
python train.py --config configs/cifar/phase2_cifar.yaml
# Optional: force device
python train.py --config configs/cifar/phase2_cifar.yaml --device cpu
# Fast smoke tests (small subset, CPU)
python train.py --config configs/cifar/smoke/phase2_smoke.yaml
python train.py --config configs/cifar/smoke/phase2_smoke_mutate.yaml
# Full CIFAR-10 (50k/10k), longer training — paper baseline
python train.py --config configs/cifar/phase2_cifar_full.yaml
# Same as baseline + one edge_widen after epoch 10 (compare metrics CSVs)
python train.py --config configs/cifar/phase2_cifar_full_mutate.yaml
```
- **Synthetic / Phase-0 MLP** (no CIFAR): `python train.py --config configs/synthetic/base.yaml`
- Per-epoch metrics append to the CSV path in each YAML (`training.log_csv`); includes **`num_parameters`** each epoch.
- When **`mutation.log_jsonl`** is set and a widen runs, one **JSON line** is appended per mutation (layer id, param count before/after, etc.).
