# Paper & project documentation

This folder holds materials for the CGSE paper and reproducibility.

| Document | Role |
|----------|------|
| [`phase-plan-overview.pdf`](phase-plan-overview.pdf) | Phased roadmap (Phases 0–8). |
| [`project-doc.pdf`](project-doc.pdf) | Full research narrative, prior art, SEArch/CGSE positioning, experimental design. |
| [`CGSE-implementation-log.md`](CGSE-implementation-log.md) | **Living implementation log**: code changes, decisions, configs, and training run records (for paper Methods / Appendix). Includes a **[plain-language Phase 2](CGSE-implementation-log.md#plain-language-phase-2)** summary (non-technical “what we built and why”). |
| [`CGSE-codebase-guide.md`](CGSE-codebase-guide.md) | **Codebase map (file-by-file)**: architecture, execution paths, module roles, configs, and maintenance notes—read alongside the source for a full in-depth understanding. **Update this when you add or rename code.** |
| [`CGSE-detailed-phase-walkthrough.md`](CGSE-detailed-phase-walkthrough.md) | **Long-form English narrative**: phase-by-phase goals, step-by-step what was built, **why** key choices were made, and **which files** implement each step (companion to the log + codebase guide). |

Training **metrics, JSONL, and console logs** live only in **[`runs/`](../runs/)** at the **repo root**. The path **`paper_documentation/runs`** is intentionally a **small text file** (not a directory) so nothing can create a nested `runs` folder here by mistake. **Do not delete it** to make a directory with that name.

### Phase 2 training (quick reference)

```bash
pip install -r requirements.txt
python train.py --config configs/phase2_cifar.yaml
# Optional: force device
python train.py --config configs/phase2_cifar.yaml --device cpu
# Fast smoke tests (small subset, CPU)
python train.py --config configs/phase2_smoke.yaml
python train.py --config configs/phase2_smoke_mutate.yaml
# Full CIFAR-10 (50k/10k), longer training — paper baseline
python train.py --config configs/phase2_cifar_full.yaml
# Same as baseline + one edge_widen after epoch 10 (compare metrics CSVs)
python train.py --config configs/phase2_cifar_full_mutate.yaml
```
- **Synthetic / Phase-0 MLP** (no CIFAR): `python train.py --config configs/base.yaml`
- Per-epoch metrics append to the CSV path in each YAML (`training.log_csv`); includes **`num_parameters`** each epoch.
- When **`mutation.log_jsonl`** is set and a widen runs, one **JSON line** is appended per mutation (layer id, param count before/after, etc.).
