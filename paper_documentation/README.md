# Paper & project documentation

This folder holds materials for the CGSE paper and reproducibility.

| Document | Role |
|----------|------|
| [`phase-plan-overview.pdf`](phase-plan-overview.pdf) | Phased roadmap (Phases 0–8). |
| [`project-doc.pdf`](project-doc.pdf) | Full research narrative, prior art, SEArch/CGSE positioning, experimental design. |
| [`CGSE-implementation-log.md`](CGSE-implementation-log.md) | **Living implementation log**: code changes, decisions, configs, and training run records (for paper Methods / Appendix). Includes a **[plain-language Phase 2](CGSE-implementation-log.md#plain-language-phase-2)** summary (non-technical “what we built and why”). |
| [`CGSE-codebase-guide.md`](CGSE-codebase-guide.md) | **Codebase map (file-by-file)**: architecture, execution paths, module roles, configs, and maintenance notes—read alongside the source for a full in-depth understanding. **Update this when you add or rename code.** |

Raw console logs from training are stored under [`runs/`](runs/) with one file per run; the implementation log summarizes them and links to the files.

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
```

- **Synthetic / Phase-0 MLP** (no CIFAR): `python train.py --config configs/base.yaml`
- Per-epoch metrics append to the CSV path in each YAML (`training.log_csv`); includes **`num_parameters`** each epoch.
- When **`mutation.log_jsonl`** is set and a widen runs, one **JSON line** is appended per mutation (layer id, param count before/after, etc.).
