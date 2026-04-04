# Checkpoints (`.pt`)

Weights are grouped by **experiment family** (same logic as `utils/artifact_families.py`):

| Directory | Typical contents |
|-----------|------------------|
| **`tier1/`** | Phase-2 full baseline, mutate, CGSE single-widen, phase-3 KD, teacher+mutate; multi-seed `*_seed<N>.pt`. |
| **`tier1b/`** | Multi-stage evolution runs (`cgse_evolution_tier1b_*`) and `*_discrete_critic.pt`. |
| **`smoke/`** | Subset / short-run smokes. |
| **`other/`** | Synthetic / dev (e.g. `phase0.pt`). |

**Teacher for KD:** `configs/cifar/phase3_*.yaml` point at **`checkpoints/tier1/cgse_phase2_cifar_full.pt`**. A symlink **`checkpoints/cgse_phase2_cifar_full.pt`** → `tier1/…` is kept for older notes and scripts.

`train.py` writes students (and critics) under the correct subfolder automatically from `experiment.name`.
