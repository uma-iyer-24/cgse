# Experiment configs

Only **subdirectories** here — no YAML at the root (keeps the tree easy to scan).

```
configs/
├── cifar/           # CIFAR-10 (Tier 1 grid + dev default)
│   ├── smoke/       # subset / quick runs
│   └── *.yaml       # full training recipes
├── evolution/       # Tier 1b multi-stage
│   ├── smoke/
│   └── *.yaml       # schedule + critic (full CIFAR)
└── synthetic/
    └── base.yaml    # random-data MLP (`StudentNet`)
```

| Location | Contents |
|----------|----------|
| **`cifar/`** | `phase2_cifar.yaml`, `phase2_cifar_full*.yaml`, `phase3_cifar_kd.yaml`, `baseline_sear_ch_teacher_mutate.yaml`, … |
| **`cifar/smoke/`** | `phase2_smoke.yaml`, `phase2_smoke_mutate.yaml`, `phase2_cifar_cgse_smoke.yaml`, `phase3_cifar_kd_smoke.yaml` |
| **`evolution/`** | `evolution_tier1b_schedule.yaml`, `evolution_tier1b_critic.yaml` |
| **`evolution/smoke/`** | `evolution_tier1b_smoke.yaml`, `evolution_tier1b_critic_smoke.yaml` |
| **`synthetic/`** | `base.yaml` |

**CLI:** `python train.py --config configs/cifar/phase2_cifar.yaml` (same path as `train.py` default).

**Artifacts:** `train.py` sends metrics/JSONL/checkpoints to `runs/<family>/…` and `checkpoints/<family>/…` from `experiment.name` (see `utils/artifact_families.py`).
