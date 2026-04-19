# Tier 2 — SEArch-style parity track (approx.)

Goal: run a **ResNet teacher/student KD setting** closer to SEArch’s CIFAR KD table:

- **Teacher:** `resnet_cifar` depth **56**
- **Student:** `resnet_cifar` depth **20** (~0.27M params)
- **Optimizer:** **SGD** + LR schedule (longer training than Tier 1)

This repo does **not** reproduce SEArch’s exact DAG + sep-conv search space; Tier 2 is
for **numeric/training-recipe parity** (teacher strength, student size, long SGD).

## Suggested order

1. Train a teacher:

```bash
python train.py --config configs/tier2/teacher_resnet56_cifar10.yaml --device auto --seed 41
```

2. Train students (CE baseline and KD):

```bash
python train.py --config configs/tier2/student_resnet20_cifar10_ce.yaml --device auto --seed 41
python train.py --config configs/tier2/student_resnet20_cifar10_kd.yaml --device auto --seed 41
```

For paper claims, run seeds 41–43 and report mean ± std.

