# Tier 2 — SEArch-style parity track (approx.)

Goal: run a **ResNet teacher/student KD setting** closer to SEArch’s CIFAR KD table:

- **Teacher:** `resnet_cifar` depth **56**
- **Student:** `resnet_cifar` depth **20** (~0.27M params)
- **Optimizer:** **SGD** + LR schedule (longer training than Tier 1)

This repo does **not** reproduce SEArch’s exact DAG + sep-conv search space; Tier 2 is
for **numeric/training-recipe parity** (teacher strength, student size, long SGD).

## What to report (Tier 2 + “beat SEArch/NAS” metrics)

In addition to accuracy and params, Tier 2 is where we report **efficiency / dependency**
metrics that CGSE is designed to improve:

- **Teacher-free constraint**: results for arms with **`teacher.enabled: false`** (teacher compute = 0).
- **Teacher compute**: `teacher_forwards` (cumulative) — number of teacher forward passes.
- **Wall-clock**: `wall_seconds` (cumulative), `epoch_seconds` (per epoch).
- **Accuracy per compute**: compare `val_acc` vs `wall_seconds` curves.
- **Edits-to-threshold** (Tier 1b / evolution arms): # mutation events in JSONL until reaching a target `val_acc`.

All of these are logged in the per-epoch metrics CSV columns produced by `train.py`.

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

3. Optional additional Tier 2 rows (structural edit on ResNet head):

These mirror Tier 1’s “scheduled widen” and “CGSE” rows, but using a **ResNet-safe**
mutation: **Net2Net-style widening of the classifier head** (`mutation.op: resnet_head_widen`).

```bash
python train.py --config configs/tier2/student_resnet20_cifar10_sched_headwiden.yaml --device auto --seed 41
python train.py --config configs/tier2/student_resnet20_cifar10_cgse_headwiden.yaml --device auto --seed 41
```

4. Tier 2 multi-op CGSE (critic chooses which edit to apply):

This is a discrete action variant where the critic chooses one action at the decision epoch:

- `noop`
- `resnet_head_widen` (Net2Net head widen)
- `resnet_layer3_widen` (widen conv channels in `layer3`, function-preserving init)
- `resnet_insert_block` (deepen by inserting a residual block into `layer3`)

```bash
python train.py --config configs/tier2/student_resnet20_cifar10_cgse_multiop.yaml --device auto --seed 41
```

For paper claims, run seeds 41–43 and report mean ± std.

