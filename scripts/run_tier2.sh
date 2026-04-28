#!/usr/bin/env bash
# Tier 2: ResNet CIFAR parity track (teacher ResNet-56, student ResNet-20).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-42}"

run_one() {
  local cfg="$1"
  local tag="$2"
  local s="$3"
  local exp_base="$4"
  local log="runs_paper/tier2/logs/tier2_${tag}_seed${s}.log"
  local ckpt="checkpoints/tier2/${exp_base}_seed${s}.pt"
  if [[ "${RESUME:-0}" == "1" ]] && [[ -f "$ckpt" ]]; then
    echo "=== [resume skip] ${tag} seed=${s} (found $(basename "$ckpt")) ==="
    return 0
  fi
  echo "=== tier2 ${tag} seed=${s} -> ${log} ==="
  PYTHONUNBUFFERED=1 python train.py --config "$cfg" --device "$DEVICE" --seed "$s" 2>&1 | tee "$log"
}

mkdir -p runs_paper/tier2/logs checkpoints/tier2

for s in $SEEDS; do
  run_one configs/tier2/teacher_resnet56_cifar10.yaml teacher "$s" tier2_teacher_resnet56_cifar10
  run_one configs/tier2/student_resnet20_cifar10_ce.yaml student_ce "$s" tier2_student_resnet20_cifar10_ce
  run_one configs/tier2/student_resnet20_cifar10_kd.yaml student_kd "$s" tier2_student_resnet20_cifar10_kd
  run_one configs/tier2/student_resnet20_cifar10_kd_budgeted.yaml student_kd_budgeted "$s" tier2_student_resnet20_cifar10_kd_budgeted
  run_one configs/tier2/student_resnet20_cifar10_sched_headwiden.yaml student_sched_headwiden "$s" tier2_student_resnet20_cifar10_sched_headwiden
  run_one configs/tier2/student_resnet20_cifar10_cgse_headwiden.yaml student_cgse_headwiden "$s" tier2_student_resnet20_cifar10_cgse_headwiden
  run_one configs/tier2/student_resnet20_cifar10_sched_layer3widen.yaml student_sched_layer3widen "$s" tier2_student_resnet20_cifar10_sched_layer3widen
  run_one configs/tier2/student_resnet20_cifar10_cgse_multiop.yaml student_cgse_multiop "$s" tier2_student_resnet20_cifar10_cgse_multiop
  run_one configs/tier2/student_resnet20_cifar10_cgse_multiop_kd_budgeted.yaml student_cgse_multiop_kd_budgeted "$s" tier2_student_resnet20_cifar10_cgse_multiop_kd_budgeted
  # Paper-faithful SEArch (channel-attention KD + MV scoring + sep-conv edge-splitting).
  run_one configs/tier2/student_resnet20_cifar10_searh.yaml student_searh "$s" tier2_student_resnet20_cifar10_searh
  # CGSE built on top of the same SEArch outer loop (teacher swapped for critic, high-frequency mutation).
  run_one configs/tier2/student_resnet20_cifar10_cgse_searh.yaml student_cgse_searh "$s" tier2_student_resnet20_cifar10_cgse_searh
done

echo "Tier 2 sweep done. See runs/tier2/."

