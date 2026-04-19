#!/usr/bin/env bash
# Tier 2 DEV: fast sanity-check sweep (small subsets, few epochs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-42}"

run_one() {
  local cfg="$1"
  local tag="$2"
  local s="$3"
  local log="runs/tier2/logs/tier2dev_${tag}_seed${s}.log"
  echo "=== tier2dev ${tag} seed=${s} -> ${log} ==="
  python train.py --config "$cfg" --device "$DEVICE" --seed "$s" 2>&1 | tee "$log"
}

mkdir -p runs/tier2/logs checkpoints/tier2

for s in $SEEDS; do
  run_one configs/tier2/dev/teacher_resnet56_cifar10_dev.yaml teacher "$s"
  run_one configs/tier2/dev/student_resnet20_cifar10_ce_dev.yaml student_ce "$s"
  run_one configs/tier2/dev/student_resnet20_cifar10_kd_dev.yaml student_kd "$s"
  run_one configs/tier2/dev/student_resnet20_cifar10_cgse_multiop_dev.yaml cgse_multiop "$s"
done

echo "Tier 2 DEV sweep done. See runs/tier2/."

