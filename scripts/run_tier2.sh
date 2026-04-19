#!/usr/bin/env bash
# Tier 2: ResNet CIFAR parity track (teacher ResNet-56, student ResNet-20).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-41 42 43}"

run_one() {
  local cfg="$1"
  local tag="$2"
  local s="$3"
  local log="runs/tier2/logs/tier2_${tag}_seed${s}.log"
  echo "=== tier2 ${tag} seed=${s} -> ${log} ==="
  python train.py --config "$cfg" --device "$DEVICE" --seed "$s" 2>&1 | tee "$log"
}

mkdir -p runs/tier2/logs checkpoints/tier2

for s in $SEEDS; do
  run_one configs/tier2/teacher_resnet56_cifar10.yaml teacher "$s"
  run_one configs/tier2/student_resnet20_cifar10_ce.yaml student_ce "$s"
  run_one configs/tier2/student_resnet20_cifar10_kd.yaml student_kd "$s"
done

echo "Tier 2 sweep done. See runs/tier2/."

