#!/usr/bin/env bash
# Tier 2: run ONLY the new "budgeted teacher-forwards" arms.
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
  run_one configs/tier2/student_resnet20_cifar10_kd_budgeted.yaml student_kd_budgeted "$s" tier2_student_resnet20_cifar10_kd_budgeted
  run_one configs/tier2/student_resnet20_cifar10_cgse_multiop_kd_budgeted.yaml student_cgse_multiop_kd_budgeted "$s" tier2_student_resnet20_cifar10_cgse_multiop_kd_budgeted
done

echo "Tier 2 budgeted-KD sweep done."

