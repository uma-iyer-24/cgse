#!/usr/bin/env bash
# Tier 1 grid: same CifarGraphNet recipe, five arms × seeds (see runs/README.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-41 42 43}"
TEACHER_CKPT="${TEACHER_CKPT:-checkpoints/tier1/cgse_phase2_cifar_full.pt}"
RUN_FIXED="${RUN_FIXED:-1}"
# Set RESUME=1 to skip a job if checkpoints/<experiment_basename>_seed<N>.pt already exists
# (train.py only writes the .pt at the end of a run — partial runs have no ckpt and will re-run).

run_one() {
  local cfg="$1"
  local tag="$2"
  local s="$3"
  local exp_base="${4:-}"
  local log="runs/tier1/logs/tier1_${tag}_seed${s}.log"
  local ckpt="checkpoints/tier1/${exp_base}_seed${s}.pt"
  if [[ "${RESUME:-0}" == "1" ]] && [[ -n "$exp_base" ]] && [[ -f "$ckpt" ]]; then
    echo "=== [resume skip] ${tag} seed=${s} (found $(basename "$ckpt")) ==="
    return 0
  fi
  echo "=== ${tag} seed=${s} -> ${log} ==="
  python train.py --config "$cfg" --device "$DEVICE" --seed "$s" 2>&1 | tee "$log"
}

for s in $SEEDS; do
  if [[ "$RUN_FIXED" == "1" ]]; then
    run_one configs/cifar/phase2_cifar_full.yaml fixed "$s" cgse_phase2_cifar_full
  fi
  run_one configs/cifar/phase2_cifar_full_mutate.yaml mutate "$s" cgse_phase2_cifar_full_mutate
  run_one configs/cifar/phase2_cifar_full_cgse.yaml cgse "$s" cgse_phase2_cifar_full_cgse

  if [[ -f "$TEACHER_CKPT" ]]; then
    run_one configs/cifar/phase3_cifar_kd.yaml kd "$s" cgse_phase3_cifar_kd
    run_one configs/cifar/baseline_sear_ch_teacher_mutate.yaml teacher_mutate "$s" baseline_sear_ch_teacher_mutate
  else
    echo "Skipping kd + teacher_mutate for seed $s (missing $TEACHER_CKPT)"
  fi
done

echo "Tier 1 sweep done. Collect last-epoch val_acc from runs/*_metrics_seed*.csv"
