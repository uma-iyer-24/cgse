#!/usr/bin/env bash
# Tier 1b: schedule vs critic evolution, multi-seed (see runs/README.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-41 42 43}"
# RESUME=1 skips a seed if checkpoints/tier1b/<experiment>_seed<N>.pt exists (finished run only).

run_one() {
  local cfg="$1"
  local tag="$2"
  local s="$3"
  local exp_base="$4"
  local log="runs/tier1b/logs/tier1b_${tag}_seed${s}.log"
  local ckpt="checkpoints/tier1b/${exp_base}_seed${s}.pt"
  if [[ "${RESUME:-0}" == "1" ]] && [[ -f "$ckpt" ]]; then
    echo "=== [resume skip] ${tag} seed=${s} (found $(basename "$ckpt")) ==="
    return 0
  fi
  echo "=== tier1b ${tag} seed=${s} -> ${log} ==="
  python train.py --config "$cfg" --device "$DEVICE" --seed "$s" 2>&1 | tee "$log"
}

for s in $SEEDS; do
  run_one configs/evolution/evolution_tier1b_schedule.yaml schedule "$s" cgse_evolution_tier1b_schedule
  run_one configs/evolution/evolution_tier1b_critic.yaml critic "$s" cgse_evolution_tier1b_critic
done

echo "Tier 1b sweep done. Check runs/tier1b_*_seed*.log and runs/*_metrics_seed*.csv"
