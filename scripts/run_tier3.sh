#!/usr/bin/env bash
# Tier 3 — paper-faithful SEArch reproduction + CGSE-on-SEArch ablation grid.
#
# Models (8 total):
#   1. teacher              — ResNet-56, 100 ep, one-time, shared by all student seeds
#   2. student_ce           — ResNet-20, 50 ep CE (floor)
#   3. student_kd           — ResNet-20, 50 ep logit KD (KD-only baseline)
#   4. student_searh        — ResNet-20, paper-faithful SEArch (the system to beat)
#   5. student_cgse_base    — CGSE: probe OFF, baseline OFF
#   6. student_cgse_baseline— CGSE: probe OFF, baseline ON  (variance reduction only)
#   7. student_cgse_probe   — CGSE: probe ON,  baseline OFF (locality only)
#   8. student_cgse_full    — CGSE: probe ON,  baseline ON  (HEADLINE)
#
# Cadence: SEArch and ALL four CGSE arms run epochs_per_stage = 8 (Choice A,
# matched cadence). Per-seed totals: 100 (teacher, one-off) / 50 / 50 / 234
# / 234 / 234 / 234 / 234.
#
# Toggles:
#   SEEDS="42 43 44"          # default = "42"; headline sweep uses "42 43 44"
#   DEVICE=auto               # default
#   RESUME=1                  # skip arms whose checkpoint already exists
#   TEACHER_ONLY=1            # train only the (one) teacher and exit
#   STUDENTS_ONLY=1           # skip teacher (use existing checkpoint)
#   ARMS="searh cgse_full"    # whitelist (space-separated). Default = all
#                             # student arms. Teacher is gated by TEACHER_ONLY
#                             # / STUDENTS_ONLY, not by ARMS.
#
# Examples:
#   bash scripts/run_tier3.sh TEACHER_ONLY=1
#   SEEDS="42 43 44" bash scripts/run_tier3.sh STUDENTS_ONLY=1
#   ARMS="searh cgse_full" bash scripts/run_tier3.sh STUDENTS_ONLY=1 SEEDS=42

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-42}"
TEACHER_ONLY="${TEACHER_ONLY:-0}"
STUDENTS_ONLY="${STUDENTS_ONLY:-0}"
DEFAULT_ARMS="ce kd searh cgse_base cgse_baseline cgse_probe cgse_full"
ARMS="${ARMS:-$DEFAULT_ARMS}"

mkdir -p runs_paper/tier3/logs runs_paper/tier3/metrics runs_paper/tier3/mutations checkpoints/tier3

run_one() {
  local cfg="$1"
  local tag="$2"
  local s="$3"
  local exp_base="$4"
  local log="runs_paper/tier3/logs/tier3_${tag}_seed${s}.log"
  local ckpt="checkpoints/tier3/${exp_base}_seed${s}.pt"
  if [[ "${RESUME:-0}" == "1" ]] && [[ -f "$ckpt" ]]; then
    echo "=== [resume skip] ${tag} seed=${s} (found $(basename "$ckpt")) ==="
    return 0
  fi
  echo "=== tier3 ${tag} seed=${s} -> ${log} ==="
  PYTHONUNBUFFERED=1 python train.py --config "$cfg" --device "$DEVICE" --seed "$s" 2>&1 | tee "$log"
}

arm_enabled() {
  local arm="$1"
  for a in $ARMS; do
    if [[ "$a" == "$arm" ]]; then return 0; fi
  done
  return 1
}

# ---------- Teacher (one-time, seed 42 only) ----------
if [[ "$STUDENTS_ONLY" != "1" ]]; then
  run_one configs/tier3/teacher_resnet56_cifar10.yaml teacher 42 tier3_teacher_resnet56_cifar10
fi

if [[ "$TEACHER_ONLY" == "1" ]]; then
  echo "Tier 3 teacher-only run done."
  exit 0
fi

# ---------- Student arms (one pass per seed) ----------
for s in $SEEDS; do
  arm_enabled "ce"            && run_one configs/tier3/student_resnet20_cifar10_ce.yaml            student_ce            "$s" tier3_student_resnet20_cifar10_ce
  arm_enabled "kd"            && run_one configs/tier3/student_resnet20_cifar10_kd.yaml            student_kd            "$s" tier3_student_resnet20_cifar10_kd
  arm_enabled "searh"         && run_one configs/tier3/student_resnet20_cifar10_searh.yaml         student_searh         "$s" tier3_student_resnet20_cifar10_searh
  arm_enabled "cgse_base"     && run_one configs/tier3/student_resnet20_cifar10_cgse_base.yaml     student_cgse_base     "$s" tier3_student_resnet20_cifar10_cgse_base
  arm_enabled "cgse_baseline" && run_one configs/tier3/student_resnet20_cifar10_cgse_baseline.yaml student_cgse_baseline "$s" tier3_student_resnet20_cifar10_cgse_baseline
  arm_enabled "cgse_probe"    && run_one configs/tier3/student_resnet20_cifar10_cgse_probe.yaml    student_cgse_probe    "$s" tier3_student_resnet20_cifar10_cgse_probe
  arm_enabled "cgse_full"     && run_one configs/tier3/student_resnet20_cifar10_cgse_full.yaml     student_cgse_full     "$s" tier3_student_resnet20_cifar10_cgse_full
done

echo "Tier 3 sweep done. Artifacts under runs_paper/tier3/ and checkpoints/tier3/."
