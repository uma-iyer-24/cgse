# Training & experiment outputs

This folder lives at the **repository root** (`cgse/runs/`) so metrics and logs are separate from **paper PDFs and narrative docs** in `paper_documentation/`.

**Important:** `paper_documentation/runs` is a **regular file** (not a folder) that explains this layout. It exists so tools cannot recreate a misleading `paper_documentation/runs/` directory. Always write outputs here under `runs/`.

---

### Folder layout (by experiment family)

Artifacts are grouped so **Tier 1**, **Tier 1b**, **smokes**, and **other** runs do not share one flat directory.

| Family | Path pattern | What goes here |
|--------|----------------|----------------|
| **Tier 1** | `runs/tier1/{metrics,logs,mutations}/` | Paper grid: fixed / mutate / CGSE / teacher / teacher+mutate; `scripts/run_tier1.sh` logs (`tier1_*.log`). |
| **Tier 1b** | `runs/tier1b/{metrics,logs,mutations}/` | Multi-stage evolution configs; `run_tier1b.sh` logs (`tier1b_*.log`). |
| **Smoke** | `runs/smoke/{metrics,logs,mutations}/` | Any config whose `experiment.name` contains **`smoke`** (subset / quick checks). |
| **Other** | `runs/other/{metrics,logs,mutations}/` | Default dev run (`phase2_cifar.yaml`), synthetic trails, anything else. |

**Automatic placement (`train.py`):** after resolving seed suffixes and legacy `paper_documentation/runs/` redirects, **`training.log_csv`** and **`mutation.log_jsonl`** are passed through **`canonicalize_runs_artifact`** (`utils/artifact_families.py`). Any path that starts with `runs/` but is **not** already under `runs/tier1|tier1b|smoke|other/` is rewritten to `runs/<family>/metrics|mutations/<basename>`, where `<family>` is inferred from **`experiment.name`** (including `_seed<N>`). So you can set YAML to a short legacy shape like `runs/phase3_cifar_kd_metrics.csv` or `runs/metrics/foo.csv` and it still lands in the correct bucket.

**Checkpoints** are always written under **`checkpoints/<family>/`** from the same inference (see `student_checkpoint_path` in `train.py`).

**Console logs** (`tee`, `nohup`) are up to you; use **`runs/<family>/logs/`** (e.g. `runs/tier1/logs/my_run.log`) so they stay with the same experiment family as the sweep scripts.

**CSV** (`*_metrics.csv`) — per-epoch curves. **JSONL** — one line per structural mutation (`utils/mutation_log.py`). **Logs** — console captures from `tee` or scripts.

Run registry and narrative: [`paper_documentation/CGSE-implementation-log.md`](../paper_documentation/CGSE-implementation-log.md).

---

## Tier 1 comparison grid (CIFAR-10, `CifarGraphNet`)

Use **matched** `training` blocks (epochs, lr, batch, weight decay) across rows; only the arm changes. Details: [`paper_documentation/SEArch-baseline-and-CGSE-evaluation-plan.md`](../paper_documentation/SEArch-baseline-and-CGSE-evaluation-plan.md).

| Row | Config | Notes |
|-----|--------|--------|
| Fixed student | `configs/cifar/phase2_cifar_full.yaml` | `mutation.enabled: false` |
| Scheduled widen | `configs/cifar/phase2_cifar_full_mutate.yaml` | Widen after epoch 10, `widen_delta: 32` |
| Teacher + KD | `configs/cifar/phase3_cifar_kd.yaml` | Needs `checkpoints/tier1/cgse_phase2_cifar_full.pt` |
| Teacher + KD + widen | `configs/cifar/baseline_sear_ch_teacher_mutate.yaml` | Same teacher ckpt; widen epoch 10 |
| CGSE (critic) | `configs/cifar/phase2_cifar_full_cgse.yaml` | No teacher; critic window **epoch 10 only** (same decision point as scheduled mutate) |

**Multi-seed** (recommended ≥3 for paper claims): pass **`--seed`** so RNG and outputs do not overwrite. That appends **`_seed<N>`** to `experiment.name` and to metric/mutation stems (e.g. `runs/tier1/metrics/phase2_cifar_full_cgse_metrics_seed43.csv`).

Example sweep from repo root (adjust `--device`):

```bash
for s in 41 42 43; do
  python train.py --config configs/cifar/phase2_cifar_full_cgse.yaml --device auto --seed "$s" \
    2>&1 | tee "runs/tier1/logs/train_cgse_tier1_seed${s}.log"
done
```

Repeat the same loop with `--config configs/cifar/phase2_cifar_full_mutate.yaml` (and other Tier 1 configs) for comparable CSVs.

**Automated sweep** (sequential; long): from repo root, `DEVICE=auto bash scripts/run_tier1.sh`  
Optional: `SEEDS="42"` for a single seed, `RUN_FIXED=0` to skip the fixed-arch repeats; per-run logs → **`runs/tier1/logs/tier1_*.log`**.

**If the sweep died** (sleep, crash, closed laptop): inspect **`runs/tier1/logs/tier1_master.log`** and the latest **`runs/tier1/logs/tier1_*_seed*.log`** for a Python **Traceback**. Then restart with **`RESUME=1`** so finished jobs are skipped (uses **`checkpoints/tier1/<experiment>_seed<N>.pt`** — partial runs have no file and will train again):

```bash
nohup env DEVICE=auto RESUME=1 SEEDS="41 42 43" RUN_FIXED=1 bash scripts/run_tier1.sh >> runs/tier1/logs/tier1_master.log 2>&1 &
```

Use **`>>`** to append to the master log; use **`>`** for a fresh master log.

---

### Tier 1 vs Tier 1b development in parallel

- **Yes:** you can keep a long Tier 1 job running (`run_tier1.sh`, `nohup`, etc.) **while coding the next wave** on another **git branch**. The Python process that already started **does not reload** `train.py` when you save files; only **new** invocations of `train.py` pick up changes.
- **Avoid** checking out a branch that removes files your running job needs, or **stopping** the machine / putting it to sleep if you want the sweep to finish.
- **Tier 1b** (multi-stage evolution, multiple operators) — spec **[§7](../paper_documentation/SEArch-baseline-and-CGSE-evaluation-plan.md)**.

| Row | Config | Notes |
|-----|--------|--------|
| Scheduled multi-op | `configs/evolution/evolution_tier1b_schedule.yaml` | Full CIFAR-10, 5×10 epochs; `widen_conv3` → `widen_fc1` → `split_before_fc2` |
| Critic multi-op | `configs/evolution/evolution_tier1b_critic.yaml` | Same stage budget; same candidate set; ε-greedy + REINFORCE |

Smoke / dev: `configs/evolution/smoke/evolution_tier1b_smoke.yaml`, `configs/evolution/smoke/evolution_tier1b_critic_smoke.yaml`.

**Sweep** (6 long jobs for seeds 41–43): `DEVICE=auto bash scripts/run_tier1b.sh` → logs **`runs/tier1b/logs/tier1b_{schedule,critic}_seed<N>.log`**, metrics **`runs/tier1b/metrics/evolution_tier1b_*_metrics_seed<N>.csv`**, checkpoints **`checkpoints/tier1b/cgse_evolution_tier1b_{schedule,critic}_seed<N>.pt`**. **`RESUME=1`** skips seeds whose `.pt` already exists.
