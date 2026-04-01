# Training run logs

Each file is a **verbatim** capture of stdout/stderr from a training or validation command, named for traceability:

`YYYYMMDD-HHmm_<short-label>.log`

**CSV** (`*_metrics.csv`) — per-epoch training curves (see `training.log_csv` in YAML).

**JSONL** (`*_mutations.jsonl`) — one JSON object per **structural mutation** when `mutation.log_jsonl` is set (see `utils/mutation_log.py`).

The main narrative and metrics table live in [`../CGSE-implementation-log.md`](../CGSE-implementation-log.md).
