# Training & experiment outputs

This folder lives at the **repository root** (`cgse/runs/`) so metrics and logs are separate from **paper PDFs and narrative docs** in `paper_documentation/`.

Each `.log` file can be a **verbatim** capture of stdout/stderr from a training command, named for traceability:

`YYYYMMDD-HHmm_<short-label>.log`

**CSV** (`*_metrics.csv`) — per-epoch training curves (path set by `training.log_csv` in each YAML config).

**JSONL** (`*_mutations.jsonl`) — one JSON object per **structural mutation** when `mutation.log_jsonl` is set (see `utils/mutation_log.py`).

Run registry and narrative: [`paper_documentation/CGSE-implementation-log.md`](../paper_documentation/CGSE-implementation-log.md).
