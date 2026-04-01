import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def append_mutation_jsonl(path: str | Path, event: Mapping[str, Any]) -> None:
    """Append one JSON object per line (JSONL), for paper / plotting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault(
        "logged_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
