"""
Run artifacts (metrics CSV, mutation JSONL, logs) belong under repo-root ``runs/``.
Redirect legacy ``paper_documentation/runs/`` paths so old configs and notebooks
cannot recreate that folder.
"""

from __future__ import annotations

import warnings

_LEGACY = "paper_documentation/runs/"


def normalize_run_artifact_path(path: str | None) -> str | None:
    if path is None:
        return None
    if not isinstance(path, str):
        return path
    p = path.replace("\\", "/")
    if _LEGACY not in p and not p.endswith("paper_documentation/runs"):
        return path

    warnings.warn(
        f"Config path still points at legacy '{_LEGACY.rstrip('/')}'. "
        f"Redirecting to repo-root runs/. Update YAML to use 'runs/...'. Was: {path!r}",
        UserWarning,
        stacklevel=2,
    )

    if p.endswith("paper_documentation/runs"):
        return "runs/"
    tail = p.split(_LEGACY, 1)[-1].lstrip("/")
    return f"runs/{tail}" if tail else "runs/"
