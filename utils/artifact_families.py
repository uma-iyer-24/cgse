"""Group experiment artifacts under runs/ and checkpoints/ by family (tier1, tier1b, …)."""

from __future__ import annotations

from pathlib import Path

# Subdirectory names under runs/ and checkpoints/
TIER1 = "tier1"
TIER1B = "tier1b"
SMOKE = "smoke"
OTHER = "other"

RUNS_FAMILIES = frozenset({TIER1, TIER1B, SMOKE, OTHER})


def infer_artifact_family(experiment_name: str) -> str:
    """
    Map experiment_name (may include _seed<N>) to a stable folder bucket.
    """
    n = experiment_name.lower()
    if "evolution_tier1b" in n or "cgse_evolution_tier1b" in n:
        return TIER1B
    if "smoke" in n:
        return SMOKE
    if any(
        x in n
        for x in (
            "phase2_cifar_full",
            "phase3_cifar_kd",
            "baseline_sear_ch",
        )
    ):
        return TIER1
    if "phase2_smoke" in n or n.startswith("cgse_phase2_smoke"):
        return SMOKE
    return OTHER


def canonicalize_runs_artifact(
    path: str | None,
    experiment_name: str,
    kind: str,
) -> str | None:
    """
    Force CSV/JSONL under ``runs/<family>/{metrics|mutations}/`` from ``experiment_name``.

    Accepts legacy shapes: ``runs/foo.csv``, ``runs/metrics/foo.csv``, ``runs/logs/foo.csv``.
    Leaves absolute paths and non-``runs/`` paths unchanged.
    """
    if path is None:
        return None
    if kind not in ("metrics", "mutations"):
        raise ValueError(f"kind must be metrics or mutations, got {kind!r}")

    raw = str(path).replace("\\", "/")
    p = Path(raw)
    parts = p.parts
    if not parts:
        return path
    if p.is_absolute():
        return path
    if parts[0] != "runs":
        return path
    if len(parts) >= 2 and parts[1] in RUNS_FAMILIES:
        return raw

    fam = infer_artifact_family(experiment_name)
    name = p.name
    return str(Path("runs") / fam / kind / name)


def runs_metrics_dir(experiment_name: str) -> Path:
    return Path("runs") / infer_artifact_family(experiment_name) / "metrics"


def runs_mutations_dir(experiment_name: str) -> Path:
    return Path("runs") / infer_artifact_family(experiment_name) / "mutations"


def checkpoints_dir(experiment_name: str) -> Path:
    return Path("checkpoints") / infer_artifact_family(experiment_name)


def student_checkpoint_path(experiment_name: str) -> Path:
    return checkpoints_dir(experiment_name) / f"{experiment_name}.pt"


def structural_critic_checkpoint_path(experiment_name: str) -> Path:
    return checkpoints_dir(experiment_name) / f"{experiment_name}_critic.pt"


def discrete_critic_checkpoint_path(experiment_name: str) -> Path:
    return checkpoints_dir(experiment_name) / f"{experiment_name}_discrete_critic.pt"


def resolve_teacher_checkpoint(configured_path: str) -> Path:
    """
    Load teacher weights: use YAML path if present; else try tier1/ then flat checkpoints/.
    """
    p = Path(configured_path)
    if p.is_file():
        return p
    name = p.name
    tier1 = Path("checkpoints") / TIER1 / name
    if tier1.is_file():
        return tier1
    flat = Path("checkpoints") / name
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"teacher checkpoint not found: {configured_path!r} "
        f"(also tried {tier1} and {flat})"
    )
