"""
SEArch modification value (MV) scorer — paper Eq. (5).

For every student node ``n`` we compute:

    MV(n) = D(n) · deg+(n) / deg-(n)

where ``D(n)`` is the channel-attention KD distance between student node n
and its matched teacher node n̂ (paper Eq. 2), ``deg+(n)`` is the number of
successor nodes in the student graph, ``deg-(n)`` is the number of
predecessor nodes.

For our linear CIFAR-ResNet stack every internal block has deg+ = deg- = 1,
so the adjustment factor collapses to 1 and MV(n) = D(n). We still keep
the structure of Eq. 5 in code so a future DAG generalisation drops in
without API changes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from training.searh_attention import MultiNodeAttentionKD


@torch.no_grad()
def compute_per_node_distances(
    student: nn.Module,
    teacher: nn.Module,
    attn_kd: MultiNodeAttentionKD,
    score_loader: Iterable,
    device: torch.device,
    *,
    num_batches: int = 2,
) -> Dict[str, float]:
    """
    Run a small score-batch loop and average ``D(n)`` per matched node.
    Hooks must already be attached on ``attn_kd``.
    """
    student.eval()
    teacher.eval()
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    seen = 0
    for x, _y in score_loader:
        if seen >= num_batches:
            break
        x = x.to(device, non_blocking=True)
        # Forward both — hooks populate kd's caches.
        student(x)
        teacher(x)
        d_per_node = attn_kd.per_node_distances()
        for k, v in d_per_node.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach().item())
            counts[k] = counts.get(k, 0) + 1
        seen += 1
    return {k: sums[k] / max(counts[k], 1) for k in sums}


def deg_plus_minus(node_id: str) -> Tuple[int, int]:
    """Out- and in-degree for a CIFAR-ResNet linear node — both are 1."""
    return 1, 1


def modification_values(
    distances: Dict[str, float],
) -> Dict[str, float]:
    """Eq. 5 applied per node id."""
    mvs: Dict[str, float] = {}
    for nid, d in distances.items():
        dp, dm = deg_plus_minus(nid)
        mvs[nid] = float(d) * (float(dp) / max(dm, 1))
    return mvs


def rank_candidates(
    candidates: List,                   # list of evolution.candidates.Candidate
    mvs: Dict[str, float],
) -> List[Tuple[object, float]]:
    """Rank candidates by MV, descending. Returns list of (Candidate, mv)."""
    ranked = []
    for c in candidates:
        mv = mvs.get(c.node_id, 0.0)
        ranked.append((c, mv))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked
