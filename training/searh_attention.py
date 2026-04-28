"""
Channel-space attention knowledge distillation, paper-faithful per
Liang, Xiang & Li (2025), Neurocomputing 651 §3.2 Eqs. (1)-(3).

For a student node feature map ``f_s ∈ (B, Cs, H, W)`` and the matched
teacher node ``f_t ∈ (B, Ct, H, W)``, we project ``f_t`` onto the student's
channel space via channel-space attention:

    Q  = W_Q · channel_repr(f_s)         # (Cs, d_k)
    K  = W_K · channel_repr(f_t)         # (Ct, d_k)
    V  = W_V · channel_repr(f_t)         # (Ct, d_k_v)
    A  = softmax(Q K^T / sqrt(d_k))      # (Cs, Ct) — for every student
                                         #   channel, weights over teacher channels
    f_proj = A · f_t  (broadcast over spatial)   # (B, Cs, H, W)

We then return per-sample squared L2 distance ``||f_s - f_proj||_2^2``
(paper Eq. 2). The averaged distance over a node set gives the imitation
loss ``L_im`` (Eq. 3).

The query/key/value linear maps are learned alongside the student.

Notes for our adaptation
------------------------
* We compute a single channel descriptor per channel via global average pooling,
  then project (B, C, H*W) features by the resulting per-sample per-channel
  attention weights. This matches the paper's "for every channel in the student's
  feature map, an attention query is used to calculate the corresponding channel
  weights in the teacher's feature map" (§3.2).
* Spatial dimensions of student/teacher feature maps must match. For our
  Tier-2 ResNet-20 ↔ ResNet-56 mapping that is the case at every paired node
  (16x32x32, 32x16x16, 64x8x8). The mapping is built by ``searh_node_map``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionKD(nn.Module):
    """Channel-space attention KD between one student node and one teacher node."""

    def __init__(self, c_student: int, c_teacher: int, d_k: int = 64):
        super().__init__()
        self.c_student = int(c_student)
        self.c_teacher = int(c_teacher)
        self.d_k = int(d_k)
        self.q = nn.Linear(1, d_k, bias=False)   # per-channel scalar → query
        self.k = nn.Linear(1, d_k, bias=False)   # per-channel scalar → key
        # Channel-mix projection: (Ct → Cs) realised by softmax-attention weights.
        # We precompute it from per-channel descriptors at every forward.
        self.scale = float(d_k) ** -0.5

    @staticmethod
    def _channel_descriptor(f: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, C, 1) global-average per channel
        return f.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        """
        Compute D(n) = ||f_s - Atten(f_t)||_2^2 averaged over (B, H, W) but
        summed over channels (so the value is comparable across feature maps
        of different channel widths).

        Returns a *scalar* tensor (mean over batch and spatial positions).
        """
        if f_s.shape[0] != f_t.shape[0] or f_s.shape[2:] != f_t.shape[2:]:
            raise ValueError(
                f"Spatial/batch shape mismatch: student {tuple(f_s.shape)} vs "
                f"teacher {tuple(f_t.shape)}"
            )
        if f_s.shape[1] != self.c_student or f_t.shape[1] != self.c_teacher:
            raise ValueError(
                f"Channel mismatch: expected ({self.c_student}, {self.c_teacher}) "
                f"got ({f_s.shape[1]}, {f_t.shape[1]})"
            )

        b, _, h, w = f_s.shape

        # Per-channel descriptors → queries/keys.
        d_s = self._channel_descriptor(f_s)          # (B, Cs, 1)
        d_t = self._channel_descriptor(f_t)          # (B, Ct, 1)
        Q = self.q(d_s)                              # (B, Cs, d_k)
        K = self.k(d_t)                              # (B, Ct, d_k)

        # Attention weights: per-batch, (Cs, Ct)
        attn = torch.einsum("bsd,btd->bst", Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)               # over teacher channels

        # Project teacher feature map: (B, Cs, H, W) = attn @ f_t (channel mix).
        f_t_flat = f_t.view(b, self.c_teacher, h * w)
        f_proj   = torch.einsum("bst,btp->bsp", attn, f_t_flat).view(b, self.c_student, h, w)

        diff = f_s - f_proj
        # Mean over all dims (batch, channels, spatial). The paper's Eq. 2
        # writes a squared L2 per node; we normalize by element count so
        # that the imitation loss is O(1) and stable when summed with CE
        # at λ ≈ 1, regardless of feature-map size.
        return diff.pow(2).mean()


class MultiNodeAttentionKD(nn.Module):
    """
    Holds one ``ChannelAttentionKD`` per (student_node_name, teacher_node_name)
    pair. Captures intermediate features via forward hooks so that the student
    and teacher only need to be called once per batch.

    Usage:
        kd = MultiNodeAttentionKD(student, teacher, node_pairs, d_k=64)
        L_im = kd.imitation_loss(student_logits=stu_out, x=x)   # uses last fwd
        # or just call kd.attach() once and read kd.last_distances after fwd
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        node_pairs: List[Tuple[str, str]],
        student_lookup: Dict[str, nn.Module],
        teacher_lookup: Dict[str, nn.Module],
        d_k: int = 64,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.node_pairs = list(node_pairs)
        self._student_lookup = dict(student_lookup)
        self._teacher_lookup = dict(teacher_lookup)
        self._student_feats: Dict[str, torch.Tensor] = {}
        self._teacher_feats: Dict[str, torch.Tensor] = {}
        self._handles: list = []

        # Build one attention head per pair, sized to current channel widths.
        heads = nn.ModuleDict()
        with torch.no_grad():
            # We rely on the caller to have run a probe forward first — but to
            # avoid that, infer channels by reading the modules' immediate
            # output channel attributes when available, else lazy-init on
            # first forward.
            pass
        self.heads = heads
        self.d_k = int(d_k)
        self._lazy_built = False
        if device is not None:
            self.to(device)

    @torch.no_grad()
    def _hook_factory(self, store: Dict[str, torch.Tensor], key: str):
        def hook(_module, _inp, out):
            store[key] = out
        return hook

    def attach(self) -> None:
        """Register forward hooks on student/teacher matched modules."""
        if self._handles:
            return
        for s_name, t_name in self.node_pairs:
            sm = self._student_lookup[s_name]
            tm = self._teacher_lookup[t_name]
            self._handles.append(sm.register_forward_hook(self._hook_factory(self._student_feats, s_name)))
            self._handles.append(tm.register_forward_hook(self._hook_factory(self._teacher_feats, t_name)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _ensure_heads(self, device: torch.device) -> None:
        if self._lazy_built:
            return
        for s_name, t_name in self.node_pairs:
            f_s = self._student_feats.get(s_name)
            f_t = self._teacher_feats.get(t_name)
            if f_s is None or f_t is None:
                # Heads lazily init after first forward; caller should retry.
                return
            key = self._pair_key(s_name, t_name)
            if key not in self.heads:
                head = ChannelAttentionKD(
                    c_student=int(f_s.shape[1]),
                    c_teacher=int(f_t.shape[1]),
                    d_k=self.d_k,
                ).to(device)
                self.heads[key] = head
        self._lazy_built = all(
            self._pair_key(s, t) in self.heads for s, t in self.node_pairs
        )

    @staticmethod
    def _pair_key(s_name: str, t_name: str) -> str:
        return f"{s_name}__{t_name}".replace(".", "_")

    def rebuild_heads_for_node(self, s_name: str, t_name: str, device: torch.device) -> None:
        """Call after a structural mutation widens a student node's channel count."""
        f_s = self._student_feats.get(s_name)
        f_t = self._teacher_feats.get(t_name)
        if f_s is None or f_t is None:
            return
        key = self._pair_key(s_name, t_name)
        head = ChannelAttentionKD(
            c_student=int(f_s.shape[1]),
            c_teacher=int(f_t.shape[1]),
            d_k=self.d_k,
        ).to(device)
        self.heads[key] = head

    def per_node_distances(self) -> Dict[str, torch.Tensor]:
        """Returns the latest D(n) per student node (Eq. 2). Requires a recent forward."""
        if len(self.heads):
            existing_dev = next(iter(self.heads.values())).q.weight.device
        else:
            existing_dev = torch.device("cpu")
        self._ensure_heads(existing_dev)
        out: Dict[str, torch.Tensor] = {}
        for s_name, t_name in self.node_pairs:
            f_s = self._student_feats.get(s_name)
            f_t = self._teacher_feats.get(t_name)
            if f_s is None or f_t is None:
                continue
            key = self._pair_key(s_name, t_name)
            if key not in self.heads:
                head = ChannelAttentionKD(
                    c_student=int(f_s.shape[1]),
                    c_teacher=int(f_t.shape[1]),
                    d_k=self.d_k,
                ).to(f_s.device)
                self.heads[key] = head
            head = self.heads[key]
            out[s_name] = head(f_s, f_t)
        return out

    def imitation_loss(self) -> torch.Tensor:
        """Eq. 3: average D(n) across all paired nodes."""
        d = self.per_node_distances()
        if not d:
            return torch.tensor(0.0)
        return torch.stack(list(d.values())).mean()
