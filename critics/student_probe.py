"""
Student-only state probe for CGSE.

Gives the critic the locality signal SEArch gets from the teacher, but
derived purely from the student's own activations / weights / gradients
— no teacher reference, zero teacher forwards.

Per stage we compute three scalar features (this is the descriptor that
gets concatenated into the per-candidate ``local_descriptor``):

* **act_var_ratio** — top-1 principal-component variance of the pooled
  per-channel activation matrix divided by total variance. This is the
  *unsupervised analogue of SEArch's D(n)*: when a stage's representations
  collapse onto a few directions the ratio jumps toward 1, which is a
  direct bottleneck signal. Computed without SVD via trace-of-covariance
  identities for speed (≈0.5 ms per stage on MPS).
* **grad_l2** — Euclidean norm of the per-stage parameter gradients,
  captured from the last backward pass. Where the network is actually
  learning vs. where signal is dying.
* **weight_delta** — Frobenius distance between current stage weights and
  the snapshot taken at the most recent mutation in this stage. "Has this
  stage converged?" detector; collapses to 0 right after a mutation.

Both ``grad_l2`` and ``weight_delta`` are min-max normalised across
stages at probe time so the critic sees a comparable scale across
features and across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.resnet_cifar import ResNetCifar


STAGE_IDS: Tuple[str, str, str] = ("stage1", "stage2", "stage3")
PROBE_DIM = 3  # act_var_ratio, grad_l2, weight_delta


def _stage_module(model: ResNetCifar, stage_id: str) -> nn.Module:
    return getattr(model, "layer" + stage_id[-1])


def _stage_params(model: ResNetCifar, stage_id: str) -> List[torch.Tensor]:
    return [p for p in _stage_module(model, stage_id).parameters() if p.requires_grad]


@torch.no_grad()
def _activation_variance_ratio(act: torch.Tensor) -> float:
    """
    Top-1 principal-component variance ratio of the *channel* covariance.

    For ``act`` of shape (B, C, H, W) we view each channel as a sample of
    length B*H*W, centre, then compute the channel covariance matrix
    ``Σ ∈ R^{C×C}`` and return ``λ_max(Σ) / trace(Σ)``.

    To avoid the C×C eigendecomposition we approximate ``λ_max`` by power
    iteration with 4 steps from a random init — exact enough for a noisy
    REINFORCE input, and ~10× faster than ``torch.linalg.eigh`` for the
    64-channel feature maps we hit.
    """
    b, c, h, w = act.shape
    x = act.reshape(b, c, -1).transpose(1, 2).reshape(-1, c)        # (B·H·W, C)
    x = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    if n <= 1:
        return 0.0
    # trace(Σ) = (1/(n-1)) · sum_i ||x_i||² (per-sample squared norm sum)
    total = float((x.pow(2).sum() / (n - 1)).item())
    if total <= 1e-12:
        return 0.0
    # Power iteration on Σ via repeated x.T @ (x @ v).
    v = torch.randn(c, device=act.device, dtype=act.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(4):
        Av = x.t() @ (x @ v) / (n - 1)
        v = Av / (Av.norm() + 1e-12)
    Av = x.t() @ (x @ v) / (n - 1)
    lam = float((v @ Av).item())
    return float(max(0.0, min(1.0, lam / total)))


@dataclass
class StudentProbe:
    """
    Lightweight per-stage telemetry for CGSE.

    Usage:
        probe = StudentProbe()
        probe.attach(student)         # one-time
        probe.snapshot_all_stages(student)   # call at the very start (baseline)
        ...
        # End-of-stage scoring step (after a backward pass already happened):
        probe.update_grads(student)
        probe.run_forward(student, x_batch)   # one no-grad fwd to populate acts
        feats = probe.per_stage_features(student)   # Dict[stage_id, List[float]]
        # ...applied a mutation in stage_k...
        probe.snapshot_stage(student, "stage" + str(stage_k))   # reset Δw baseline

        probe.detach()                 # before pickle / final save
    """

    _hooks: list = field(default_factory=list)
    _acts: Dict[str, torch.Tensor] = field(default_factory=dict)
    _grad_l2: Dict[str, float] = field(default_factory=dict)
    _weight_snapshots: Dict[str, List[torch.Tensor]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------
    def attach(self, model: ResNetCifar) -> None:
        if self._hooks:
            return
        for sid in STAGE_IDS:
            mod = _stage_module(model, sid)

            def make_hook(s=sid):
                def fn(_m, _i, out):
                    self._acts[s] = out.detach()
                return fn

            self._hooks.append(mod.register_forward_hook(make_hook()))

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Telemetry collection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_forward(self, model: ResNetCifar, x: torch.Tensor) -> None:
        """One unlabelled forward to populate per-stage activation stores."""
        was_training = model.training
        model.eval()
        try:
            model(x)
        finally:
            if was_training:
                model.train()

    def update_grads(self, model: ResNetCifar) -> None:
        """Capture per-stage parameter-gradient L2 from the last backward pass."""
        for sid in STAGE_IDS:
            total = 0.0
            for p in _stage_params(model, sid):
                if p.grad is not None:
                    total += float(p.grad.detach().pow(2).sum().item())
            self._grad_l2[sid] = float(total) ** 0.5

    @torch.no_grad()
    def snapshot_all_stages(self, model: ResNetCifar) -> None:
        for sid in STAGE_IDS:
            self.snapshot_stage(model, sid)

    @torch.no_grad()
    def snapshot_stage(self, model: ResNetCifar, stage_id: str) -> None:
        self._weight_snapshots[stage_id] = [
            p.detach().clone() for p in _stage_params(model, stage_id)
        ]

    @torch.no_grad()
    def _weight_delta(self, model: ResNetCifar, stage_id: str) -> float:
        snap = self._weight_snapshots.get(stage_id)
        if snap is None:
            return 0.0
        cur = _stage_params(model, stage_id)
        # Snapshot may be from an older architecture (pre-mutation); zip
        # over the smaller list and ignore extra new params (they have no
        # baseline to diff against — counted as 0 delta until next snapshot).
        total = 0.0
        for old, new in zip(snap, cur):
            if old.shape != new.shape:
                # Mutation grew this tensor; take diff over overlap region.
                slicer = tuple(slice(0, min(o, n)) for o, n in zip(old.shape, new.shape))
                total += float((new[slicer] - old[slicer]).pow(2).sum().item())
            else:
                total += float((new - old).pow(2).sum().item())
        return total ** 0.5

    # ------------------------------------------------------------------
    # Critic-facing aggregation
    # ------------------------------------------------------------------
    def per_stage_features(self, model: ResNetCifar) -> Dict[str, List[float]]:
        """Returns per-stage [act_var_ratio, grad_l2_norm, weight_delta_norm]."""
        # 1. Variance ratio per stage (uses cached activations).
        var_ratio = {sid: _activation_variance_ratio(self._acts[sid])
                     if sid in self._acts else 0.0
                     for sid in STAGE_IDS}
        # 2. Grad L2 — min-max normalise across stages so the critic sees
        #    a relative magnitude (which stage is most active).
        grads = {sid: float(self._grad_l2.get(sid, 0.0)) for sid in STAGE_IDS}
        gmax = max(grads.values()) if grads else 1.0
        grads_norm = {sid: (grads[sid] / gmax) if gmax > 1e-12 else 0.0
                      for sid in STAGE_IDS}
        # 3. Weight-Δ since last snapshot — same min-max normalisation.
        deltas = {sid: self._weight_delta(model, sid) for sid in STAGE_IDS}
        dmax = max(deltas.values()) if deltas else 1.0
        deltas_norm = {sid: (deltas[sid] / dmax) if dmax > 1e-12 else 0.0
                       for sid in STAGE_IDS}
        return {sid: [var_ratio[sid], grads_norm[sid], deltas_norm[sid]]
                for sid in STAGE_IDS}
