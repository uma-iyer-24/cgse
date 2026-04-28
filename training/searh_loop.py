"""
Unified SEArch / CGSE outer loop (paper-faithful Algorithm 1).

This single function implements the iterative train-then-evolve loop for
both arms, switching only the *selector* that produces modification
values:

* **SEArch** (paper-faithful): teacher channel-attention KD trains the
  student with ``L = L_CE + λ · L_im`` (Eq. 4, λ cosine-annealed across
  the stage), and at each stage end the per-node attention distance D(n)
  is turned into ``MV(n) = D(n) · deg+/deg-`` (Eq. 5). The candidate
  with the largest MV is split (deepen until B_op cap, then widen).

* **CGSE**: same outer loop, no teacher. The student trains on plain CE.
  At each stage end the critic scores every candidate from the global
  training state plus a per-candidate local descriptor; argmax picks the
  edit. REINFORCE updates the critic at the next stage end with the
  Δval reward.

Both arms terminate when the parameter budget cap is hit or when no
legal candidates remain. An optional final retrain phase trains the
converged architecture for a fixed number of epochs after the loop.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from critics.student_probe import PROBE_DIM, StudentProbe
from evolution.candidates import (
    Candidate,
    count_deepens_in_stage,
    enumerate_candidates,
)
from evolution.searh_mv import (
    compute_per_node_distances,
    modification_values,
    rank_candidates,
)
from models.resnet_cifar import ResNetCifar
from ops.searh_deepen import deepen_resnet_stage
from ops.searh_widen import widen_resnet_stage
from training.loop import evaluate
from training.searh_attention import MultiNodeAttentionKD
from training.searh_node_map import build_node_map
from utils.metrics_csv import append_metrics_csv
from utils.model_info import count_trainable_parameters
from utils.mutation_log import append_mutation_jsonl
from utils.optimizer_utils import refresh_optimizer


# ----------------------------------------------------------------------
# Training step with optional channel-attention imitation loss.
# ----------------------------------------------------------------------
def _train_one_epoch_with_attn(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loader,
    *,
    teacher: Optional[nn.Module],
    attn_kd: Optional[MultiNodeAttentionKD],
    lambda_im: float,
) -> Tuple[float, float, dict]:
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    train_steps = 0
    teacher_forwards = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce = F.cross_entropy(logits, y)

        loss = ce
        if teacher is not None and attn_kd is not None and lambda_im > 0.0:
            with torch.no_grad():
                _ = teacher(x)
            teacher_forwards += 1
            l_im = attn_kd.imitation_loss()
            loss = ce + float(lambda_im) * l_im
        loss.backward()
        optimizer.step()
        train_steps += 1
        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        {"train_steps": train_steps, "teacher_forwards": teacher_forwards},
    )


# ----------------------------------------------------------------------
# Selectors: teacher MV (SEArch) and critic MV (CGSE).
# ----------------------------------------------------------------------
def _teacher_mv_selector(
    student: ResNetCifar,
    teacher: nn.Module,
    attn_kd: MultiNodeAttentionKD,
    score_loader,
    device: torch.device,
    *,
    candidates: List[Candidate],
    score_batches: int,
) -> Tuple[Candidate, Dict[str, float], List[Tuple[Candidate, float]]]:
    """SEArch's argmax over Eq.(5) MV scores."""
    if not candidates:
        raise ValueError("No legal candidates")
    distances = compute_per_node_distances(
        student, teacher, attn_kd, score_loader, device, num_batches=score_batches
    )
    mvs = modification_values(distances)
    ranked = rank_candidates(candidates, mvs)
    return ranked[0][0], mvs, ranked


def _make_local_descriptor(
    cand: Candidate,
    *,
    n_blocks_in_stage: int,
    deepens_in_stage: int,
    b_op_cap: int,
    probe_features: Optional[List[float]] = None,
) -> List[float]:
    """Per-candidate local descriptor used by the CGSE critic.

    Base layout (5 dims, always present):
        [is_stage1, is_stage2, is_stage3, op_is_widen, deepens_in_stage_norm]

    Optional probe extension (3 dims, when ``searh.use_student_probe: true``):
        [act_var_ratio, grad_l2_norm, weight_delta_norm]    — see
        ``critics/student_probe.py`` for definitions.

    The probe extension gives the critic a per-stage *bottleneck signal*
    derived purely from the student's own activations / gradients /
    weights — the unsupervised analogue of SEArch's teacher-derived D(n).
    """
    stage_oh = [0.0, 0.0, 0.0]
    stage_oh[cand.stage - 1] = 1.0
    op_is_widen = 1.0 if cand.op == "widen" else 0.0
    deepens_norm = float(deepens_in_stage) / max(b_op_cap, 1)
    base = [*stage_oh, op_is_widen, deepens_norm]
    if probe_features:
        base.extend(probe_features)
    return base


def _critic_mv_selector(
    student: ResNetCifar,
    critic: nn.Module,
    global_state: torch.Tensor,
    candidates: List[Candidate],
    *,
    b_op_cap: int,
    epsilon: float,
    sample: bool = True,
    probe_features_per_stage: Optional[Dict[str, List[float]]] = None,
) -> Tuple[Candidate, Dict[str, float], List[Tuple[Candidate, float]], int, bool]:
    """
    CGSE's per-candidate scoring. Returns (chosen, mvs, ranked, choice_idx, explored).

    ``probe_features_per_stage`` (when provided) is a dict
    ``{"stage1": [..K..], "stage2": [...], "stage3": [...]}`` whose
    entries are concatenated into the corresponding candidate's local
    descriptor. The critic's ``local_dim`` must equal
    ``5 + len(probe_features_per_stage["stage1"])``.
    """
    if not candidates:
        raise ValueError("No legal candidates")
    device = global_state.device
    desc_rows = []
    for c in candidates:
        n_blocks = len(list(getattr(student, f"layer{c.stage}")))
        deepens = count_deepens_in_stage(student, c.stage)
        probe = (probe_features_per_stage.get(c.node_id)
                 if probe_features_per_stage else None)
        desc_rows.append(_make_local_descriptor(
            c, n_blocks_in_stage=n_blocks, deepens_in_stage=deepens, b_op_cap=b_op_cap,
            probe_features=probe,
        ))
    local = torch.tensor(desc_rows, device=device, dtype=torch.float32)
    g = global_state.detach()
    if g.dim() == 1:
        g = g.unsqueeze(0)
    g_rep = g.expand(len(candidates), -1)
    inp = torch.cat([g_rep, local], dim=-1)
    scores = critic(inp).squeeze(-1)
    probs = F.softmax(scores.detach(), dim=0)

    explored = bool((torch.rand((), device=device) < float(epsilon)).item())
    if explored:
        choice = int(torch.randint(0, len(candidates), (1,), device=device).item())
    elif sample:
        choice = int(torch.multinomial(probs, 1).item())
    else:
        choice = int(scores.argmax().item())

    mvs = {c.node_id: float(scores[i].item()) for i, c in enumerate(candidates)}
    ranked = rank_candidates(candidates, mvs)
    return candidates[choice], mvs, ranked, choice, explored


# ----------------------------------------------------------------------
# Main loop.
# ----------------------------------------------------------------------
def run_searh(
    *,
    cfg: dict,
    student: ResNetCifar,
    optimizer: torch.optim.Optimizer,
    teacher: Optional[ResNetCifar],
    device: torch.device,
    train_loader,
    test_loader,
    experiment_name: str,
    log_csv: Optional[str],
    run_ts: str,
    mlog_path: Optional[Path],
    critic: Optional[nn.Module] = None,
    critic_optimizer: Optional[torch.optim.Optimizer] = None,
    build_critic_state_fn: Optional[Callable] = None,
):
    """
    Run the unified SEArch / CGSE outer loop.

    ``build_critic_state_fn`` is the same ``critics.state_features.build_critic_state``
    used elsewhere in the repo; we expect it to return a (1, 8) tensor.
    """
    s_cfg = cfg["searh"]
    selector_kind = str(s_cfg.get("selector", "teacher")).lower()  # "teacher" | "critic"
    epochs_per_stage = int(s_cfg.get("epochs_per_stage", 8))
    b_op_cap = int(s_cfg.get("B_op", 7))
    deepen_first = bool(s_cfg.get("deepen_first", True))
    score_batches = int(s_cfg.get("score_batches", 2))
    final_retrain_epochs = int(s_cfg.get("final_retrain_epochs", 8))
    gamma = float(s_cfg.get("gamma", 0.5))
    d_k = int(s_cfg.get("d_k", 64))
    lambda_init = float(s_cfg.get("lambda_init", 1.0))
    eps = float(s_cfg.get("epsilon", 0.10))
    entropy_beta = float(s_cfg.get("entropy_beta", 0.01))
    pg_lr = float(s_cfg.get("critic_lr", 0.01))

    # CGSE-only knobs.
    use_probe = bool(s_cfg.get("use_student_probe", False)) and selector_kind == "critic"
    baseline_momentum = float(s_cfg.get("baseline_momentum", 0.9))
    baseline_value = 0.0  # EMA of REINFORCE rewards — subtracted to cut variance

    # Param budget = factor × initial student params.
    p0 = count_trainable_parameters(student)
    cap = int(float(s_cfg.get("param_budget_factor", 1.5)) * p0)
    abs_cap = s_cfg.get("param_budget_abs")
    if abs_cap is not None:
        cap = max(cap, int(abs_cap))
    print(f"[searh] selector={selector_kind} initial_params={p0} budget_cap={cap} "
          f"epochs_per_stage={epochs_per_stage} B_op={b_op_cap}")

    # Build attention KD (teacher arm only).
    attn_kd: Optional[MultiNodeAttentionKD] = None
    if selector_kind == "teacher":
        if teacher is None:
            raise ValueError("searh.selector=teacher requires teacher.enabled")
        node_map = build_node_map(student, teacher, gamma=gamma)
        attn_kd = MultiNodeAttentionKD(
            student=student,
            teacher=teacher,
            node_pairs=list(node_map.pairs),
            student_lookup=dict(node_map.student_lookup),
            teacher_lookup=dict(node_map.teacher_lookup),
            d_k=d_k,
            device=device,
        )
        attn_kd.attach()
        # Probe forward to lazily build attention heads.
        with torch.no_grad():
            x_probe, _ = next(iter(train_loader))
            x_probe = x_probe[:2].to(device)
            student(x_probe)
            teacher(x_probe)
            attn_kd._ensure_heads(device)
        # Add attention parameters to optimizer.
        optimizer = refresh_optimizer(optimizer, _ParamWrapper(student, attn_kd))

    # Logging state.
    run_started = time.perf_counter()
    cum_teacher_forwards = 0
    cum_train_steps = 0
    global_epoch = 0
    mutations_count = 0
    last_mutation_epoch = -1

    # Critic PG bookkeeping (CGSE only).
    pending_pg = None  # dict: state_g, local_inp, choice_idx, val_at_mutation, num_candidates

    # Student probe (CGSE-only): per-stage activation/grad/weight telemetry.
    probe: Optional[StudentProbe] = None
    if use_probe:
        probe = StudentProbe()
        probe.attach(student)
        probe.snapshot_all_stages(student)
        print(f"[searh] student probe enabled (per-stage local_dim grows by {PROBE_DIM})")

    # ------------------------------------------------------------------
    # Stage loop.
    # ------------------------------------------------------------------
    stage_idx = 0
    while True:
        # Stop if budget hit and we've executed at least one stage of training.
        params_now = count_trainable_parameters(student)
        if params_now >= cap and stage_idx > 0:
            print(f"[searh] param budget reached ({params_now} ≥ {cap}); breaking to final retrain.")
            break

        stage_idx += 1
        # ----- 1. Train this stage with cosine λ-anneal (Eq. 4). -----
        prev_val_acc = None
        prev_train_loss = None
        for ep_in_stage in range(epochs_per_stage):
            # cosine anneal λ from lambda_init → 0 across the stage.
            t_frac = ep_in_stage / max(epochs_per_stage - 1, 1)
            lam = 0.5 * lambda_init * (1.0 + math.cos(math.pi * t_frac)) if selector_kind == "teacher" else 0.0
            t0 = time.perf_counter()
            train_loss, train_acc, stats = _train_one_epoch_with_attn(
                model=student,
                optimizer=optimizer,
                device=device,
                loader=train_loader,
                teacher=teacher if selector_kind == "teacher" else None,
                attn_kd=attn_kd if selector_kind == "teacher" else None,
                lambda_im=lam,
            )
            val_loss, val_acc = evaluate(student, device, test_loader)
            epoch_seconds = time.perf_counter() - t0
            wall_seconds = time.perf_counter() - run_started
            cum_teacher_forwards += int(stats.get("teacher_forwards", 0))
            cum_train_steps += int(stats.get("train_steps", 0))

            line = (
                f"[searh] stage {stage_idx} ep {ep_in_stage+1}/{epochs_per_stage} "
                f"(global {global_epoch:03d}) | train_loss {train_loss:.4f} | "
                f"train_acc {train_acc:.4f} | val_loss {val_loss:.4f} | "
                f"val_acc {val_acc:.4f} | λ={lam:.3f} | params {params_now}"
            )
            print(line)

            if log_csv:
                lr_now = float(optimizer.param_groups[0].get("lr", float("nan")))
                row = {
                    "utc_ts": run_ts,
                    "experiment": experiment_name,
                    "epoch": global_epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                    "num_parameters": str(count_trainable_parameters(student)),
                    "mutation_applied_yet": str(mutations_count > 0),
                    "critic_score": "",
                    "optimizer": str(type(optimizer).__name__),
                    "lr": f"{lr_now:.8f}",
                    "epoch_seconds": f"{epoch_seconds:.6f}",
                    "wall_seconds": f"{wall_seconds:.6f}",
                    "train_steps": str(cum_train_steps),
                    "teacher_forwards": str(cum_teacher_forwards),
                    "stage": str(stage_idx),
                    "lambda_im": f"{lam:.6f}",
                    "mutations_count": str(mutations_count),
                }
                append_metrics_csv(Path(log_csv), row)
            global_epoch += 1
            prev_val_acc = val_acc
            prev_train_loss = train_loss

        # ----- 2. PG update for any pending CGSE mutation from prior stage. -----
        if pending_pg is not None and selector_kind == "critic":
            R = float(prev_val_acc) - pending_pg["val_at_mutation"]
            # EMA baseline subtracted from the reward to cut REINFORCE
            # variance — sparse Δval rewards are extremely noisy at small
            # architecture changes, and a baseline lets the critic learn
            # which actions beat the *recent average* rather than chasing
            # absolute Δ noise.
            advantage = R - baseline_value
            baseline_value = (
                baseline_momentum * baseline_value
                + (1.0 - baseline_momentum) * R
            )
            inp = pending_pg["inp"]
            ch_idx = pending_pg["choice"]
            scores_full = critic(inp).squeeze(-1)
            logp = F.log_softmax(scores_full, dim=0)
            entropy = -(F.softmax(scores_full, dim=0) * logp).sum()
            loss_pg = -logp[ch_idx] * torch.as_tensor(
                advantage, device=device, dtype=torch.float32
            )
            loss_pg = loss_pg - float(entropy_beta) * entropy
            critic_optimizer.zero_grad()
            loss_pg.backward()
            critic_optimizer.step()
            print(f"[searh] critic PG update: choice={ch_idx} R={R:+.4f} "
                  f"baseline={baseline_value:+.4f} adv={advantage:+.4f} "
                  f"entropy={entropy.item():.3f}")
            pending_pg = None

        # ----- 3. End-of-stage decision: enumerate, score, split. -----
        params_now = count_trainable_parameters(student)
        if params_now >= cap:
            print(f"[searh] stage {stage_idx} end: params {params_now} ≥ {cap} → no further mutation.")
            continue

        candidates = enumerate_candidates(student, b_op_cap=b_op_cap, deepen_first=deepen_first)
        if not candidates:
            print(f"[searh] stage {stage_idx} end: no legal candidates; stopping evolution.")
            break

        if selector_kind == "teacher":
            chosen, mvs, ranked = _teacher_mv_selector(
                student, teacher, attn_kd, test_loader, device,
                candidates=candidates, score_batches=score_batches,
            )
            cum_teacher_forwards += int(score_batches)  # one teacher fwd per score batch
            extra_pg = None
        else:
            assert critic is not None and build_critic_state_fn is not None
            state = build_critic_state_fn(
                train_loss=prev_train_loss if prev_train_loss is not None else 0.0,
                train_acc=0.0,
                val_loss=0.0,
                val_acc=prev_val_acc if prev_val_acc is not None else 0.0,
                epoch=global_epoch,
                max_epochs=int(s_cfg.get("max_epochs_hint", 50)),
                num_params=params_now,
                prev_train_loss=prev_train_loss,
                prev_val_acc=prev_val_acc,
                anchor_train_loss=prev_train_loss,
                device=device,
            )

            # ---- Probe telemetry (CGSE-only, when use_student_probe=true) ----
            probe_feats: Optional[Dict[str, List[float]]] = None
            if probe is not None:
                probe.update_grads(student)
                try:
                    x_probe, _ = next(iter(test_loader))
                except StopIteration:
                    x_probe, _ = next(iter(train_loader))
                x_probe = x_probe[:64].to(device)
                probe.run_forward(student, x_probe)
                probe_feats = probe.per_stage_features(student)

            chosen, mvs, ranked, choice_idx, explored = _critic_mv_selector(
                student, critic, state, candidates,
                b_op_cap=b_op_cap, epsilon=eps, sample=True,
                probe_features_per_stage=probe_feats,
            )
            # Re-build the input tensor used for PG, identical to selector run.
            desc_rows = []
            for c in candidates:
                n_blocks = len(list(getattr(student, f"layer{c.stage}")))
                deepens = count_deepens_in_stage(student, c.stage)
                pf = probe_feats.get(c.node_id) if probe_feats else None
                desc_rows.append(_make_local_descriptor(
                    c, n_blocks_in_stage=n_blocks, deepens_in_stage=deepens, b_op_cap=b_op_cap,
                    probe_features=pf,
                ))
            local = torch.tensor(desc_rows, device=device, dtype=torch.float32)
            g = state.detach()
            if g.dim() == 1: g = g.unsqueeze(0)
            inp = torch.cat([g.expand(len(candidates), -1), local], dim=-1).detach()
            extra_pg = {"inp": inp, "choice": int(choice_idx),
                        "val_at_mutation": float(prev_val_acc or 0.0)}

        # ----- 4. Apply chosen mutation. -----
        params_before = count_trainable_parameters(student)
        if chosen.op == "deepen":
            deepen_resnet_stage(student, stage=chosen.stage, after_block_idx=None)
        elif chosen.op == "widen":
            widen_resnet_stage(student, stage=chosen.stage, block_idx=None)
        else:
            raise ValueError(f"Unknown op '{chosen.op}'")
        # Refresh optimizer to include new params.
        if selector_kind == "teacher" and attn_kd is not None:
            optimizer = refresh_optimizer(optimizer, _ParamWrapper(student, attn_kd))
        else:
            optimizer = refresh_optimizer(optimizer, student)
        params_after = count_trainable_parameters(student)
        mutations_count += 1
        last_mutation_epoch = global_epoch - 1

        # Reset the probe's weight-Δ baseline for the mutated stage so the
        # next per-stage feature reflects post-mutation drift (= "have we
        # learned anything in this stage since the last edit?").
        if probe is not None:
            probe.snapshot_stage(student, chosen.node_id)

        print(f"[searh] stage {stage_idx} mutation #{mutations_count}: "
              f"{chosen.describe()} (mv={mvs.get(chosen.node_id, 0.0):+.4f}) "
              f"params {params_before} → {params_after}")
        if mlog_path is not None:
            top5 = [(c.describe(), v) for c, v in ranked[:5]]
            append_mutation_jsonl(
                mlog_path,
                {
                    "event": "mutation",
                    "op": f"searh_{chosen.op}",
                    "selector": selector_kind,
                    "gate": "searh_loop",
                    "run_id": f"{experiment_name}_{run_ts}",
                    "experiment": experiment_name,
                    "epoch_completed": global_epoch - 1,
                    "stage": stage_idx,
                    "mutation_index": mutations_count,
                    "stage_target": chosen.stage,
                    "node_id": chosen.node_id,
                    "mv": float(mvs.get(chosen.node_id, 0.0)),
                    "ranked_top5": top5,
                    "num_parameters_before": params_before,
                    "num_parameters_after": params_after,
                    "param_budget_cap": int(cap),
                    "use_probe": bool(probe is not None),
                    "baseline_value": float(baseline_value),
                },
            )

        if selector_kind == "critic" and extra_pg is not None:
            pending_pg = extra_pg

    # ----------------------------------------------------------------------
    # Final retrain phase (no mutations) — paper's appendix recipe.
    # ----------------------------------------------------------------------
    if final_retrain_epochs > 0:
        print(f"[searh] final retrain for {final_retrain_epochs} epochs (no mutations).")
        for k in range(final_retrain_epochs):
            t0 = time.perf_counter()
            train_loss, train_acc, stats = _train_one_epoch_with_attn(
                model=student,
                optimizer=optimizer,
                device=device,
                loader=train_loader,
                teacher=teacher if selector_kind == "teacher" else None,
                attn_kd=attn_kd if selector_kind == "teacher" else None,
                lambda_im=0.0,  # final retrain: pure CE per paper
            )
            val_loss, val_acc = evaluate(student, device, test_loader)
            epoch_seconds = time.perf_counter() - t0
            wall_seconds = time.perf_counter() - run_started
            cum_teacher_forwards += int(stats.get("teacher_forwards", 0))
            cum_train_steps += int(stats.get("train_steps", 0))
            print(f"[searh] retrain {k+1}/{final_retrain_epochs} (global {global_epoch:03d}) | "
                  f"val_acc {val_acc:.4f}")
            if log_csv:
                lr_now = float(optimizer.param_groups[0].get("lr", float("nan")))
                row = {
                    "utc_ts": run_ts,
                    "experiment": experiment_name,
                    "epoch": global_epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                    "num_parameters": str(count_trainable_parameters(student)),
                    "mutation_applied_yet": str(mutations_count > 0),
                    "critic_score": "",
                    "optimizer": str(type(optimizer).__name__),
                    "lr": f"{lr_now:.8f}",
                    "epoch_seconds": f"{epoch_seconds:.6f}",
                    "wall_seconds": f"{wall_seconds:.6f}",
                    "train_steps": str(cum_train_steps),
                    "teacher_forwards": str(cum_teacher_forwards),
                    "stage": "retrain",
                    "lambda_im": "0.000000",
                    "mutations_count": str(mutations_count),
                }
                append_metrics_csv(Path(log_csv), row)
            global_epoch += 1

    if attn_kd is not None:
        attn_kd.detach()
    if probe is not None:
        probe.detach()
    print(f"[searh] DONE. mutations={mutations_count} final_params={count_trainable_parameters(student)} "
          f"epochs_run={global_epoch}")


class _ParamWrapper:
    """Adapter so refresh_optimizer can collect params from student + attn_kd jointly."""

    def __init__(self, *modules):
        self._modules = modules

    def parameters(self):
        seen = set()
        for m in self._modules:
            if m is None: continue
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p
