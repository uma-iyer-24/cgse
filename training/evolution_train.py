"""
Multi-stage evolution training (Tier 1b): SEArch-like stages + discrete mutations.

See paper_documentation/SEArch-baseline-and-CGSE-evaluation-plan.md §7.
"""

from __future__ import annotations

from pathlib import Path
import time

import torch
import torch.nn.functional as F

from critics import STATE_DIM, build_critic_state
from critics.discrete_critic import DiscreteMutationCritic
from training.loop import evaluate, train_one_epoch
from utils.evolution_apply import apply_named_mutation, filter_legal_candidates
from utils.graph_validator import validate_forward
from utils.model_info import count_trainable_parameters

from utils.artifact_families import discrete_critic_checkpoint_path
from utils.metrics_csv import append_metrics_csv


def run_evolution_training(
    *,
    cfg: dict,
    model,
    optimizer,
    teacher,
    device,
    train_loader,
    test_loader,
    experiment_name: str,
    log_csv: str | None,
    run_ts: str,
    t_cfg: dict,
    mlog_path: Path | None,
) -> None:
    evo = cfg["evolution"]
    epochs_per_stage = int(evo["epochs_per_stage"])
    num_stages = int(evo["num_stages"])
    mode = str(evo.get("mode", "schedule"))
    widen_delta = int(evo.get("widen_fc_delta", cfg.get("mutation", {}).get("widen_delta", 32)))
    conv_delta = int(evo.get("widen_conv_delta", 32))
    max_params = evo.get("max_parameters")
    if max_params is not None:
        max_params = int(max_params)

    kd_temp = float(t_cfg.get("temperature", 4.0))
    kd_alpha = float(t_cfg.get("alpha", 0.5))

    candidate_names: list[str] = list(evo.get("candidates") or [])
    schedule = list(evo.get("schedule") or [])

    critic_mod = None
    critic_opt = None
    max_act = max(8, len(candidate_names) if candidate_names else 1)
    eps = 0.2
    if mode == "critic":
        if not candidate_names:
            raise ValueError("evolution.mode=critic requires evolution.candidates")
        cc = evo.get("critic") or {}
        eps = float(cc.get("epsilon", 0.2))
        critic_mod = DiscreteMutationCritic(
            STATE_DIM, max_actions=max_act, hidden_dim=int(cc.get("hidden_dim", 64))
        ).to(device)
        critic_opt = torch.optim.Adam(
            critic_mod.parameters(),
            lr=float(cc.get("lr", 0.01)),
            weight_decay=float(cc.get("weight_decay", 0.0)),
        )
        print(f"[evolution] critic mode: {len(candidate_names)} candidate types, ε={eps}")

    applied_ops: set[str] = set()
    pending_pg: dict | None = None
    prev_train_loss = None
    prev_val_acc = None
    anchor_train_loss = None
    global_epoch = 0
    total_epochs = num_stages * epochs_per_stage
    run_started = time.perf_counter()
    wall_seconds = 0.0
    cum_teacher_forwards = 0
    cum_train_steps = 0

    sample = next(iter(train_loader))[0][:2].to(device)
    stopped_early = False

    for stage in range(num_stages):
        for _ in range(epochs_per_stage):
            t0 = time.perf_counter()
            train_loss, train_acc, stats = train_one_epoch(
                model,
                optimizer,
                device,
                train_loader,
                teacher=teacher,
                kd_temperature=kd_temp,
                kd_alpha=kd_alpha,
                return_stats=True,
            )
            val_loss, val_acc = evaluate(model, device, test_loader)
            epoch_seconds = time.perf_counter() - t0
            wall_seconds = time.perf_counter() - run_started
            cum_teacher_forwards += int(stats.get("teacher_forwards", 0))
            cum_train_steps += int(stats.get("train_steps", 0))

            if pending_pg is not None and critic_mod is not None and critic_opt is not None:
                R = val_acc - pending_pg["val_at_mutation"]
                if not pending_pg["skip_pg"]:
                    critic_opt.zero_grad()
                    st = pending_pg["state"].to(device)
                    legal_names = pending_pg["legal_names"]
                    ch = int(pending_pg["choice"])
                    idx_t = torch.tensor(
                        [candidate_names.index(n) for n in legal_names],
                        device=device,
                        dtype=torch.long,
                    )
                    logits_full = critic_mod(st)
                    logits = logits_full[idx_t]
                    logp = F.log_softmax(logits, dim=-1)[ch]
                    loss_pg = -logp * torch.as_tensor(
                        R, device=device, dtype=torch.float32
                    )
                    loss_pg.backward()
                    critic_opt.step()
                pending_pg = None

            if global_epoch == 0:
                anchor_train_loss = train_loss

            n_params_pre = count_trainable_parameters(model)
            if max_params is not None and n_params_pre > max_params:
                print(f"[evolution] hit max_parameters {max_params}; stopping early.")
                stopped_early = True
                break

            critic_score_val = None
            state = build_critic_state(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                epoch=global_epoch,
                max_epochs=max(total_epochs - 1, 1),
                num_params=n_params_pre,
                prev_train_loss=prev_train_loss,
                prev_val_acc=prev_val_acc,
                anchor_train_loss=anchor_train_loss,
                device=device,
            )

            line = (
                f"Epoch {global_epoch:03d} (stage {stage}) | train_loss {train_loss:.4f} | "
                f"train_acc {train_acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
            )
            if teacher is not None:
                line += f" | kd T={kd_temp} α={kd_alpha}"
            print(line)

            n_params = count_trainable_parameters(model)
            if log_csv:
                append_metrics_csv(
                    Path(log_csv),
                    {
                        "utc_ts": run_ts,
                        "experiment": experiment_name,
                        "epoch": global_epoch,
                        "train_loss": f"{train_loss:.6f}",
                        "train_acc": f"{train_acc:.6f}",
                        "val_loss": f"{val_loss:.6f}",
                        "val_acc": f"{val_acc:.6f}",
                        "num_parameters": str(n_params),
                        "mutation_applied_yet": str(bool(applied_ops)),
                        "critic_score": (
                            f"{critic_score_val:.6f}"
                            if critic_score_val is not None
                            else ""
                        ),
                        "optimizer": str(type(optimizer).__name__).lower(),
                        "lr": f"{float(optimizer.param_groups[0].get('lr', float('nan'))):.8f}",
                        "epoch_seconds": f"{epoch_seconds:.6f}",
                        "wall_seconds": f"{wall_seconds:.6f}",
                        "train_steps": str(cum_train_steps),
                        "teacher_forwards": str(cum_teacher_forwards),
                    },
                )

            prev_train_loss = train_loss
            prev_val_acc = val_acc
            global_epoch += 1

        if stopped_early:
            break

        # --- end of stage: optional mutation ---
        validate_forward(model, sample)

        if mode == "schedule":
            for row in schedule:
                if int(row.get("after_stage", -1)) != stage:
                    continue
                op = str(row["op"])
                if op in applied_ops:
                    continue
                wd = int(row.get("widen_delta", widen_delta))
                cd = int(row.get("conv_delta", conv_delta))
                optimizer = apply_named_mutation(
                    model,
                    optimizer,
                    op,
                    widen_delta=wd,
                    conv_delta=cd,
                    mutation_log_jsonl=mlog_path,
                    experiment_name=experiment_name,
                    run_ts=run_ts,
                    global_epoch=global_epoch - 1,
                    gate_tag="evolution_schedule",
                )
                applied_ops.add(op)
                validate_forward(model, sample)

        elif mode == "critic":
            legal = filter_legal_candidates(model, candidate_names, applied_ops)
            if not legal:
                continue
            logits_full = critic_mod(state)
            idx_t = torch.tensor(
                [candidate_names.index(n) for n in legal],
                device=device,
                dtype=torch.long,
            )
            logits = logits_full[idx_t]
            k = len(legal)
            probs = F.softmax(logits.detach(), dim=-1)
            explored = bool((torch.rand((), device=device) < eps).item())
            if explored:
                choice = int(torch.randint(0, k, (1,), device=device).item())
                skip_pg = True
            else:
                choice = int(torch.multinomial(probs, 1).item())
                skip_pg = False
            op = legal[choice]
            optimizer = apply_named_mutation(
                model,
                optimizer,
                op,
                widen_delta=widen_delta,
                conv_delta=conv_delta,
                mutation_log_jsonl=mlog_path,
                experiment_name=experiment_name,
                run_ts=run_ts,
                global_epoch=global_epoch - 1,
                gate_tag="evolution_critic",
            )
            applied_ops.add(op)
            pending_pg = {
                "state": state.detach().clone(),
                "legal_names": list(legal),
                "choice": choice,
                "val_at_mutation": float(val_acc),
                "skip_pg": skip_pg,
            }
            validate_forward(model, sample)

    validate_forward(model, sample)

    if critic_mod is not None:
        c_path = discrete_critic_checkpoint_path(experiment_name)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"critic": critic_mod.state_dict()}, c_path)
        print(f"Saved discrete critic to {c_path}")
