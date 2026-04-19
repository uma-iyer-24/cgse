"""
Train structurally mutable students (CIFAR or synthetic MLP).

Experimental arms (see root README): (1) optional frozen *teacher* + KD — SEArch-style
control; (2) CGSE — *critic* gates widen timing (REINFORCE on post-mutation val gain).
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from critics import StructuralCritic, STATE_DIM, build_critic_state
from models.cifar_student import CifarGraphNet
from models.resnet_cifar import ResNetCifar
from models.student import StudentNet
from ops.edge_widen import edge_widen
from training.data import build_cifar10_loaders
from training.loop import evaluate, train_one_epoch
from training.synthetic import build_synthetic_loaders
from utils.checkpoint import load_model_weights, save_checkpoint
from utils.graph_validator import validate_forward
from utils.model_info import (
    count_trainable_parameters,
    first_linear_node_id,
    linear_layer_shapes,
)
from utils.metrics_csv import append_metrics_csv
from utils.mutation_log import append_mutation_jsonl
from utils.optimizer_utils import refresh_optimizer
from utils.repro import set_seed
from utils.artifact_families import (
    canonicalize_runs_artifact,
    resolve_teacher_checkpoint,
    student_checkpoint_path,
    structural_critic_checkpoint_path,
)
from utils.run_paths import normalize_run_artifact_path


def _path_stem_suffix(path: str | None, suffix: str) -> str | None:
    """Insert suffix before extension (e.g. metrics_seed42.csv)."""
    if not path or not suffix:
        return path
    p = Path(path)
    return str(p.parent / f"{p.stem}{suffix}{p.suffix}")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _run_edge_widen_mutation(
    model,
    optimizer,
    widen_delta: int,
    *,
    mutation_log_jsonl: Path | None,
    experiment_name: str,
    run_ts: str,
    epoch: int,
    gate_tag: str,
):
    params_before = count_trainable_parameters(model)
    target_id = first_linear_node_id(model)
    if target_id is None:
        raise RuntimeError("mutation enabled but no Linear layer found in graph")
    in_f, out_before = linear_layer_shapes(model, target_id)
    edge_widen(model, target_node_id=target_id, delta=widen_delta)
    optimizer = refresh_optimizer(optimizer, model)
    _, out_after = linear_layer_shapes(model, target_id)
    params_after = count_trainable_parameters(model)
    print(
        f"[mutation] ({gate_tag}) Applied edge_widen({target_id}, delta={widen_delta}) after epoch {epoch}; "
        f"params {params_before} -> {params_after}; optimizer refreshed."
    )
    if mutation_log_jsonl:
        append_mutation_jsonl(
            mutation_log_jsonl,
            {
                "event": "mutation",
                "op": "edge_widen",
                "gate": gate_tag,
                "run_id": f"{experiment_name}_{run_ts}",
                "experiment": experiment_name,
                "epoch_completed": epoch,
                "target_node_id": target_id,
                "delta": widen_delta,
                "target_linear_in": in_f,
                "target_linear_out_before": out_before,
                "target_linear_out_after": out_after,
                "num_parameters_before": params_before,
                "num_parameters_after": params_after,
            },
        )
    return optimizer


def main():
    parser = argparse.ArgumentParser(description="CGSE training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cifar/phase2_cifar.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override config device (e.g. cpu, mps, cuda, auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training.seed; also appends _seed<N> to experiment name and metrics/mutation paths",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    device = resolve_device(args.device or cfg.get("device", "cpu"))

    seed = int(args.seed) if args.seed is not None else int(train_cfg.get("seed", 42))
    seed_tag = f"_seed{seed}" if args.seed is not None else ""
    experiment_name = cfg["experiment"]["name"] + seed_tag
    set_seed(seed)
    if args.seed is not None:
        print(f"[repro] Using --seed {seed}; experiment_name={experiment_name}")

    model_name = cfg.get("model", {}).get("name", "mlp")

    teacher = None
    if model_name in {"cifar_cnn", "resnet_cifar"}:
        train_loader, test_loader = build_cifar10_loaders(cfg)
        num_classes = int(cfg["model"]["num_classes"])
        if model_name == "cifar_cnn":
            model = CifarGraphNet(num_classes=num_classes).to(device)
            sample = next(iter(train_loader))[0][:2].to(device)
            validate_forward(model, sample)
        else:
            depth = int(cfg["model"].get("depth", 20))
            base_width = int(cfg["model"].get("base_width", 16))
            model = ResNetCifar(depth=depth, num_classes=num_classes, base_width=base_width).to(device)

    else:
        train_loader, test_loader = build_synthetic_loaders(cfg)
        model = StudentNet(**cfg["model"]).to(device)

    t_cfg = cfg.get("teacher") or {}
    if model_name in {"cifar_cnn", "resnet_cifar"} and bool(t_cfg.get("enabled", False)):
        ckpt_path = t_cfg.get("checkpoint")
        if not ckpt_path:
            raise ValueError("teacher.enabled is true but teacher.checkpoint is missing")
        num_classes = int(cfg["model"]["num_classes"])
        if model_name == "cifar_cnn":
            teacher = CifarGraphNet(num_classes=num_classes).to(device)
        else:
            t_depth = int(t_cfg.get("depth", cfg["model"].get("teacher_depth", 56)))
            t_base_width = int(t_cfg.get("base_width", cfg["model"].get("base_width", 16)))
            teacher = ResNetCifar(depth=t_depth, num_classes=num_classes, base_width=t_base_width).to(device)
        resolved_teacher = str(resolve_teacher_checkpoint(str(ckpt_path)))
        load_model_weights(teacher, resolved_teacher)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[teacher] Loaded frozen teacher from {resolved_teacher} (KD training).")

    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        nesterov = bool(train_cfg.get("nesterov", True))
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}' (expected 'adam' or 'sgd').")

    # Optional per-epoch LR schedule (Tier 2 parity uses long SGD).
    sched_cfg = train_cfg.get("lr_schedule") or {}
    sched_name = str(sched_cfg.get("name", "none")).lower()
    scheduler = None
    if sched_name in {"none", ""}:
        scheduler = None
    elif sched_name == "multistep":
        milestones = [int(x) for x in (sched_cfg.get("milestones") or [])]
        gamma = float(sched_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif sched_name == "cosine":
        tmax = int(sched_cfg.get("t_max", int(train_cfg["epochs"])))
        eta_min = float(sched_cfg.get("eta_min", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tmax, eta_min=eta_min
        )
    else:
        raise ValueError(
            f"Unknown lr_schedule.name '{sched_name}' (expected none|multistep|cosine)."
        )

    mutation_cfg = cfg.get("mutation") or {}
    mutation_enabled = bool(mutation_cfg.get("enabled", False))
    once_after = mutation_cfg.get("once_after_epoch")
    widen_delta = int(mutation_cfg.get("widen_delta", 32))
    mutation_applied = False
    mutation_log_jsonl = normalize_run_artifact_path(
        _path_stem_suffix(mutation_cfg.get("log_jsonl"), seed_tag)
    )
    mutation_log_jsonl = canonicalize_runs_artifact(
        mutation_log_jsonl, experiment_name, "mutations"
    )

    c_cfg = cfg.get("critic") or {}
    critic_on = bool(c_cfg.get("enabled", False))
    if critic_on:
        if not mutation_enabled:
            raise ValueError("critic.enabled requires mutation.enabled")
        if teacher is not None:
            raise ValueError("Disable teacher when critic.enabled (CGSE arm uses no KD teacher).")

    critic = None
    critic_opt = None
    win_start = int(c_cfg.get("window_start_epoch", 5))
    win_end = int(c_cfg.get("window_end_epoch", 35))
    eps = float(c_cfg.get("epsilon", 0.2))
    force_end = bool(c_cfg.get("force_mutate_end_of_window", True))
    if critic_on:
        critic = StructuralCritic(
            state_dim=STATE_DIM,
            hidden_dim=int(c_cfg.get("hidden_dim", 32)),
        ).to(device)
        critic_opt = torch.optim.Adam(
            critic.parameters(),
            lr=float(c_cfg.get("lr", 0.01)),
            weight_decay=float(c_cfg.get("weight_decay", 0.0)),
        )
        print(
            f"[critic] CGSE mutation gating: window [{win_start}, {win_end}], "
            f"ε={eps}, force_end={force_end}"
        )

    log_csv = normalize_run_artifact_path(
        _path_stem_suffix(train_cfg.get("log_csv"), seed_tag)
    )
    log_csv = canonicalize_runs_artifact(log_csv, experiment_name, "metrics")
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    evo_cfg = cfg.get("evolution") or {}
    if bool(evo_cfg.get("enabled")):
        if model_name != "cifar_cnn":
            raise ValueError("evolution.enabled requires model.name cifar_cnn")
        if mutation_enabled or critic_on:
            raise ValueError(
                "When evolution.enabled, set mutation.enabled: false and critic.enabled: false"
            )
        from training.evolution_train import run_evolution_training

        evo_mlog = Path(mutation_log_jsonl) if mutation_log_jsonl else None
        run_evolution_training(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            teacher=teacher,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            experiment_name=experiment_name,
            log_csv=log_csv,
            run_ts=run_ts,
            t_cfg=t_cfg,
            mlog_path=evo_mlog,
        )
        ck_path = student_checkpoint_path(experiment_name)
        save_checkpoint(model, optimizer, str(ck_path))
        print(f"Saved checkpoint to {ck_path}")
        return

    kd_temp = float(t_cfg.get("temperature", 4.0))
    kd_alpha = float(t_cfg.get("alpha", 0.5))

    pending_pg = None
    prev_train_loss = None
    prev_val_acc = None
    anchor_train_loss = None

    epochs = int(train_cfg["epochs"])
    mlog_path = Path(mutation_log_jsonl) if mutation_log_jsonl else None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            optimizer,
            device,
            train_loader,
            teacher=teacher,
            kd_temperature=kd_temp,
            kd_alpha=kd_alpha,
        )
        val_loss, val_acc = evaluate(model, device, test_loader)
        if scheduler is not None:
            scheduler.step()

        if pending_pg is not None:
            assert critic_on and critic is not None and critic_opt is not None
            R = val_acc - pending_pg["val_at_mutation"]
            if not pending_pg["skip_pg"]:
                critic_opt.zero_grad()
                s = pending_pg["state"]
                loss_pg = -F.logsigmoid(critic(s)) * torch.as_tensor(
                    R, device=device, dtype=torch.float32
                )
                loss_pg.backward()
                critic_opt.step()
            pending_pg = None

        if epoch == 0:
            anchor_train_loss = train_loss

        n_params_pre = count_trainable_parameters(model)
        critic_score_val = None
        state = None
        if critic_on:
            state = build_critic_state(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                epoch=epoch,
                max_epochs=epochs,
                num_params=n_params_pre,
                prev_train_loss=prev_train_loss,
                prev_val_acc=prev_val_acc,
                anchor_train_loss=anchor_train_loss,
                device=device,
            )
            with torch.no_grad():
                critic_score_val = critic(state).item()

        if (
            mutation_enabled
            and not critic_on
            and not mutation_applied
            and once_after is not None
            and epoch == int(once_after)
        ):
            optimizer = _run_edge_widen_mutation(
                model,
                optimizer,
                widen_delta,
                mutation_log_jsonl=mlog_path,
                experiment_name=experiment_name,
                run_ts=run_ts,
                epoch=epoch,
                gate_tag="schedule",
            )
            mutation_applied = True

        if (
            mutation_enabled
            and critic_on
            and not mutation_applied
            and state is not None
            and win_start <= epoch <= win_end
        ):
            logit = critic(state)
            p = torch.sigmoid(logit)
            explored = bool((torch.rand((), device=device) < eps).item())
            if explored:
                action_val = int(torch.randint(0, 2, (1,), device=device).item())
            else:
                action_val = int(torch.bernoulli(p).item())
            forced = bool(force_end and epoch == win_end and action_val == 0)
            if forced:
                action_val = 1
            if action_val == 1:
                optimizer = _run_edge_widen_mutation(
                    model,
                    optimizer,
                    widen_delta,
                    mutation_log_jsonl=mlog_path,
                    experiment_name=experiment_name,
                    run_ts=run_ts,
                    epoch=epoch,
                    gate_tag="critic",
                )
                mutation_applied = True
                pending_pg = {
                    "state": state.detach().clone(),
                    "val_at_mutation": float(val_acc),
                    "skip_pg": explored or forced,
                }

        line = (
            f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )
        if mutation_applied:
            line += " | mutation=on"
        if teacher is not None:
            line += f" | kd T={kd_temp} α={kd_alpha}"
        if critic_score_val is not None:
            line += f" | critic {critic_score_val:.4f}"
        print(line)

        n_params = count_trainable_parameters(model)
        if log_csv:
            row = {
                "utc_ts": run_ts,
                "experiment": experiment_name,
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "num_parameters": str(n_params),
                "mutation_applied_yet": str(mutation_applied),
                "critic_score": (
                    f"{critic_score_val:.6f}" if critic_score_val is not None else ""
                ),
            }
            append_metrics_csv(Path(log_csv), row)

        prev_train_loss = train_loss
        prev_val_acc = val_acc

    ck_path = student_checkpoint_path(experiment_name)
    save_checkpoint(model, optimizer, str(ck_path))
    print(f"Saved checkpoint to {ck_path}")
    if critic_on and critic is not None:
        c_path = structural_critic_checkpoint_path(experiment_name)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"critic": critic.state_dict()}, c_path)
        print(f"Saved critic weights to {c_path}")


if __name__ == "__main__":
    main()
