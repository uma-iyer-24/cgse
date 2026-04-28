"""
Quick end-to-end smoke test for the SEArch + CGSE-on-SEArch outer loop.

Runs:
  * 2 stages × 1 epoch each (no final retrain) on CIFAR-10
  * Both selectors (teacher + critic) in sequence
  * A small subset_train so the test finishes in <60s on CPU/MPS

Verifies that:
  - the loop spins, mutations apply, optimizer refreshes correctly
  - attn-KD hooks attach and produce per-stage distances (teacher arm)
  - the critic produces per-candidate scores and REINFORCE updates fire (CGSE arm)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from critics import build_critic_state, STATE_DIM
from critics.searh_critic import PerCandidateCritic
from models.resnet_cifar import ResNetCifar
from training.data import build_cifar10_loaders
from training.searh_loop import run_searh
from utils.checkpoint import load_model_weights


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _make_loaders():
    cfg = {
        "data": {
            "name": "cifar10",
            "root": "./data",
            "num_workers": 0,
            "subset_train": 256,
            "subset_test": 256,
        },
        "training": {"batch_size": 64},
    }
    return build_cifar10_loaders(cfg)


def smoke_teacher(device, train_loader, test_loader, teacher_ckpt: Path):
    print("\n========= SEArch (teacher) smoke =========")
    student = ResNetCifar(depth=20, num_classes=10, base_width=16).to(device)
    teacher = ResNetCifar(depth=56, num_classes=10, base_width=16).to(device)
    load_model_weights(teacher, str(teacher_ckpt))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    cfg = {
        "searh": {
            "enabled": True,
            "selector": "teacher",
            "param_budget_factor": 1.5,
            "epochs_per_stage": 1,
            "B_op": 7,
            "deepen_first": True,
            "gamma": 0.5,
            "d_k": 32,
            "lambda_init": 1.0,
            "score_batches": 1,
            "final_retrain_epochs": 0,
        },
    }
    run_searh(
        cfg=cfg, student=student, optimizer=opt, teacher=teacher, device=device,
        train_loader=train_loader, test_loader=test_loader,
        experiment_name="smoke_searh_teacher", log_csv=None,
        run_ts="smoke", mlog_path=None,
    )


def smoke_critic(device, train_loader, test_loader, *, use_probe: bool = False):
    print(f"\n========= CGSE-on-SEArch (critic) smoke (probe={use_probe}) =========")
    student = ResNetCifar(depth=20, num_classes=10, base_width=16).to(device)
    opt = torch.optim.SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    from critics.student_probe import PROBE_DIM
    local_dim = 5 + (PROBE_DIM if use_probe else 0)
    critic = PerCandidateCritic(state_dim=STATE_DIM, local_dim=local_dim, hidden_dim=32).to(device)
    crit_opt = torch.optim.Adam(critic.parameters(), lr=0.01)
    cfg = {
        "searh": {
            "enabled": True,
            "selector": "critic",
            "param_budget_factor": 1.5,
            "epochs_per_stage": 1,
            "B_op": 7,
            "deepen_first": False,
            "epsilon": 0.30,
            "entropy_beta": 0.01,
            "final_retrain_epochs": 0,
            "max_epochs_hint": 4,
            "use_student_probe": use_probe,
            "baseline_momentum": 0.9,
        },
    }
    run_searh(
        cfg=cfg, student=student, optimizer=opt, teacher=None, device=device,
        train_loader=train_loader, test_loader=test_loader,
        experiment_name="smoke_searh_critic", log_csv=None,
        run_ts="smoke", mlog_path=None,
        critic=critic, critic_optimizer=crit_opt,
        build_critic_state_fn=build_critic_state,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", default="checkpoints/tier2/tier2_teacher_resnet56_cifar10_seed42.pt")
    p.add_argument("--skip_teacher", action="store_true")
    p.add_argument("--skip_critic", action="store_true")
    args = p.parse_args()

    device = _device()
    print(f"[smoke] device={device}")
    train_loader, test_loader = _make_loaders()
    print(f"[smoke] train batches={len(train_loader)} test batches={len(test_loader)}")

    if not args.skip_teacher:
        ckpt = Path(args.teacher_ckpt)
        if not ckpt.exists():
            print(f"[smoke] WARNING: {ckpt} not found, skipping teacher arm.")
        else:
            smoke_teacher(device, train_loader, test_loader, ckpt)

    if not args.skip_critic:
        smoke_critic(device, train_loader, test_loader, use_probe=False)
        smoke_critic(device, train_loader, test_loader, use_probe=True)

    print("\n[smoke] OK.")


if __name__ == "__main__":
    main()
