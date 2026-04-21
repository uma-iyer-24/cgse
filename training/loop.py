from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_distillation_loss(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    """KL(student || teacher) on softened distributions, scaled by T^2 (standard KD)."""
    t = float(temperature)
    s_logp = F.log_softmax(student_logits / t, dim=-1)
    p_t = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(s_logp, p_t, reduction="batchmean") * (t * t)


def train_one_epoch(
    model,
    optimizer,
    device,
    loader,
    teacher: Optional[nn.Module] = None,
    kd_temperature: float = 4.0,
    kd_alpha: float = 0.5,
    kd_teacher_every_n_steps: int = 1,
    kd_max_teacher_forwards: int | None = None,
    teacher_forwards_so_far: int = 0,
    *,
    return_stats: bool = False,
):
    """
    If `teacher` is set, loss = (1 - kd_alpha) * CE + kd_alpha * KD (student logits vs frozen teacher).
    """
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
        use_teacher = False
        if teacher is not None:
            every = max(1, int(kd_teacher_every_n_steps))
            if (train_steps % every) == 0:
                if kd_max_teacher_forwards is None:
                    use_teacher = True
                else:
                    budget_left = int(kd_max_teacher_forwards) - int(teacher_forwards_so_far) - int(
                        teacher_forwards
                    )
                    use_teacher = budget_left > 0

        if use_teacher:
            with torch.no_grad():
                t_logits = teacher(x)
            teacher_forwards += 1
            kd = kd_distillation_loss(logits, t_logits, kd_temperature)
            alpha = float(kd_alpha)
            loss = (1.0 - alpha) * ce + alpha * kd
        else:
            loss = ce
        loss.backward()
        optimizer.step()
        train_steps += 1
        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    out = (total_loss / max(total, 1), correct / max(total, 1))
    if not return_stats:
        return out
    return out[0], out[1], {"train_steps": train_steps, "teacher_forwards": teacher_forwards}


@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return total_loss / max(total, 1), correct / max(total, 1)
