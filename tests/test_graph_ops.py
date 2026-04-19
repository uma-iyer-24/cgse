"""Graph mutations, KD, and checkpoint helpers (see paper_documentation/SEArch-baseline-and-CGSE-evaluation-plan.md §4.1)."""

import os
import tempfile

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from critics import STATE_DIM, StructuralCritic, build_critic_state
from models.cifar_student import CifarGraphNet
from ops.edge_split import edge_split
from ops.edge_widen import edge_widen
from ops.edge_widen_conv import edge_widen_conv3_cifar
from ops.resnet_head_widen import widen_resnet_head
from training.loop import kd_distillation_loss, train_one_epoch
from utils.artifact_families import (
    canonicalize_runs_artifact,
    infer_artifact_family,
    student_checkpoint_path,
)
from utils.checkpoint import load_model_weights, save_checkpoint


def _test_devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devs.append(torch.device("mps"))
    return devs


def test_kd_distillation_loss_matches_torch_kl():
    T = 4.0
    student = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    got = kd_distillation_loss(student, teacher, T)
    s_logp = F.log_softmax(student / T, dim=-1)
    p_t = F.softmax(teacher / T, dim=-1)
    want = F.kl_div(s_logp, p_t, reduction="batchmean") * (T * T)
    assert torch.allclose(got, want)


def test_checkpoint_roundtrip_cifar_graph_net():
    model = CifarGraphNet(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    try:
        save_checkpoint(model, optimizer, path)
        restored = CifarGraphNet(num_classes=10)
        load_model_weights(restored, path)
        for (a, b) in zip(model.state_dict().values(), restored.state_dict().values()):
            assert torch.equal(a, b)
    finally:
        os.unlink(path)


@pytest.mark.parametrize("device", _test_devices())
def test_edge_widen_keeps_device_and_forward(device):
    m = CifarGraphNet(num_classes=10).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    _ = m(x)
    edge_widen(m, target_node_id="fc1", delta=16)
    for p in m.parameters():
        assert p.device.type == device.type
    y = m(x)
    assert y.shape == (2, 10)


@pytest.mark.parametrize("device", _test_devices())
def test_edge_split_keeps_device_and_near_identity_output(device):
    m = CifarGraphNet(num_classes=10).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        y0 = m(x)
    edge_split(m, target_node_id="fc1")
    for p in m.parameters():
        assert p.device.type == device.type
    with torch.no_grad():
        y1 = m(x)
    assert y1.shape == y0.shape
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-4)


def test_train_one_epoch_with_teacher_smoke():
    device = torch.device("cpu")
    model = CifarGraphNet(10).to(device)
    teacher = CifarGraphNet(10).to(device)
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    loss, acc = train_one_epoch(
        model,
        opt,
        device,
        loader,
        teacher=teacher,
        kd_temperature=4.0,
        kd_alpha=0.5,
    )
    assert loss == loss and 0.0 <= acc <= 1.0


def test_edge_widen_conv3_cifar_shapes():
    device = torch.device("cpu")
    m = CifarGraphNet(num_classes=10).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    y0 = m(x)
    edge_widen_conv3_cifar(m, delta=16)
    y1 = m(x)
    assert y0.shape == y1.shape == (2, 10)


def test_split_before_fc1_then_widen_conv3_forward():
    """split inserts square Linear on old flatten dim; conv widen must resize it."""
    device = torch.device("cpu")
    m = CifarGraphNet(num_classes=10).to(device)
    edge_split(m, target_node_id="fc1")
    edge_widen_conv3_cifar(m, delta=16)
    x = torch.randn(2, 3, 32, 32, device=device)
    y = m(x)
    assert y.shape == (2, 10)


@pytest.mark.parametrize("device", _test_devices())
def test_resnet_head_widen_preserves_logits_nearly(device):
    from models.resnet_cifar import ResNetCifar

    m = ResNetCifar(depth=20, num_classes=10).to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        y0 = m(x)
    widen_resnet_head(m, hidden_delta=32)
    with torch.no_grad():
        y1 = m(x)
    # Net2Net init should preserve outputs up to floating error
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_split_before_fc2_forward():
    """Identity block before logits; output shape unchanged."""
    device = torch.device("cpu")
    m = CifarGraphNet(num_classes=10).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    y0 = m(x)
    edge_split(m, target_node_id="fc2")
    y1 = m(x)
    assert y0.shape == y1.shape == (2, 10)


def test_infer_artifact_family_buckets():
    assert infer_artifact_family("cgse_phase2_cifar_full_seed41") == "tier1"
    assert infer_artifact_family("cgse_evolution_tier1b_schedule") == "tier1b"
    assert infer_artifact_family("cgse_phase2_cifar_cgse_smoke") == "smoke"
    assert infer_artifact_family("cgse_phase0") == "other"
    p = student_checkpoint_path("cgse_phase2_cifar_full")
    assert p.parts[:3] == ("checkpoints", "tier1", "cgse_phase2_cifar_full.pt")


def test_canonicalize_runs_artifact():
    exp = "cgse_phase2_cifar_full_seed43"
    assert canonicalize_runs_artifact("runs/foo.csv", exp, "metrics") == "runs/tier1/metrics/foo.csv"
    assert canonicalize_runs_artifact("runs/metrics/foo.csv", exp, "metrics") == "runs/tier1/metrics/foo.csv"
    assert (
        canonicalize_runs_artifact("runs/tier1/metrics/foo.csv", exp, "metrics")
        == "runs/tier1/metrics/foo.csv"
    )
    assert canonicalize_runs_artifact("/tmp/x.csv", exp, "metrics") == "/tmp/x.csv"
    assert canonicalize_runs_artifact(None, exp, "metrics") is None


def test_build_critic_state_shape_and_critic_forward():
    device = torch.device("cpu")
    s = build_critic_state(
        train_loss=1.5,
        train_acc=0.4,
        val_loss=1.6,
        val_acc=0.35,
        epoch=3,
        max_epochs=50,
        num_params=620_810,
        prev_train_loss=1.6,
        prev_val_acc=0.33,
        anchor_train_loss=2.0,
        device=device,
    )
    assert s.shape == (1, STATE_DIM)
    c = StructuralCritic(state_dim=STATE_DIM, hidden_dim=8)
    out = c(s)
    assert out.shape == (1,)
