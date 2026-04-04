"""Apply named structural ops for Tier 1b evolution (CifarGraphNet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.optim as optim

from ops.edge_split import edge_split
from ops.edge_widen import edge_widen
from ops.edge_widen_conv import edge_widen_conv3_cifar
from utils.model_info import count_trainable_parameters, linear_layer_shapes
from utils.mutation_log import append_mutation_jsonl
from utils.optimizer_utils import refresh_optimizer


def apply_named_mutation(
    model,
    optimizer: optim.Optimizer,
    op: str,
    *,
    widen_delta: int = 32,
    conv_delta: int = 32,
    mutation_log_jsonl: Path | None,
    experiment_name: str,
    run_ts: str,
    global_epoch: int,
    gate_tag: str,
) -> optim.Optimizer:
    """
    Dispatch Tier 1b ops on CifarGraphNet. Returns refreshed optimizer.
    """
    params_before = count_trainable_parameters(model)
    meta: dict[str, Any] = {
        "event": "mutation",
        "gate": gate_tag,
        "run_id": f"{experiment_name}_{run_ts}",
        "experiment": experiment_name,
        "epoch_completed": global_epoch,
        "num_parameters_before": params_before,
    }

    if op == "widen_fc1":
        target_id = "fc1"
        in_f, out_before = linear_layer_shapes(model, target_id)
        edge_widen(model, target_node_id=target_id, delta=widen_delta)
        optimizer = refresh_optimizer(optimizer, model)
        _, out_after = linear_layer_shapes(model, target_id)
        meta.update(
            {
                "op": "edge_widen",
                "target_node_id": target_id,
                "delta": widen_delta,
                "target_linear_in": in_f,
                "target_linear_out_before": out_before,
                "target_linear_out_after": out_after,
            }
        )
    elif op == "split_before_fc1":
        target_id = "fc1"
        edge_split(model, target_node_id=target_id)
        optimizer = refresh_optimizer(optimizer, model)
        meta.update({"op": "edge_split", "target_node_id": target_id})
    elif op == "split_before_fc2":
        target_id = "fc2"
        edge_split(model, target_node_id=target_id)
        optimizer = refresh_optimizer(optimizer, model)
        meta.update({"op": "edge_split", "target_node_id": target_id})
    elif op == "widen_conv3":
        edge_widen_conv3_cifar(model, delta=conv_delta)
        optimizer = refresh_optimizer(optimizer, model)
        meta.update({"op": "edge_widen_conv3", "delta": conv_delta})
    else:
        raise ValueError(f"unknown evolution op: {op}")

    params_after = count_trainable_parameters(model)
    meta["num_parameters_after"] = params_after
    print(
        f"[evolution] ({gate_tag}) {op} @ global_epoch {global_epoch}; "
        f"params {params_before} -> {params_after}"
    )
    if mutation_log_jsonl:
        append_mutation_jsonl(mutation_log_jsonl, meta)
    return optimizer


def _split_done(model, fc_id: str) -> bool:
    """Heuristic: edge_split inserts a node id containing fc_id and '_split_'."""
    for nid in model.execution_order:
        if fc_id in nid and "_split_" in nid:
            return True
    return False


def filter_legal_candidates(
    model, candidate_names: list[str], applied: set[str]
) -> list[str]:
    leg: list[str] = []
    for name in candidate_names:
        if name in applied:
            continue
        if name == "split_before_fc1" and _split_done(model, "fc1"):
            continue
        if name == "split_before_fc2" and _split_done(model, "fc2"):
            continue
        leg.append(name)
    return leg
