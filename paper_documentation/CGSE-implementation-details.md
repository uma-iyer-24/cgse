# CGSE — Implementation details (SEArch + CGSE-on-SEArch)

**Purpose.** Deep technical walkthrough of the paper-faithful SEArch reproduction and the CGSE-on-SEArch system that was built on top of it. This doc explains *how* each component works inside — the algorithms, the data flow, the function-preserving initialization tricks, the optimizer-state plumbing across mutations, the REINFORCE update, the student probe internals, the logging schemas, and the extension points.

**Audience.** Anyone reading the code who wants the *mechanism*, not just the file map. If you want a chronological log, see [`CGSE-implementation-log.md`](CGSE-implementation-log.md). If you want a file-by-file index, see [`CGSE-codebase-guide.md`](CGSE-codebase-guide.md). If you want the equations the paper will print, see [`CGSE-math-and-equations.md`](CGSE-math-and-equations.md).

**Scope.** This document focuses on the SEArch + CGSE-on-SEArch implementation introduced in late April 2026 (Tier 2 / Tier 3 stack on ResNet-CIFAR). The earlier `CifarGraphNet` work (Tier 1 / Tier 1b) is covered in [`CGSE-detailed-phase-walkthrough.md`](CGSE-detailed-phase-walkthrough.md).

---

## Table of contents

1. [High-level architecture and control flow](#1-high-level-architecture-and-control-flow)
2. [Student and teacher backbones](#2-student-and-teacher-backbones)
3. [Channel-attention KD (the SEArch loss)](#3-channel-attention-kd-the-searh-loss)
4. [Stage-output node correspondence](#4-stage-output-node-correspondence)
5. [Modification value scorer](#5-modification-value-scorer)
6. [Candidate enumeration](#6-candidate-enumeration)
7. [Edge-splitting operators (deepen, widen)](#7-edge-splitting-operators-deepen-widen)
8. [The unified outer loop (`run_searh`)](#8-the-unified-outer-loop-run_searh)
9. [The CGSE per-candidate critic](#9-the-cgse-per-candidate-critic)
10. [The student probe](#10-the-student-probe)
11. [REINFORCE with EMA baseline](#11-reinforce-with-ema-baseline)
12. [Function-preserving initialization in detail](#12-function-preserving-initialization-in-detail)
13. [Optimizer state preservation across mutations](#13-optimizer-state-preservation-across-mutations)
14. [Configuration system (YAML → code path)](#14-configuration-system-yaml--code-path)
15. [Logging schemas (CSV and JSONL)](#15-logging-schemas-csv-and-jsonl)
16. [Smoke test expectations](#16-smoke-test-expectations)
17. [Reproducibility and determinism](#17-reproducibility-and-determinism)
18. [Performance characteristics](#18-performance-characteristics)
19. [Common gotchas](#19-common-gotchas)
20. [Extension points](#20-extension-points)

---

## 1. High-level architecture and control flow

When `train.py` parses a config with `searh.enabled: true` it dispatches to `training.searh_loop.run_searh`, bypassing the legacy mutation/critic branches entirely. The same function services both arms — the only branch is on `searh.selector` (`"teacher"` or `"critic"`). At a glance:

```text
train.py
  └─ if cfg["searh"]["enabled"]:
        ├─ if selector == "critic":
        │     instantiate PerCandidateCritic + Adam optimizer
        ├─ run_searh(
        │     cfg, student, optimizer, teacher, train_loader, test_loader,
        │     log_csv, mlog_path, critic, critic_optimizer, build_critic_state_fn)
        ↓
training/searh_loop.py::run_searh
  ├─ build channel-attention KD if selector=teacher
  ├─ attach student probe if selector=critic and use_student_probe
  ├─ initial param-budget cap = ⌊1.5 × |θ_0|⌋
  └─ stage loop (until budget hit OR no candidates):
       ├─ for ep in range(epochs_per_stage):
       │     train one epoch — CE + (λ * L_im if teacher arm)
       │     evaluate on test set, append metrics CSV
       ├─ if pending CGSE PG update from prior stage:
       │     advantage = R - baseline; update critic via REINFORCE + entropy bonus
       ├─ enumerate candidates(B_op cap, deepen-first heuristic)
       ├─ if candidates empty OR params >= cap: break / continue
       ├─ score candidates:
       │     teacher arm  → MV(n) = D(n) · deg+/deg- (Eq. 9)
       │     critic arm   → π_ψ(s ⊕ local_descriptor) softmax-sample
       ├─ apply chosen op (deepen or widen), refresh optimizer
       └─ if critic arm: stash (state, choice, val_at_mutation) for next-stage PG
  └─ final retrain (T_retrain epochs, λ=0, no mutations)
  └─ detach KD hooks, detach probe, write final metrics row
```

Critical control-flow guarantees:

- `run_searh` is **idempotent in its first epoch** — the param budget is only checked *after* the first stage runs at least once, so no zero-mutation runs.
- The CGSE policy update at line "PG update for any pending mutation from prior stage" runs **after** the stage has trained and *before* the next stage's mutation. This means the reward `R = val_acc(t) − val_acc(t-1)` is computed against the validation accuracy *after* the post-mutation training stage, not the value immediately after the structural edit. That delay is intentional: it gives the network time to react to the mutation before being judged.
- The **same** input tensor is rebuilt for the PG update (lines 472–484 in `training/searh_loop.py`) as was used for selector sampling. This preserves the action-distribution gradient identity through the deferred update.

---

## 2. Student and teacher backbones

Both the student (ResNet-20) and the teacher (ResNet-56) are CIFAR-style ResNets defined in `models/resnet_cifar.py`. The structure is canonical:

```text
input (3, 32, 32)
  → Conv3x3(3 → 16) → BN → ReLU                      # stem
  → layer1: nn.Sequential[BasicBlock × N1]            # 16 channels, 32×32
  → layer2: nn.Sequential[BasicBlock × N2]            # 32 channels, 16×16 (first block: stride 2)
  → layer3: nn.Sequential[BasicBlock × N3]            # 64 channels, 8×8   (first block: stride 2)
  → AdaptiveAvgPool2d(1) → Flatten
  → Linear(64 → num_classes)
```

For ResNet-20, `(N1, N2, N3) = (3, 3, 3)`. For ResNet-56, `(9, 9, 9)`. Each `BasicBlock` is the standard

```text
x → Conv3x3(c → c) → BN → ReLU → Conv3x3(c → c) → BN → (+ shortcut) → ReLU
```

with the shortcut being identity unless `stride > 1` or channels grow, in which case a 1×1 conv with BN is used.

**Why this matters for the SEArch loop:**

- The **stage-output shapes** `(16, 32, 32) / (32, 16, 16) / (64, 8, 8)` are identical between student and teacher. That is the precondition for the channel-attention KD module to match student stage `i` with teacher stage `i` directly — see [§4](#4-stage-output-node-correspondence).
- Mutations *inside* a stage (deepen or widen) **preserve the stage's I/O shape**. Both `DeepenBlock` and `WidenedBlock` are channel-preserving by construction. As a consequence, attention-KD hooks remain valid across mutations without any rebuild.
- The first block of `layer2` and `layer3` (the stride-2 block with channel doubling) is **never a widen target** — see [§7](#7-edge-splitting-operators-deepen-widen) for why and how the candidate enumerator filters this case.

---

## 3. Channel-attention KD (the SEArch loss)

`training/searh_attention.py` implements the paper's Eqs. 1–5: a learned channel-space attention that re-projects teacher feature maps into the student's channel space and then takes the squared-L2 distance.

### 3.1 The single-pair head: `ChannelAttentionKD`

For one paired (student node, teacher node) with channel widths $C_s, C_t$:

```python
def forward(f_s, f_t):                     # both (B, C, H, W), shapes match in B/H/W
    d_s = f_s.mean((2,3), keepdim=False).unsqueeze(-1)   # (B, Cs, 1)  channel descriptor
    d_t = f_t.mean((2,3), keepdim=False).unsqueeze(-1)   # (B, Ct, 1)
    Q = self.q(d_s)                          # (B, Cs, d_k)      W_Q
    K = self.k(d_t)                          # (B, Ct, d_k)      W_K
    attn = einsum("bsd,btd->bst", Q, K) * d_k**-0.5
    attn = softmax(attn, dim=-1)             # (B, Cs, Ct), per student channel
    f_t_flat = f_t.view(B, Ct, H*W)
    f_proj   = einsum("bst,btp->bsp", attn, f_t_flat).view(B, Cs, H, W)
    return (f_s - f_proj).pow(2).mean()      # scalar — note: mean, not sum!
```

**Two implementation choices worth flagging:**

1. **Channel descriptors are global-average-pooled scalars per channel**, not full $H\times W$ feature maps. This matches the paper's "for every channel in the student's feature map, an attention query is used to calculate the corresponding channel weights in the teacher's feature map" (§3.2). Computationally it means `Q, K` are tiny `Linear(1, d_k)` maps applied to the pooled descriptors, not on the spatial grid.
2. **Output is `.mean()`, not `.sum(dim=1).mean()`.** The paper's Eq. 2 writes a squared L2 per node. We average over *all* dims (batch, channels, spatial) so the imitation loss is $O(1)$ regardless of feature-map size — otherwise stage 3 (64 channels × 8 × 8 = 4,096 elements) would dominate stage 1 (16 × 32 × 32 = 16,384 elements with smaller channel count) and the cosine-annealed $\lambda$ in Eq. 6 would need per-stage rescaling. **The smoke test caught this** — initial sum-over-channels loss was ~159,000, swamping CE (~1.5).

### 3.2 The multi-pair holder: `MultiNodeAttentionKD`

Holds one `ChannelAttentionKD` head per `(student_node, teacher_node)` pair, registered as forward hooks so the student and teacher only need a single forward call per batch:

- `attach()` registers hooks on the matched modules; outputs are cached in `self._student_feats / self._teacher_feats`.
- The first forward of training calls `_ensure_heads(device)` which lazily instantiates each `ChannelAttentionKD` with the actual channel widths it just observed (avoids hard-coding teacher/student channel widths in `__init__`).
- `imitation_loss()` runs a forward pass on the cached features through every head and returns the mean of per-pair distances (Eq. 3).
- `detach()` cleans up hooks before final save.

**Hook-lifecycle gotcha.** The hooks read `module.forward` outputs and cache them as detached tensors. If you ever forget to call `detach()` at the end of training the hooks survive across runs and cause memory leaks in long jobs. `run_searh` calls it unconditionally in its `finally`-equivalent block.

### 3.3 The training step under the SEArch loss

In `_train_one_epoch_with_attn`:

```python
logits = student(x)                    # populates KD's student-feat cache via hooks
ce = F.cross_entropy(logits, y)
loss = ce
if teacher is not None and lambda_im > 0:
    with torch.no_grad():
        _ = teacher(x)                 # populates KD's teacher-feat cache
    teacher_forwards += 1
    L_im = attn_kd.imitation_loss()    # Eq. 3
    loss = ce + λ * L_im               # Eq. 4
loss.backward()
optimizer.step()
```

Two important details:

- The teacher forward is `no_grad`. Teacher parameters never see a gradient.
- The attention KD module's `q` / `k` linear maps are part of the *optimizer*, not the teacher. They are trained alongside the student. `_ParamWrapper` (bottom of `searh_loop.py`) is the adapter that lets `refresh_optimizer` collect parameters from `student + attn_kd` jointly after every mutation.

---

## 4. Stage-output node correspondence

`training/searh_node_map.py::build_node_map` returns three pairs:

```python
("stage1", "stage1"),  # → student.layer1, teacher.layer1   (16ch, 32×32)
("stage2", "stage2"),  # → student.layer2, teacher.layer2   (32ch, 16×16)
("stage3", "stage3"),  # → student.layer3, teacher.layer3   (64ch, 8×8)
```

The student/teacher lookup tables map these IDs to actual `nn.Module` references (the `nn.Sequential` of basic blocks for each layer). Hooks attach to those sequential containers; their *output* is the post-stage feature map — the natural "node" identity in the paper's sense.

**Why stage outputs and not individual blocks?** Inside a stage, mutations happen freely (deepen appends a `DeepenBlock`, widen wraps a `BasicBlock`). The stage *as a whole* keeps the same I/O shape. So the hook attached to `student.layer2` continues to fire correctly whether `layer2` is `[BB, BB, BB]` or `[BB, BB, BB, DB, DB, WB]`. If we'd hooked individual blocks instead, every mutation would invalidate the hook map.

The `gamma` hyperparameter in `searh_node_map.build_node_map` is reserved for a future intra-stage Eq. 6 selection rule (which block within a stage to map to which) but is currently unused — stage-to-stage correspondence is exact.

---

## 5. Modification value scorer

`evolution/searh_mv.py` is where the paper's Eq. 5 lives:

$$\text{MV}(n) = D(n)\cdot\frac{\deg^+(n)}{\deg^-(n)}.$$

For our linear ResNet stack, every internal node has $\deg^+ = \deg^- = 1$, so the multiplicative correction collapses to 1 and `MV(n) = D(n)`. We keep the structure of Eq. 5 intact (`deg_plus_minus()` returns `(1, 1)`) so that a future DAG-based student drops in without an API change.

`compute_per_node_distances` averages $D(n)$ over `score_batches` (default 2) batches drawn from the test loader, in `no_grad` mode. The averaged distances are passed through `modification_values` (Eq. 5) and then `rank_candidates` produces a descending `(Candidate, mv)` list. The teacher-MV selector simply takes the top of this list.

---

## 6. Candidate enumeration

`evolution/candidates.py::enumerate_candidates` produces the legal `(stage, op)` action set at the current architecture, with two filters:

| Filter | Rule | Why |
|---|---|---|
| **B_op cap on deepens** | `deepen` is offered only while `count_deepens_in_stage(model, stage) < b_op_cap` | Matches the paper's bounded-stack-depth rule. Beyond `B_op = 7` the loop must switch to widen. |
| **Stride/channel safety on widens** | `widen` is offered only when the stage contains at least one `BasicBlock` with `stride == 1 AND in_channels == out_channels` | The first block of `layer2` / `layer3` has stride 2 and doubled channels. Wrapping it with a same-channel parallel branch would mismatch shapes. The **smoke test caught this** — `RuntimeError: Given groups=64, weight of size [64, 1, 3, 3], expected input[64, 32, 16, 16]` — and `_is_widenable_basic_block` was added as the safety filter. |
| **Deepen-first heuristic** | When `deepen_first=True`, `widen` is suppressed *until* the stage's deepen count hits `B_op` | Matches the paper's deepen-first preference (§3.4). The CGSE arm typically sets `deepen_first=False` so the critic chooses freely. |

The output is a list of `Candidate(stage, op, node_id)` rows where `node_id` matches the `searh_node_map` IDs (`"stage1" / "stage2" / "stage3"`) so MV scores can be looked up by name.

When the candidate list is empty (every stage has hit `B_op` deepens AND there are no widenable blocks left) the outer loop **terminates the search early**, before the param budget. This is the empirical termination condition for ResNet-20 — typically the budget hits first at $\rho = 1.5$.

---

## 7. Edge-splitting operators (deepen, widen)

### 7.1 `ops/searh_deepen.py::deepen_resnet_stage`

```python
def deepen_resnet_stage(model, *, stage, after_block_idx=None) -> int:
    layer = model.layer{stage}                    # nn.Sequential[BasicBlock or DeepenBlock or WidenedBlock]
    if after_block_idx is None:
        after_block_idx = len(blocks) - 1         # default: append at end of stage
    base = blocks[after_block_idx]
    ch = infer_channels(base)                     # robust to base being BB / DB / WB
    new_block = DeepenBlock(ch, init_zero=True)   # SepConv(x) ≡ 0 at init → identity
    blocks.insert(after_block_idx + 1, new_block)
    setattr(model, f"layer{stage}", nn.Sequential(*blocks))
    return len(blocks)                            # new total count for this stage
```

`DeepenBlock` is a residual sep-conv block (depthwise + pointwise 3×3, two BNs, residual added with ReLU). With `init_zero=True` the *last* BN's `gamma` is zeroed, so `SepConv(x) ≡ 0` at insert time and the block reduces to `ReLU(x + 0) = ReLU(x) = x` for the post-ReLU input it sees from the previous block — function-preserving. See [§12](#12-function-preserving-initialization-in-detail) for why this matters.

### 7.2 `ops/searh_widen.py::widen_resnet_stage`

```python
def widen_resnet_stage(model, *, stage, block_idx=None) -> str:
    layer = model.layer{stage}
    if block_idx is None:
        # Pick the LAST stride-1 same-channel BasicBlock that exists (paper's
        # "edge at end of stage" semantics). Filtered by _is_widenable_basic_block.
        candidates = [i for i, b in enumerate(blocks)
                      if isinstance(b, BasicBlock)
                      and b.conv1.in_channels == b.bn2.num_features
                      and stride_of(b) == 1]
        if not candidates:
            raise ValueError(f"layer{stage} has no widenable BasicBlock")
        block_idx = candidates[-1]
    base = blocks[block_idx]
    ch = infer_channels(base)
    wrapped = WidenedBlock(base, ch)              # parallel sep-conv branch, init_zero=True
    blocks[block_idx] = wrapped
    setattr(model, f"layer{stage}", nn.Sequential(*blocks))
    return wrapped_kind
```

`WidenedBlock(base, ch)` wraps `base` and adds one parallel `SepConv3x3(ch, init_zero=True)` whose output is summed into `base(x)` before returning. Identity-init of the branch's last BN means `branch(x) ≡ 0` at insert and the wrapped block's behaviour is exactly preserved.

A `WidenedBlock` can itself be re-widened (the paper does not forbid repeated widens), and in fact our enumerator allows it provided the inner channel-equality / stride-1 invariants hold — see the `WidenedBlock` branch in `widen_resnet_stage`'s channel-inference code.

---

## 8. The unified outer loop (`run_searh`)

`training/searh_loop.py::run_searh` is ~370 lines and implements the full Algorithm 1 for both arms in one function. The skeleton:

```python
def run_searh(*, cfg, student, optimizer, teacher, device, train_loader, test_loader,
              experiment_name, log_csv, run_ts, mlog_path,
              critic=None, critic_optimizer=None, build_critic_state_fn=None):
    # 1. Read config, decide arm, set up budget cap.
    selector = cfg["searh"]["selector"].lower()      # "teacher" or "critic"
    epochs_per_stage = cfg["searh"]["epochs_per_stage"]
    b_op_cap = cfg["searh"]["B_op"]
    cap = int(cfg["searh"]["param_budget_factor"] * count_trainable_parameters(student))

    # 2. (teacher arm only) build attention KD, attach hooks, lazy-build heads.
    if selector == "teacher":
        node_map = build_node_map(student, teacher)
        attn_kd = MultiNodeAttentionKD(student, teacher, node_map.pairs, ...)
        attn_kd.attach()
        with torch.no_grad():                        # one tiny probe forward
            x_probe, _ = next(iter(train_loader))
            student(x_probe[:2]); teacher(x_probe[:2])
            attn_kd._ensure_heads(device)
        optimizer = refresh_optimizer(optimizer, _ParamWrapper(student, attn_kd))

    # 3. (critic arm only) attach student probe, snapshot baseline weights.
    if selector == "critic" and cfg["searh"]["use_student_probe"]:
        probe = StudentProbe()
        probe.attach(student)
        probe.snapshot_all_stages(student)

    pending_pg = None       # critic arm: deferred PG update
    baseline_value = 0.0    # critic arm: EMA reward baseline
    stage_idx = 0

    while True:
        # 4. Termination check (after at least one stage trained).
        if count_trainable_parameters(student) >= cap and stage_idx > 0:
            break
        stage_idx += 1

        # 5. Train this stage (epochs_per_stage epochs, cosine λ-anneal teacher arm).
        for ep in range(epochs_per_stage):
            t_frac = ep / max(epochs_per_stage - 1, 1)
            λ = 0.5 * lambda_init * (1 + cos(pi * t_frac)) if selector == "teacher" else 0.0
            train_one_epoch_with_attn(student, optimizer, ..., teacher, attn_kd, λ)
            evaluate_and_log()

        # 6. (critic arm) PG update for the *previous* stage's mutation.
        if pending_pg is not None and selector == "critic":
            R = current_val_acc - pending_pg["val_at_mutation"]
            advantage = R - baseline_value
            baseline_value = baseline_momentum * baseline_value + (1 - baseline_momentum) * R
            scores = critic(pending_pg["inp"]).squeeze(-1)
            log_p = log_softmax(scores, dim=0)
            entropy = -(softmax(scores, dim=0) * log_p).sum()
            loss_pg = -log_p[pending_pg["choice"]] * advantage - entropy_beta * entropy
            critic_optimizer.zero_grad(); loss_pg.backward(); critic_optimizer.step()
            pending_pg = None

        # 7. Enumerate candidates, score, choose.
        candidates = enumerate_candidates(student, b_op_cap=b_op_cap, deepen_first=deepen_first)
        if not candidates: break
        if selector == "teacher":
            chosen = teacher_mv_selector(student, teacher, attn_kd, ...)
        else:
            probe_feats = collect_probe_features_if_enabled()
            chosen, choice_idx, _explored = critic_mv_selector(
                student, critic, build_critic_state_fn(...), candidates,
                b_op_cap=b_op_cap, epsilon=epsilon, sample=True,
                probe_features_per_stage=probe_feats)

        # 8. Apply mutation, refresh optimizer.
        if chosen.op == "deepen":
            deepen_resnet_stage(student, stage=chosen.stage, after_block_idx=None)
        else:
            widen_resnet_stage(student, stage=chosen.stage, block_idx=None)
        optimizer = refresh_optimizer(optimizer, _ParamWrapper(student, attn_kd) if attn_kd else student)
        if probe is not None:
            probe.snapshot_stage(student, chosen.node_id)   # reset Δw baseline for this stage
        append_mutation_jsonl(mlog_path, {...})

        # 9. Defer PG update to the *next* stage's end (so reward = post-training val).
        if selector == "critic":
            pending_pg = build_pg_payload(state, candidates, choice_idx, prev_val_acc)

    # 10. Final retrain (optional, λ=0, no mutations).
    for k in range(final_retrain_epochs):
        train_one_epoch_with_attn(student, optimizer, ..., teacher, attn_kd, λ=0.0)
        evaluate_and_log()

    if attn_kd: attn_kd.detach()
    if probe:   probe.detach()
```

Two subtle points worth zooming in on:

### 8.1 Why the PG update is deferred one stage

When we mutate at the end of stage $k$, the network *just* changed. Its validation accuracy at that moment is meaningless as a reward — it reflects pre-mutation training. We need the reward to capture *what the mutation bought us after the network had a chance to react*. So we stash the (state, choice, val) at mutation time, train the *next* stage (which is `epochs_per_stage` more SGD steps with the new architecture), and only *then* compute the reward and update the critic. This is what `pending_pg` carries between iterations of the `while`-loop.

### 8.2 Cosine $\lambda$-anneal is per-stage, not global

Inside each stage we anneal $\lambda$ from `lambda_init` down to 0 across the `epochs_per_stage` epochs:

```python
λ = 0.5 * lambda_init * (1 + cos(pi * (ep / (epochs_per_stage - 1))))
```

This matches the paper's intuition that the imitation loss is *most* useful early in a stage (when the post-mutation network is still reorganizing) and least useful late in the stage (when CE dominates and the imitation loss is fighting the labels). Notice that the anneal *resets* at each stage — the new architectural state benefits from another phase of strong teacher guidance.

---

## 9. The CGSE per-candidate critic

`critics/searh_critic.py::PerCandidateCritic` is a tiny MLP:

```python
class PerCandidateCritic(nn.Module):
    def __init__(self, state_dim, local_dim, hidden_dim=64):
        super().__init__()
        self.in_dim = state_dim + local_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):              # x: (K candidates, state_dim + local_dim)
        return self.net(x)             # (K, 1) logits
```

Each candidate's input row is `[global_state ⊕ local_descriptor]`. The critic produces one scalar score per candidate; softmax over candidates gives the action policy.

### 9.1 Global state (`critics/state_features.py::build_critic_state`, `STATE_DIM = 8`)

A hand-crafted, gradient-free, roughly-normalized-to-`[-1, 1]` vector:

| Index | Feature | Definition | Range |
|---|---|---|---|
| 0 | `tl_n` | normalized current train loss | `clip(train_loss / anchor_loss, [0, 3]) / 3` |
| 1 | `val_acc` | current validation accuracy | `[0, 1]` |
| 2 | `e_n` | normalized epoch | `epoch / max(max_epochs - 1, 1)` |
| 3 | `p_n` | normalized log-param-count | `clip(log2(num_params) / 22, [0, 1])` |
| 4 | `dva_n` | clipped Δval since prev epoch | `clip(val_acc - prev_val_acc, [-0.2, 0.2]) / 0.2` |
| 5 | `dtl_n` | clipped Δtrain-loss since prev epoch | `clip((tl - prev_tl) / anchor, [-1, 1])` |
| 6 | `vl_n` | normalized current val loss | `clip(val_loss / 5, [0, 2]) / 2` |
| 7 | `1.0` | bias term | const |

The choices are deliberately conservative (lots of clipping, lots of normalization) because the critic gets very sparse REINFORCE updates and needs an input scale that is comparable across runs.

### 9.2 Local descriptor (`_make_local_descriptor`)

For each candidate, a small per-candidate vector (5-dim base, 8-dim with probe):

| Index | Feature | Definition | Range |
|---|---|---|---|
| 0 | `is_stage1` | one-hot of `cand.stage == 1` | `{0, 1}` |
| 1 | `is_stage2` | one-hot of `cand.stage == 2` | `{0, 1}` |
| 2 | `is_stage3` | one-hot of `cand.stage == 3` | `{0, 1}` |
| 3 | `op_is_widen` | 1 if op is widen, 0 if deepen | `{0, 1}` |
| 4 | `deepens_norm` | `count_deepens_in_stage / B_op` | `[0, 1]` |
| 5–7 | (probe) | `[act_var_ratio, grad_l2_norm, weight_delta_norm]` | `[0, 1]³` |

The probe slots are present iff `searh.use_student_probe: true` AND `selector: critic`. The critic's `local_dim` is set in `train.py` based on the flag (`5 + PROBE_DIM` or just `5`) so the same checkpoint format works across ablation pairs.

### 9.3 Action selection

`_critic_mv_selector` does the following at each stage end:

```python
1. desc_rows[k] = _make_local_descriptor(cand_k, ...)
2. inp = concat(global_state.expand(K, -1), local_descs)        # (K, in_dim)
3. scores = critic(inp).squeeze(-1)                              # (K,)
4. probs  = softmax(scores.detach(), dim=0)
5. with prob ε:    choice = uniform([0, K-1])      # exploration
   else if sample: choice = multinomial(probs)     # stochastic policy
   else:           choice = argmax(scores)         # greedy
```

We **store the choice index** (not the chosen `Candidate` directly) so we can rebuild the same input tensor later for the deferred PG update. The PG-update tensor is rebuilt with `_make_local_descriptor` using the *post-mutation* number of deepens; this is intentional — the policy gradient is taken w.r.t. the same forward computation that produced the action probabilities.

---

## 10. The student probe

`critics/student_probe.py::StudentProbe` provides the critic with a *locality* signal — per-stage hints about *which stage is the bottleneck* — derived purely from the student's internal state. This is the unsupervised analogue of the teacher's $D(n)$.

### 10.1 The three features (`PROBE_DIM = 3`)

#### `act_var_ratio` — top-1 PC variance ratio of the channel covariance

Given a stage output `act` of shape `(B, C, H, W)`:

1. Reshape to `x ∈ R^{(B·H·W) × C}` and centre across the batch axis.
2. Compute trace$(\Sigma) = \frac{1}{n-1}\sum_i \|x_i\|^2$ (the total variance — sum of all eigenvalues).
3. Approximate $\lambda_{\max}(\Sigma)$ via 4 steps of power iteration on $\Sigma = \frac{1}{n-1} X^\top X$ — we never materialize $\Sigma$ explicitly, just compute $\Sigma v = \frac{1}{n-1}X^\top(Xv)$ which is $O(n\cdot c)$.
4. Return $\lambda_{\max} / \mathrm{trace}(\Sigma) \in [0, 1]$.

**Interpretation.** When a stage's representations collapse onto a single direction (a bottleneck — the stage is failing to spread information across its channels), this ratio jumps toward 1. When variance is well-distributed across channels (healthy), it stays near $1/C$. Empirically on ResNet-20 / CIFAR-10 the ratio sits around 0.04–0.08 for healthy stages and rises sharply when a stage saturates.

**Why power iteration and not `torch.linalg.eigh`?** For 64-channel feature maps `eigh` is ~10× slower than 4 power-iteration steps, and for a noisy REINFORCE input we don't need exact $\lambda_{\max}$. A 4-iteration approximation costs ~0.5 ms per stage on MPS.

#### `grad_l2` — gradient L2 norm per stage

After every backward pass during the score step, we walk every parameter of `model.layer{stage}` that has `requires_grad=True` and accumulate $\|\nabla_p \mathcal{L}\|_2^2$, then take the square root. This is *captured* (`update_grads`) at the moment the probe is queried, so it reflects the *most recent* backward pass — i.e., a real training step's gradient signal, not a synthetic probe-only gradient.

The three per-stage values are then **min-max normalized** across the three stages so the critic sees `(grad_l2_stage1 / max_grad_l2, grad_l2_stage2 / max_grad_l2, grad_l2_stage3 / max_grad_l2)` ∈ `[0, 1]³`. This makes the feature scale-invariant across runs, which matters because the absolute gradient norm depends on dataset size, batch size, and lr.

#### `weight_delta` — Frobenius distance since last mutation

We snapshot every parameter of `model.layer{stage}` at two times: (i) once at the very start of training (`snapshot_all_stages`) and (ii) every time a mutation lands in that stage (`snapshot_stage(model, stage_id)`). At probe time we compute $\|\theta_{\text{stage}}^{\text{now}} - \theta_{\text{stage}}^{\text{snap}}\|_F$.

**Mutation-aware snapshot reset.** When a mutation lands in stage $k$, we re-snapshot stage $k$ immediately afterwards (line 509 in `searh_loop.py`):

```python
if probe is not None:
    probe.snapshot_stage(student, chosen.node_id)
```

This means `weight_delta` for that stage collapses to 0 immediately after the mutation and grows again as the next stage's training drifts the weights. The signal answers *"how much has this stage actually adapted since I last touched it?"* — useful for the critic to decide whether a stage has converged or still has runway.

**Snapshot-reshape gotcha.** When deepen/widen lands in a stage, the parameter list changes shape (a new block's parameters are added). The probe handles this by `zip`-ing the old and new lists and slicing to the overlap region for any tensor whose shape grew — see `_weight_delta` lines 173–183. New parameters with no baseline contribute 0 delta until the next snapshot.

Like `grad_l2`, the per-stage `weight_delta` values are min-max normalized across stages so the critic sees a *relative* magnitude.

### 10.2 Probe lifecycle

```python
# At run_searh start:
probe = StudentProbe()
probe.attach(student)                  # registers forward hooks on layer1/layer2/layer3
probe.snapshot_all_stages(student)     # baseline weights for Δw

# At each stage's end (CGSE selection step):
probe.update_grads(student)            # capture gradient L2 from last backward
probe.run_forward(student, x_batch)    # one no_grad forward to populate activations
features = probe.per_stage_features(student)   # Dict[stage_id, [3 floats]]

# Right after a mutation lands in stage k:
probe.snapshot_stage(student, f"stage{k}")    # reset Δw baseline for that stage

# At run_searh end:
probe.detach()
```

The probe forward in `run_forward` uses `model.eval()` during the call, then restores the previous training state — so BatchNorm running stats are not perturbed by the probe-only forward.

---

## 11. REINFORCE with EMA baseline

The critic is trained via REINFORCE on the deferred reward:

$$R_{k} = \mathrm{val\_acc}_{\text{after stage } k+1} - \mathrm{val\_acc}_{\text{at mutation } k}.$$

That reward is sparse and very noisy (small architecture changes give Δval on the order of $10^{-3}$ – $10^{-2}$, dominated by SGD noise). To reduce gradient variance we subtract an exponential-moving-average baseline:

$$
A_k = R_k - b_{k-1},\qquad
b_k = \mu\, b_{k-1} + (1 - \mu)\, R_k,
$$

with $\mu = $ `baseline_momentum` (default 0.9). The baseline initialises to 0.

The PG update then minimizes

$$
\mathcal{L}_{\text{PG}} = -A_k\,\log\pi_\psi(c^*_k\mid s_k) - \beta_H \cdot H[\pi_\psi(\cdot\mid s_k)]
$$

(where $\beta_H$ = `entropy_beta`, default 0.01) using `critic_optimizer` (Adam by default, `lr = critic_lr = 0.01`).

### Disabling the baseline cleanly

Setting `baseline_momentum: 1.0` keeps `baseline_value = 1.0 * baseline_value + 0.0 * R = baseline_value` forever (pinned at its 0.0 init), so $A_k = R_k - 0 = R_k$ — that's mathematically raw REINFORCE. This is exactly how the Tier 3 `cgse_base` and `cgse_probe` configs disable variance reduction without needing a separate code path.

### Entropy bonus rationale

Without the entropy bonus the critic can collapse onto deterministic action selection within a few PG updates — particularly bad when one candidate has a slightly larger reward by chance and the policy never explores again. $\beta_H = 0.01$ keeps a soft floor on $H[\pi_\psi]$ across the candidate set without overwhelming the reward signal.

### The exploration-vs-policy split

`_critic_mv_selector` actually has *two* sources of stochasticity:

1. **ε-exploration** (`epsilon` param, default 0.30): with probability ε pick a uniformly random candidate. This always runs *first* — when it fires the policy's softmax is bypassed entirely.
2. **Stochastic policy sampling** (`sample=True`): if exploration didn't fire, sample from softmax(scores). With `sample=False` the critic runs greedy `argmax` instead.

Both modes still use the *policy's* `log_p[choice]` for the PG update (since the chosen action's log-probability is well-defined either way), but the exploration mechanism gives a guaranteed lower bound on the visit frequency of every candidate — the policy can never "freeze" some option out.

---

## 12. Function-preserving initialization in detail

Both `DeepenBlock` and `WidenedBlock` initialize so that **the architectural change is the identity at insert**. The trick is in the BN `weight` (i.e., $\gamma$):

```python
class SepConv3x3(nn.Module):
    def __init__(self, ch, init_zero=False):
        # ... weights kaiming-init, BN biases zero, BN1 weight 1, ...
        if init_zero:
            nn.init.zeros_(self.bn2.weight)         # γ_last = 0
        nn.init.zeros_(self.bn2.bias)               # β_last = 0
```

Because the *last* BN has $\gamma = 0$ and $\beta = 0$, that BN outputs `0 * normalized_input + 0 = 0` *regardless of its input*. So `SepConv(x) ≡ 0` at the moment of insertion, even though every preceding conv weight is non-zero (Kaiming-initialized).

For `DeepenBlock`:

```python
def forward(self, x):
    return F.relu(x + self.body(x))         # SepConv(x) ≡ 0 → ReLU(x + 0) = ReLU(x)
```

Since the input `x` is a post-ReLU output of the previous block (already non-negative), `ReLU(x) = x`, so `DeepenBlock(x) = x` exactly. Function-preserving.

For `WidenedBlock`:

```python
def forward(self, x):
    return self.base(x) + self.branch(x)    # branch(x) ≡ 0 → just self.base(x)
```

The wrapped block's behavior is untouched at insert.

**Why this matters.** Without function-preserving init, every mutation would cause a discontinuous loss spike — sometimes accuracy drops by 5–10 percentage points for a few epochs, sometimes the network never recovers. With identity init, the loss curve is continuous through the mutation, the optimizer state remains useful (because the function the optimizer was tracking didn't change), and post-mutation training picks up where the previous stage left off.

After insertion, the new BN $\gamma$ values are non-zero (they're initialized to 0 but get gradients immediately), so the new block actually starts contributing. The function-preserving property is *only* at insertion time, not for all time.

---

## 13. Optimizer state preservation across mutations

`utils/optimizer_utils.py::refresh_optimizer` rebuilds the optimizer after every mutation while preserving state for the unchanged parameters:

```python
def refresh_optimizer(old_optimizer, model):
    opt_class = type(old_optimizer)
    defaults  = old_optimizer.defaults
    new_optimizer = opt_class(model.parameters(), **defaults)

    old_state = old_optimizer.state
    new_state = new_optimizer.state
    old_params = {id(p): p for grp in old_optimizer.param_groups for p in grp["params"]}

    for grp in new_optimizer.param_groups:
        for p in grp["params"]:
            if id(p) in old_params and old_params[id(p)] in old_state:
                # Copy momentum/exp_avg/exp_avg_sq for unchanged params.
                new_state[p] = {k: (v.clone() if torch.is_tensor(v) else v)
                                for k, v in old_state[old_params[id(p)]].items()}
    return new_optimizer
```

Three things are happening:

1. **Same optimizer class, same hyperparameters** (lr, momentum, weight decay, betas, ...). `defaults` is preserved.
2. **Existing parameter state migrates** by identity (`id(p)`). Since the model's old parameters are *the same Python objects* as before the mutation (the new block's parameters are added to the iterator, but old parameters are not re-instantiated), the `id`-based lookup catches them. Their SGD momentum or Adam `(exp_avg, exp_avg_sq)` is cloned over.
3. **New parameters start fresh** (no entry in `new_state` until the next `optimizer.step()` populates them).

For SGD this means the existing parameters' momentum buffers are untouched, so the velocity through the parameter space is preserved exactly. For Adam it means the existing parameters' first/second moments survive — the bias-correction step counter `step` continues from where it was.

**Why not use `torch.optim.lr_scheduler.add_param_group` instead?** `add_param_group` requires you to know which parameters are new and to add them explicitly. After a mutation, walking the model and identifying "which parameters are new" is awkward (you'd diff the parameter ids before/after). The id-based merge in `refresh_optimizer` is symmetric and works for any structural change.

**`_ParamWrapper` for the teacher arm.** When attention KD is active, the optimizer needs to update both the student and the attention module's `q/k` linear maps. `_ParamWrapper(student, attn_kd)` is a tiny adapter whose `.parameters()` yields the union of both modules' parameters with deduplication by `id`. We pass it to `refresh_optimizer` instead of `student`.

---

## 14. Configuration system (YAML → code path)

The `searh:` YAML block is the single source of truth for both arms. Mapping from key to behavior:

| YAML key | Type | Default | Effect |
|---|---|---|---|
| `searh.enabled` | bool | `false` | If `true`, `train.py` dispatches to `run_searh` and skips the legacy mutation/critic path. |
| `searh.selector` | `"teacher"` / `"critic"` | `"teacher"` | Picks the MV signal source. |
| `searh.epochs_per_stage` | int | 8 | Number of student-training epochs between mutations. |
| `searh.B_op` | int | 7 | Cap on stacked deepens per stage (paper §3.4). |
| `searh.deepen_first` | bool | `true` | Suppress widen until B_op deepens are stacked in a stage. SEArch sets `true`; CGSE may set `false` for free choice. |
| `searh.gamma` | float | 0.5 | Reserved for future intra-stage Eq. 6 selection. Currently unused. |
| `searh.d_k` | int | 64 | Channel-attention key/query dimension. |
| `searh.lambda_init` | float | 1.0 | Initial weight on `L_im` in Eq. 4. Cosine-anneals to 0 across each stage. Critic arm forces `λ = 0`. |
| `searh.score_batches` | int | 2 | Number of test batches to average `D(n)` over at MV time. Each batch costs one teacher forward. |
| `searh.param_budget_factor` | float | 1.5 | Cap = `factor × |θ_0|`. Loop terminates when `count_trainable_parameters ≥ cap`. |
| `searh.param_budget_abs` | int / null | null | Optional absolute cap; if set, `cap = max(factor·|θ_0|, abs)`. |
| `searh.final_retrain_epochs` | int | 8 | Number of epochs to train the final architecture after the search loop. λ=0 always. |
| `searh.hidden_dim` | int | 64 | (Critic arm) MLP hidden width. |
| `searh.critic_lr` | float | 0.01 | (Critic arm) Adam LR for the critic. |
| `searh.critic_weight_decay` | float | 0.0 | (Critic arm) Adam weight decay for the critic. |
| `searh.epsilon` | float | 0.10 | (Critic arm) ε-exploration probability per mutation step. |
| `searh.entropy_beta` | float | 0.01 | (Critic arm) Coefficient on `-H[π]` in the PG loss. |
| `searh.max_epochs_hint` | int | 50 | (Critic arm) Used by `build_critic_state` to normalise the `epoch` feature. Set this to the realistic total epoch count of the run (~234 for matched-cadence Tier 3). |
| `searh.use_student_probe` | bool | `false` | (Critic arm) If `true`, instantiate `StudentProbe`, append 3 probe features per candidate's local descriptor. |
| `searh.baseline_momentum` | float | 0.9 | (Critic arm) EMA momentum for the REINFORCE baseline. **Setting `1.0` cleanly disables the baseline.** |

**Top-level coupling outside the `searh:` block:**

- `model.name: resnet_cifar` is required (we hard-check this in `train.py`).
- `model.depth` selects the student depth (20 typical). For the teacher arm, the teacher checkpoint is loaded based on `teacher.depth + teacher.checkpoint`.
- `mutation.log_jsonl: <path>` enables structural-event logging.
- `training.log_csv: <path>` enables per-epoch metrics logging.
- `teacher.enabled: true` is required for `selector: teacher`.
- `critic.enabled: false` should be set on SEArch arms to avoid instantiating the legacy single-decision critic — `searh.selector: critic` instantiates a `PerCandidateCritic` instead.

---

## 15. Logging schemas (CSV and JSONL)

### 15.1 Per-epoch metrics CSV

`utils/metrics_csv.append_metrics_csv` writes one row per epoch (training stage AND retrain phase) with these columns:

| Column | Meaning |
|---|---|
| `utc_ts` | Run start timestamp (constant across the run). |
| `experiment` | `cfg["experiment"]["name"]`. |
| `epoch` | 0-indexed global epoch number. Resets only across runs. |
| `train_loss`, `train_acc`, `val_loss`, `val_acc` | Loss and accuracy floats. |
| `num_parameters` | Trainable param count *after* this epoch — captures growth. |
| `mutation_applied_yet` | `"True"` if any mutation has fired up to and including this epoch, else `"False"`. |
| `critic_score` | Empty in the SEArch loop (legacy column). |
| `optimizer` | Class name (`"SGD"` or `"Adam"`). |
| `lr` | First param group's current LR. |
| `epoch_seconds` | Wall-clock seconds for this epoch only. |
| `wall_seconds` | Cumulative wall-clock seconds since `run_searh` started. |
| `train_steps` | Cumulative training steps across all epochs in this run. |
| `teacher_forwards` | Cumulative teacher forward passes (KD + score-batches). 0 for critic arm. |
| `stage` | Stage index (`"1"`, `"2"`, …) during the search; `"retrain"` during the final retrain. |
| `lambda_im` | Current cosine-annealed λ (`0.0` for critic arm and during retrain). |
| `mutations_count` | Number of mutations applied so far. |

**Three pieces of information that drive the website:**

- `wall_seconds` enables the "accuracy vs wall-clock" line plots.
- `teacher_forwards` enables the "teacher-overhead-free" claim metric.
- `num_parameters` plotted against `epoch` shows the staircase of architectural growth.

### 15.2 Mutation JSONL

`utils/mutation_log.append_mutation_jsonl` writes one JSON object per structural event:

```json
{
  "event": "mutation",
  "op": "searh_deepen",
  "selector": "critic",
  "gate": "searh_loop",
  "run_id": "tier3_student_resnet20_cifar10_cgse_full_2026-04-28T12:00:00Z",
  "experiment": "tier3_student_resnet20_cifar10_cgse_full",
  "epoch_completed": 7,
  "stage": 1,
  "mutation_index": 1,
  "stage_target": 2,
  "node_id": "stage2",
  "mv": 0.183,
  "ranked_top5": [["deepen stage2 (end-of-stage)", 0.183], ...],
  "num_parameters_before": 268986,
  "num_parameters_after": 273922,
  "param_budget_cap": 403479,
  "use_probe": true,
  "baseline_value": 0.0142
}
```

`use_probe` and `baseline_value` are CGSE-only and let the analysis distinguish across the four ablation arms post-hoc by reading the JSONL alone.

---

## 16. Smoke test expectations

`scripts/smoke_searh.py` runs both arms briefly (small subset, 2 stages, 1 retrain epoch) on whichever device is available. Expected console output sketch:

```text
========== SMOKE: SEArch (teacher arm) ==========
[searh] selector=teacher initial_params=268986 budget_cap=403479 epochs_per_stage=2 B_op=7
[searh] stage 1 ep 1/2 (global 000) | train_loss 1.85 | train_acc 0.36 | val_acc 0.41 | λ=1.000 ...
[searh] stage 1 mutation #1: deepen stage1 (end-of-stage) (mv=+0.0021) params 268986 → 269082
[searh] stage 2 ep 1/2 (global 002) | val_acc 0.45 | λ=1.000 ...
[searh] stage 2 mutation #2: deepen stage2 (end-of-stage) (mv=+0.0035) params 269082 → 270070
[searh] final retrain for 1 epochs (no mutations).
[searh] DONE. mutations=2 final_params=270070 epochs_run=5

========== SMOKE: CGSE (critic arm, probe OFF) ==========
[searh] selector=critic initial_params=268986 budget_cap=403479 epochs_per_stage=2 B_op=7
[searh] stage 1 ep 1/2 ...
[searh] stage 1 mutation #1: widen stage3 (end-of-stage) (mv=-0.0124)
[searh] critic PG update: choice=2 R=+0.0312 baseline=+0.0031 adv=+0.0281 entropy=1.609
...

========== SMOKE: CGSE (critic arm, probe ON) ==========
[searh] selector=critic ... 
[searh] student probe enabled (per-stage local_dim grows by 3)
...
```

Things to verify in the smoke output:

- **All three runs complete without errors.** Most common failure is the channel-mismatch `RuntimeError` that the candidate-enumeration filter prevents — if you see it, `_is_widenable_basic_block` is broken.
- **`λ` cosine-anneals from 1.0 → 0** within each stage of the teacher arm.
- **`teacher_forwards` is 0** in both critic arms (visible in the CSV row at retrain, which is logged regardless of arm).
- **`baseline_value` is 0.0** for the first PG update (warm start) and starts moving on subsequent updates.
- **`entropy ≈ ln(K)`** at first PG update where `K` is the number of candidates — confirms the policy starts uniform.
- **Probe-on output mentions** `"student probe enabled (per-stage local_dim grows by 3)"` and the critic input dim is 8 (state) + 8 (local with probe) = 16.

---

## 17. Reproducibility and determinism

Seeds are set in `train.py` early and propagated to `torch.manual_seed`, `np.random.seed`, and `random.seed`. Beyond this, exact reproducibility is **not** guaranteed because:

- CUDA/MPS use non-deterministic algorithms by default for some ops (cuDNN's autotuner picks different conv implementations across runs).
- The data loader uses workers with their own RNG state.
- Power iteration in `student_probe.py::_activation_variance_ratio` initializes from `torch.randn(c, device=...)`, which uses the global RNG.

**Acceptable variance.** Across seeds 41/42/43 (or 42/43/44 for Tier 3), final accuracies typically vary by ±0.4% on Tier 2 and we expect similar on Tier 3. The headline "CGSE > SEArch" claim should hold across all 3 seeds; if it only holds on one, it's noise.

For runs that *do* need bit-exact reproducibility (e.g., debugging a flaky mutation):

```python
torch.use_deterministic_algorithms(True)        # forces deterministic CUDA kernels
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

These are not on by default because they cost ~10–20% wall-clock.

---

## 18. Performance characteristics

Measured on Apple M-series (MPS), batch 128, ResNet-20 student / ResNet-56 teacher, full CIFAR-10:

| Component | Cost per step | Cost per epoch |
|---|---|---|
| Student fwd + bwd (CE only) | ~6 ms | ~30 s (390 steps) |
| Teacher fwd (no_grad) | ~9 ms | ~14 s when KD on |
| Channel-attention KD (3 heads, fwd) | ~2 ms | ~3 s when KD on |
| **Total per epoch (teacher arm)** | ~17 ms | **~60 s** |
| **Total per epoch (critic arm)** | ~6 ms | **~35 s** (no teacher, no KD) |
| Score-batch loop (2 batches × 1 teacher fwd each) | — | ~0.05 s per mutation |
| Probe forward + grad collection | — | ~0.005 s per mutation |
| `enumerate_candidates` + `refresh_optimizer` | — | <1 ms per mutation |

Implications:

- The critic arm is ~1.7× faster per epoch than the teacher arm at matched batch size on MPS. At 234 matched-cadence Tier 3 epochs, that's ~3.9 h vs ~2.3 h per seed.
- The teacher's `no_grad` forward is the single biggest non-student cost. **Budgeted KD** (Tier 2 experiment) reduced this by gating teacher forwards every-N-steps; Tier 3 leaves it on every step for paper-faithfulness.
- Probe overhead is negligible (<0.5% of training time) — `act_var_ratio` is the dominant cost via power iteration but it's only computed once per stage end.

---

## 19. Common gotchas

### 19.1 Empty hooks after a model save/load round-trip

`MultiNodeAttentionKD._handles` and `StudentProbe._hooks` are *not* serialized when you `torch.save(student.state_dict())`. After loading a checkpoint, you must call `attach()` again before the next forward pass. `run_searh` does this internally; if you load a checkpoint and forward outside `run_searh`, you'll get all-zero distances and silent miscomputation.

### 19.2 `WidenedBlock` wrapped twice can hide the inner `BasicBlock`

If the candidate enumerator allows a `WidenedBlock` to be re-wrapped (which it does — see `widen_resnet_stage`'s `WidenedBlock` branch), the resulting object is `WidenedBlock(WidenedBlock(BasicBlock))`. The inner channel inference walks `base.base.bn2.num_features`, which works as long as the innermost block is a `BasicBlock`. If you ever insert a stride-changing operator at the inner edge, this will break.

### 19.3 Large `lambda_init` causes BN to drift

The cosine $\lambda$-anneal *resets* at every stage, so a large `lambda_init` (e.g., 5.0) effectively trains the imitation loss harder *every* stage. Empirically this destabilizes the teacher's running BN stats and hurts final accuracy. The paper uses 1.0; we keep that as the default.

### 19.4 `epsilon` interacts with `entropy_beta`

`epsilon` (uniform-random exploration probability) and `entropy_beta` (PG loss entropy bonus) are *both* keeping the policy from collapsing. Using both at high values (e.g., `ε=0.5`, `entropy_beta=0.1`) makes the critic essentially random — the policy gradient signal gets drowned out. Empirically `ε=0.10–0.30` and `entropy_beta=0.01` work well together; either alone tends to under-explore.

### 19.5 `score_batches` is a teacher-forward cost

Each batch in `compute_per_node_distances` runs a teacher forward. With `score_batches=2`, that's `2 × stages × seeds` extra teacher forwards on top of the per-step KD cost — counted in the cumulative `teacher_forwards` CSV column.

### 19.6 The probe's snapshot list can get stale

`probe.snapshot_stage(model, stage_id)` overwrites the snapshot for that stage. If you mutate a stage but forget to re-snapshot (line 509 in `searh_loop.py`), `weight_delta` for that stage will conflate the old architecture's weight drift with the new architecture's drift — typically yielding a huge spurious value. The mutation handler does this automatically; only worry if you're hand-coding mutation paths.

---

## 20. Extension points

### 20.1 Adding a new operator

1. Implement the operator in `ops/searh_<name>.py`. Pattern:
   - Mutates `model` in place, returns nothing or a description string.
   - Channel-preserving (all I/O shapes unchanged).
   - Function-preserving init via zero-`gamma` BN somewhere in the inserted module.
2. Add the new candidate type to `evolution/candidates.py::Candidate` and update `enumerate_candidates` to emit rows for the new op when legal.
3. Update `run_searh`'s mutation-application branch (currently a 3-line `if/elif/raise`):

```python
if chosen.op == "deepen":
    deepen_resnet_stage(student, stage=chosen.stage, after_block_idx=None)
elif chosen.op == "widen":
    widen_resnet_stage(student, stage=chosen.stage, block_idx=None)
elif chosen.op == "<your_op>":
    your_op(student, stage=chosen.stage)
else:
    raise ValueError(f"Unknown op '{chosen.op}'")
```

4. Update the local-descriptor `op_is_widen` field if your op deserves its own one-hot (currently the descriptor only distinguishes `deepen` vs `widen`).
5. Smoke-test via `scripts/smoke_searh.py` — add a config that exercises the new op.

### 20.2 Adding a new probe feature

1. Compute the per-stage scalar in `critics/student_probe.py::per_stage_features` and append it to the returned list.
2. Bump `PROBE_DIM` accordingly. The critic's `local_dim` is computed as `5 + PROBE_DIM` in `train.py`, so you don't need to change the dispatch — but you *do* need to retrain the critic (existing checkpoints won't load with the new dim).
3. Document the new feature in [`CGSE-math-and-equations.md`](CGSE-math-and-equations.md) §5 (Eqs. 11–17 area) for paper consistency.

### 20.3 Adding a reversible operator (un-deepen / un-widen)

The "future work" extension noted in [`SEArch-baseline-and-CGSE-evaluation-plan.md`](SEArch-baseline-and-CGSE-evaluation-plan.md) §3a:

1. Add `ops/searh_undeepen.py::undeepen_resnet_stage(model, stage)`: pops the last `DeepenBlock` from `model.layer{stage}`. Returns the architecture to a state the network already trained well at — no recovery training needed.
2. Add `ops/searh_unwiden.py::unwiden_resnet_stage(model, stage)`: replaces a `WidenedBlock` with its inner `block.base`. The branch's parameters are discarded; the wrapped block's parameters survive.
3. In `evolution/candidates.py::enumerate_candidates`, emit `un_deepen` whenever `count_deepens_in_stage > 0` and `un_widen` whenever `count_widens_in_stage > 0`. No new safety filter needed (both ops are always safe).
4. Update the critic's local descriptor to one-hot over `{deepen, widen, un_deepen, un_widen}` (4-dim instead of 1-dim). Bump `local_dim` accordingly.
5. Decide on a new param-budget rule: should the cap apply to the *peak* params or the *current* params? Probably the latter so the critic can shrink and grow within the cap. Update the termination check in `run_searh`.

This would let CGSE genuinely *prune*, which SEArch's monotonic teacher-MV cannot. Strictly stronger paper claim.

### 20.4 Replacing the per-candidate critic with a transformer

`PerCandidateCritic` is a 3-layer MLP. To replace it with, say, a small transformer that attends across candidates:

1. Replace `nn.Sequential(Linear, ReLU, Linear, ReLU, Linear)` with `nn.TransformerEncoder` over the (K candidates) sequence.
2. Add positional encoding for stage index.
3. Update `_critic_mv_selector` to call `critic(x.unsqueeze(0)).squeeze(0)` for batch-1 sequence input.
4. The PG update path is unchanged.

This is more capacity than needed for 3-stage ResNet-20 but worth trying for larger architectures (Tier 4 ImageNet scale-out).

---

**See also.** [`CGSE-codebase-guide.md`](CGSE-codebase-guide.md) for the file-by-file map. [`CGSE-implementation-log.md`](CGSE-implementation-log.md) for the chronology of when each component landed and why. [`CGSE-math-and-equations.md`](CGSE-math-and-equations.md) for the equations. [`CGSE-experiments-and-results-guide.md`](CGSE-experiments-and-results-guide.md) for the tier ladder and the Tier 3 sweep plan.
