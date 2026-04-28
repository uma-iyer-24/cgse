# CGSE — Mathematical formulation and equations for the paper

This document collects every equation the paper needs in publication-ready
form, with a single consistent notation throughout. All equations are
numbered for cross-reference from the manuscript. Where we adapt or
deviate from the SEArch reference paper (Liang, Xiang & Li, *Neurocomputing*
651, 2025) we mark the line with **★** and explain the deviation in line.

Companion docs:

- [`SEArch-baseline-and-CGSE-evaluation-plan.md`](SEArch-baseline-and-CGSE-evaluation-plan.md) — prose argument and ablation grid.
- [`CGSE-implementation-log.md`](CGSE-implementation-log.md) — code-side history of each equation's introduction.
- [`CGSE-codebase-guide.md`](CGSE-codebase-guide.md) — file-level mapping from each equation to its module.

---

## 1. Setup and notation

We train a **student** classifier $f_\theta : \mathcal{X} \to \mathcal{Y}$ on a labelled dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ of CIFAR-10 images. Throughout:

| Symbol | Meaning |
|---|---|
| $\theta$ | student parameters (the only quantity that ever moves) |
| $\theta_0$ | student parameters at the start of training |
| $\Theta$ | frozen teacher parameters (only used by the SEArch arm) |
| $f_\theta(x)$ | student logits in $\mathbb{R}^{|\mathcal{Y}|}$ |
| $\mathcal{S} = \{1, 2, 3\}$ | the three CIFAR-ResNet stages |
| $h_n^{\theta}(x) \in \mathbb{R}^{B \times C_n \times H_n \times W_n}$ | the *node feature* at stage $n$ (output of `model.layern`) |
| $\mathcal{C}_t$ | the legal candidate set at decision step $t$ |
| $c = (n, o)$ | a candidate: stage $n \in \mathcal{S}$ paired with operator $o \in \{\textsf{deepen}, \textsf{widen}\}$ |
| $\textsf{Op}_o(\theta; n)$ | apply operator $o$ to stage $n$ of the student |
| $\rho$ | parameter-budget multiplier (we use $\rho = 1.5$) |
| $T_{\textsf{stage}}$ | number of training epochs per outer-loop stage |
| $T_{\textsf{retrain}}$ | number of final-retrain epochs |
| $B_{\textsf{op}}$ | maximum number of deepens stacked in any one stage (we use $7$) |

All equations are written for one batch unless stated otherwise. Means
across batches are taken implicitly when an equation appears inside an
expectation.

---

## 2. Channel-space attention KD (SEArch arm only)

For each pair of matched stage outputs $(h_n^\theta(x), h_n^\Theta(x))$ we
project the teacher's feature map onto the student's channel space using
a learned channel-attention head.

**Per-channel descriptor.** Reduce each feature map to one scalar per
channel by global-average pooling:

$$
\boxed{\;
d_n^{\theta}[c]
\;=\;
\frac{1}{H_n W_n}\sum_{i=1}^{H_n}\sum_{j=1}^{W_n} h_n^{\theta}[c, i, j],
\qquad c = 1,\ldots,C_n^\theta.
\;}
\tag{1}
$$

The teacher analogue $d_n^{\Theta}[c]$ is defined identically.

**Channel-attention weights.** With learnable $W_Q, W_K \in \mathbb{R}^{d_k \times 1}$
shared across channels:

$$
\boxed{\;
A_n^{c, t}(x)
\;=\;
\textsf{softmax}_{t}\!\left(
\frac{(W_Q\, d_n^\theta[c])^{\top}\, (W_K\, d_n^\Theta[t])}{\sqrt{d_k}}
\right)
\quad\in\;[0, 1]^{\,C_n^\theta \times C_n^\Theta}.
\;}
\tag{2}
$$

**Projected teacher feature.** Mix teacher channels into the student's
channel space, broadcasting over spatial positions:

$$
\boxed{\;
\widehat{h}_n^{\,\theta}[c, i, j](x)
\;=\;
\sum_{t = 1}^{C_n^\Theta} A_n^{c, t}(x)\; h_n^{\Theta}[t, i, j].
\;}
\tag{3}
$$

**Per-node distance.** Squared $L_2$ between the student's actual feature
and the teacher's projected analogue, averaged over batch, channels, and
spatial positions:

$$
\boxed{\;
D_n(x)
\;=\;
\frac{1}{B\, C_n^\theta\, H_n\, W_n}\;
\bigl\lVert h_n^{\theta}(x) - \widehat{h}_n^{\,\theta}(x) \bigr\rVert_2^{\,2}.
\;}
\tag{4}
$$

**★ Deviation from paper.** SEArch writes $\lVert\cdot\rVert_2^2$ summed
over channels (their notation in §3.2). We normalise by the element count
so that $D_n$ is $\mathcal{O}(1)$ regardless of feature-map size; this
makes the imitation loss numerically comparable across stages and stable
when summed with cross-entropy at $\lambda \approx 1$. The relative
ordering of stages — which is what the MV scorer needs — is preserved.

**Imitation loss.** Average across all matched stage pairs:

$$
\boxed{\;
\mathcal{L}_{\textsf{im}}(\theta; x)
\;=\;
\frac{1}{|\mathcal{S}|}\sum_{n \in \mathcal{S}} D_n(x).
\;}
\tag{5}
$$

---

## 3. Combined training objective (SEArch arm)

Within each outer-loop stage of length $T_{\textsf{stage}}$ epochs, the
imitation weight $\lambda$ is annealed by a half-cosine schedule from
$\lambda_0$ down to $0$:

$$
\boxed{\;
\lambda(t)
\;=\;
\frac{\lambda_0}{2}\,\Bigl(1 + \cos\!\Bigl(\frac{\pi\, t}{T_{\textsf{stage}} - 1}\Bigr)\Bigr),
\qquad t = 0, 1, \ldots, T_{\textsf{stage}} - 1.
\;}
\tag{6}
$$

The per-batch loss is then

$$
\boxed{\;
\mathcal{L}^{\textsf{SEArch}}(\theta; x, y, t)
\;=\;
\underbrace{\textsf{CE}\!\bigl(f_\theta(x), y\bigr)}_{\textsf{label loss}}
\;+\;
\lambda(t)\;\underbrace{\mathcal{L}_{\textsf{im}}(\theta; x)}_{\textsf{imitation loss}}.
\;}
\tag{7}
$$

The CGSE arm's training loss is just the first term (no teacher term):

$$
\boxed{\;
\mathcal{L}^{\textsf{CGSE}}(\theta; x, y)
\;=\;
\textsf{CE}\!\bigl(f_\theta(x), y\bigr).
\;}
\tag{8}
$$

This is the *only* place the loss differs between the two arms.

---

## 4. Modification value (SEArch arm)

At the end of each outer-loop stage, SEArch ranks candidates by

$$
\boxed{\;
\textsf{MV}(n)
\;=\;
\mathbb{E}_{x \sim \mathcal{D}_{\textsf{score}}}[D_n(x)]\;\cdot\;\frac{\textsf{deg}^+(n)}{\textsf{deg}^-(n)},
\;}
\tag{9}
$$

where $\textsf{deg}^+(n)$ and $\textsf{deg}^-(n)$ are the out- and in-degree
of node $n$ in the student's computational graph, and the expectation is
estimated by averaging $D_n$ over a small score set $\mathcal{D}_{\textsf{score}}$
of unlabelled batches.

**★ Adaptation to CIFAR-ResNets.** In our linear ResNet stack every
internal stage has $\textsf{deg}^+(n) = \textsf{deg}^-(n) = 1$, so the
ratio collapses to $1$ and $\textsf{MV}(n) = \mathbb{E}[D_n]$. We retain
the form of Eq. (9) in code so that future DAG generalisations drop in
without API changes.

The SEArch selector picks the candidate site with highest MV:

$$
\boxed{\;
n^\star
\;=\;
\arg\max_{n \in \mathcal{S}}\,\textsf{MV}(n),
\qquad
o^\star
\;=\;
\begin{cases}
\textsf{deepen} & \text{if } |\textsf{deepens}(n^\star)| < B_{\textsf{op}}, \\
\textsf{widen}  & \text{otherwise.}
\end{cases}
\;}
\tag{10}
$$

---

## 5. CGSE — Student probe (unsupervised analogue of $D_n$)

The CGSE critic does not see the teacher. To give it a SEArch-equivalent
locality signal, we build a *student-only* probe that produces a 3-dim
descriptor per stage at each decision step.

### 5.1 Activation variance ratio

Let $a_n(x) \in \mathbb{R}^{B \times C_n \times H_n \times W_n}$ be the
stage-$n$ post-activation tensor for a probe batch $x$. Reshape into a
matrix where each row is one position's channel vector:

$$
A_n \;=\;
\begin{bmatrix}
\rule[.5ex]{2em}{.4pt}\;a_n(x)[1,\cdot,1,1]\;\rule[.5ex]{2em}{.4pt} \\
\rule[.5ex]{2em}{.4pt}\;a_n(x)[1,\cdot,1,2]\;\rule[.5ex]{2em}{.4pt} \\
\vdots \\
\rule[.5ex]{2em}{.4pt}\;a_n(x)[B,\cdot,H_n,W_n]\;\rule[.5ex]{2em}{.4pt}
\end{bmatrix}
\;\in\;\mathbb{R}^{(B H_n W_n) \times C_n}.
$$

After centring rows ($\bar A_n = A_n - \mathbf{1}\bar a_n^{\top}$) the
**channel covariance** is

$$
\boxed{\;
\Sigma_n
\;=\;
\frac{1}{B H_n W_n - 1}\,\bar A_n^{\top}\bar A_n
\;\in\;\mathbb{R}^{C_n \times C_n}.
\;}
\tag{11}
$$

The **top-1 variance ratio** is

$$
\boxed{\;
\rho_n
\;=\;
\frac{\lambda_{\max}(\Sigma_n)}{\textsf{tr}(\Sigma_n)}
\;\in\;[0, 1].
\;}
\tag{12}
$$

We compute $\lambda_{\max}(\Sigma_n)$ by $K = 4$ steps of power iteration
on $\Sigma_n$, avoiding an explicit eigendecomposition:

$$
v_{k+1}
\;=\;
\frac{\Sigma_n\, v_k}{\lVert \Sigma_n\, v_k \rVert_2},
\qquad
\lambda_{\max}(\Sigma_n)
\;\approx\;
v_K^{\top}\,\Sigma_n\,v_K.
\tag{13}
$$

**Interpretation.** $\rho_n \to 1$ when stage $n$'s representation
collapses onto a single direction (severe bottleneck); $\rho_n \to 1/C_n$
when channels carry independent variance (healthy). This is the
unsupervised analogue of SEArch's $D_n$: both detect "the student's
representation here is impoverished".

### 5.2 Stage-gradient norm

For the parameters $\theta_n \subseteq \theta$ that live in stage $n$,
the per-stage gradient norm captured from the last backward pass is

$$
\boxed{\;
g_n
\;=\;
\biggl\lVert\nabla_{\theta_n}\,\mathcal{L}^{\textsf{CGSE}}(\theta;\,\textsf{minibatch})\biggr\rVert_2.
\;}
\tag{14}
$$

We then min-max normalise across the three stages so the critic sees
*relative* gradient magnitude:

$$
\widetilde g_n
\;=\;
\frac{g_n}{\max_{m \in \mathcal{S}} g_m + \varepsilon}.
\tag{15}
$$

### 5.3 Weight-delta since last mutation

Let $\theta_n^{(\star)}$ denote the snapshot of stage-$n$ parameters
taken immediately after the most recent mutation in that stage. Define

$$
\boxed{\;
\delta_n
\;=\;
\bigl\lVert \theta_n - \theta_n^{(\star)} \bigr\rVert_F,
\qquad
\widetilde\delta_n
\;=\;
\frac{\delta_n}{\max_{m \in \mathcal{S}} \delta_m + \varepsilon}.
\;}
\tag{16}
$$

When stage $n$ has just been mutated, $\delta_n = 0$; it grows as
training proceeds, plateauing when the stage stops learning. This is
a "have I converged here yet?" signal.

### 5.4 Per-stage probe descriptor

The probe outputs

$$
\boxed{\;
\phi_n
\;=\;
\bigl(\rho_n,\;\widetilde g_n,\;\widetilde\delta_n\bigr)
\;\in\;\mathbb{R}^{3}
\quad\text{for each } n \in \mathcal{S}.
\;}
\tag{17}
$$

When the probe is disabled (`use_student_probe: false`) we set
$\phi_n = \emptyset$ and the local descriptor below shrinks accordingly.

---

## 6. CGSE — Critic policy

### 6.1 Global state

The 8-dim global state vector is built from training statistics
(`critics/state_features.py`):

$$
s \;=\; \bigl(
\textsf{train\_loss},\;
\textsf{val\_acc},\;
\Delta\textsf{train\_loss},\;
\Delta\textsf{val\_acc},\;
\tfrac{|\theta|}{|\theta_0|},\;
\tfrac{\textsf{epoch}}{\textsf{max\_epochs}},\;
\textsf{loss\_to\_anchor\_ratio},\;
1
\bigr)
\;\in\;\mathbb{R}^{8}.
\tag{18}
$$

### 6.2 Local descriptor per candidate

For candidate $c = (n_c, o_c)$:

$$
\boxed{\;
\ell_c
\;=\;
\bigl(
\underbrace{\mathbb{1}[n_c{=}1],\,\mathbb{1}[n_c{=}2],\,\mathbb{1}[n_c{=}3]}_{\textsf{stage one-hot}},\;
\underbrace{\mathbb{1}[o_c{=}\textsf{widen}]}_{\textsf{op flag}},\;
\underbrace{\tfrac{|\textsf{deepens}(n_c)|}{B_{\textsf{op}}}}_{\textsf{deepen-fill}},\;
\underbrace{\phi_{n_c}}_{\textsf{probe (opt.)}}
\bigr)
\;\in\;\mathbb{R}^{5+P},
\;}
\tag{19}
$$

with $P = 3$ if the probe is enabled, else $P = 0$.

### 6.3 Score and policy

A small MLP $\pi_\psi$ scores each candidate independently:

$$
\boxed{\;
u_c
\;=\;
\pi_\psi\!\bigl(s \,\Vert\, \ell_c\bigr)
\;\in\;\mathbb{R},
\;}
\tag{20}
$$

where $\Vert$ denotes vector concatenation. The action policy is the
candidate-wise softmax with $\varepsilon$-greedy exploration:

$$
\boxed{\;
\pi(c\mid s, \mathcal{C})
\;=\;
\begin{cases}
1/|\mathcal{C}| & \text{w.p. } \varepsilon \quad\text{(exploration)}, \\[4pt]
\dfrac{\exp u_c}{\sum_{c' \in \mathcal{C}} \exp u_{c'}} & \text{w.p. } 1 - \varepsilon \quad\text{(exploitation)}.
\end{cases}
\;}
\tag{21}
$$

The chosen candidate $c^\star$ replaces SEArch's $(n^\star, o^\star)$ in
Eq. (10).

---

## 7. CGSE — REINFORCE update with EMA baseline

After the next outer-loop stage finishes we observe the change in
validation accuracy:

$$
\boxed{\;
R_t
\;=\;
\textsf{val\_acc}_{t+1} - \textsf{val\_acc}_{t}.
\;}
\tag{22}
$$

We maintain an exponential-moving-average baseline of past rewards
(initialised at $0$):

$$
\boxed{\;
b_{t+1}
\;=\;
\mu\, b_t \;+\; (1 - \mu)\, R_t,
\qquad
\textsf{advantage}\quad
A_t \;=\; R_t - b_t.
\;}
\tag{23}
$$

The policy-gradient estimator (single-sample REINFORCE with entropy
regularisation):

$$
\boxed{\;
\mathcal{L}_{\textsf{PG}}(\psi)
\;=\;
-\,A_t\,\log \pi_\psi(c^\star_t \mid s_t, \mathcal{C}_t)
\;-\;
\beta\,\mathcal{H}\!\bigl[\pi_\psi(\cdot \mid s_t, \mathcal{C}_t)\bigr],
\;}
\tag{24}
$$

where the entropy of the policy is

$$
\mathcal{H}[\pi] \;=\; -\sum_{c \in \mathcal{C}} \pi(c) \log \pi(c).
\tag{25}
$$

We update $\psi$ by Adam on $\mathcal{L}_{\textsf{PG}}$.

**Why subtract a baseline.** $\mathbb{E}[\nabla_\psi \log \pi_\psi(c)] = 0$,
so subtracting any state-only quantity from $R_t$ leaves the gradient
*unbiased* but reduces its variance whenever $b_t$ correlates with
$R_t$. With sparse $\Delta\textsf{val}$ rewards (often within a few
$10^{-2}$) the variance reduction is substantial — empirically the
critic stops chasing absolute-$\Delta$ noise and starts ranking
*relative* improvements.

**Why entropy regularisation.** Without the $\beta\mathcal{H}$ term
REINFORCE can collapse onto a single action early if exploration draws
even mildly favour one candidate. The entropy bonus pushes the policy
toward the uniform distribution and is annealed implicitly: as the
policy gets more confident, $\mathcal{H}$ shrinks, so the bonus's effect
weakens. We use $\beta = 10^{-2}$.

---

## 8. Edge-splitting operators (function-preserving)

**Operator inventory (both arms).** The teacher-MV selector and the
critic-policy selector operate over the *exact same* candidate set:

$$
\mathcal{C}_t \;\subseteq\;
\mathcal{S}\,\times\,\bigl\{\textsf{deepen},\,\textsf{widen}\bigr\}.
\tag{*}
$$

Two grow operators, no prune, no noop, no cross-stage moves. The
architecture grows monotonically until the candidate set is exhausted
or the param-budget cap is hit. This matches SEArch's paper inventory
(Fig. 4a/4b) by design: *the contribution is in the signal that
chooses among candidates, not in the candidate set itself.*

Both operators are **function-preserving at insertion** — the
architectural change is the identity function the moment the operator
is applied — so the loss landscape is continuous through the edit and
no accuracy is sacrificed before training resumes.

### 8.1 Depthwise-separable Conv-3$\times$3 (paper §3.5)

For input $x \in \mathbb{R}^{B \times C \times H \times W}$ define

$$
\boxed{\;
\textsf{SepConv}_C(x)
\;=\;
\textsf{BN}_2\!\bigl(W_{\textsf{pw}} \star \textsf{ReLU}\bigl(\textsf{BN}_1(W_{\textsf{dw}} \circledast x)\bigr)\bigr),
\;}
\tag{26}
$$

with $W_{\textsf{dw}} \in \mathbb{R}^{C \times 1 \times 3 \times 3}$ a
depthwise convolution (groups $= C$), $W_{\textsf{pw}} \in \mathbb{R}^{C \times C \times 1 \times 1}$
a pointwise convolution, and $\circledast, \star$ the standard 2-D
convolution operators. We zero-init the second BN's affine scale,
$\gamma_{\textsf{BN}_2} = 0$, so $\textsf{SepConv}_C(x) \equiv 0$ at insert.

### 8.2 Deepen operator (Eq. 27)

Insert one residual sep-conv block at the end of stage $n$:

$$
\boxed{\;
\textsf{Op}_{\textsf{deepen}}(\theta;\, n)\colon
\quad x \longmapsto \textsf{ReLU}\!\bigl(x + \textsf{SepConv}_{C_n}(x)\bigr).
\;}
\tag{27}
$$

Since $\textsf{SepConv}_{C_n}(x) \equiv 0$ at init, the post-ReLU output
equals $\textsf{ReLU}(x) = x$ (the input is already post-ReLU from the
previous block) — the operator is the identity at insertion.

### 8.3 Widen operator (Eq. 28)

Wrap a chosen `BasicBlock` $B(\cdot)$ with a parallel sep-conv branch:

$$
\boxed{\;
\textsf{Op}_{\textsf{widen}}(\theta;\, n)\colon
\quad x \longmapsto B(x) + \textsf{SepConv}_{C_n}(x).
\;}
\tag{28}
$$

By the same zero-BN-init argument the branch outputs $0$ and the wrapped
block's behaviour is preserved.

**Eligibility constraint** (★ deviation from paper). We restrict
`widen` to blocks with stride $1$ and matching in/out channels — the
first block of stages 2 and 3 is stride-2 with channel doubling, and a
parallel same-shape branch on those would be ill-defined. The paper's
DAG widen on a non-degenerate edge does not have this constraint
because edges can carry their own stride; in our linear stack this
constraint is the natural one.

---

## 9. Outer loop (Algorithm 1, both arms)

The outer loop is identical for SEArch and CGSE; only the selector that
fills line 8 differs.

```
Algorithm 1: Iterative train-then-evolve outer loop.

Inputs:
  Initial student θ_0, optional teacher Θ, training set D, score set D_score,
  selector ∈ {SEArch, CGSE}, T_stage, T_retrain, B_op, ρ.

  1: θ ← θ_0
  2: |θ_0| ← param count of θ
  3: stage ← 0, mutations ← 0, baseline b ← 0
  4: while stage = 0 OR (|θ| < ρ |θ_0|  AND  C_stage ≠ ∅):
  5:   stage ← stage + 1
  6:   for t ← 0 .. T_stage - 1:
  7:       SGD step on L^selector(θ; ·, ·, t)               # Eq. (7) or (8)
  8:   if selector = CGSE and stage > 1:
  9:       update critic ψ on advantage A_{t-1}             # Eqs. (22-25)
 10:   C_stage ← legal candidates(θ, B_op)
 11:   if C_stage = ∅: break
 12:   if selector = SEArch:
 13:       n* ← arg max_n MV(n)                              # Eq. (9)
 14:       o* ← deepen if |deepens(n*)| < B_op else widen    # Eq. (10)
 15:       c* ← (n*, o*)
 16:   else if selector = CGSE:
 17:       collect probe φ for all stages                    # Eqs. (11-17)
 18:       sample c* ~ π_ψ(· | s, C_stage)                   # Eq. (21)
 19:   θ ← Op_{o*}(θ; n*)                                    # Eq. (27) or (28)
 20:   refresh optimizer state to include new params
 21:   mutations ← mutations + 1
 22: # final retrain phase, no mutations.
 23: for r ← 1 .. T_retrain:
 24:     SGD step on L^selector(θ; ·, ·, ·)
 25: return θ
```

The differences between arms are exactly:

- Line 7: SEArch uses $\mathcal{L}^{\textsf{SEArch}}$ (Eq. 7), CGSE uses $\mathcal{L}^{\textsf{CGSE}}$ (Eq. 8).
- Lines 12-15 vs. 16-18: deterministic teacher MV (Eq. 10) vs. learned critic policy (Eq. 21).
- Line 9 only fires for the CGSE arm.

Everything else — operators, B_op cap, param-budget termination, final
retrain — is shared code path.

**Empirical matching.** In all reported experiments
$T_{\textsf{stage}}$, $T_{\textsf{retrain}}$, $B_{\textsf{op}}$, and
$\rho$ are held *numerically identical* between the SEArch and CGSE
arms (Tier 3: $T_{\textsf{stage}} = 8$, $T_{\textsf{retrain}} = 10$,
$B_{\textsf{op}} = 7$, $\rho = 1.5$). This is a deliberate parity
choice: the contribution claim is "swapping the MV signal source from
teacher attention to a learned critic, with optional locality probe and
variance-reduced policy gradient, while everything else stays fixed."
Any cadence asymmetry would conflate the signal substitution with the
schedule, and is therefore avoided.

---

## 10. Parameter-budget bound

The outer loop terminates the moment

$$
|\theta| \;\geq\; \rho\,|\theta_0|
\quad\text{or}\quad
\mathcal{C}_{\textsf{stage}} \;=\; \emptyset.
\tag{29}
$$

For the configs in this repo $\rho = 1.5$. With $|\theta_0|$ for ResNet-20
equal to $\approx 272{,}474$ parameters, the cap is $\approx 408{,}711$.

A bound on the cardinality of $\mathcal{C}_{\textsf{stage}}$ across all
stages: with $B_{\textsf{op}} = 7$ deepens permitted per stage, and at
most $|\textsf{widenable}(n)|$ widens per stage,

$$
|\textsf{mutations}| \;\leq\; \sum_{n \in \mathcal{S}}\bigl(B_{\textsf{op}} + |\textsf{widenable}(n)|\bigr).
\tag{30}
$$

For ResNet-20 this evaluates to $7 \cdot 3 + (3 + 2 + 2) = 28$ mutations.
Empirically the loop terminates on candidate-exhaustion (right-hand
clause of Eq. 29) before hitting the param cap, since each operator adds
only $\mathcal{O}(C_n^2)$ parameters.

---

## 11. Compute and teacher-forward bookkeeping

Let $T_{\textsf{total}}$ denote the total epoch count and $|\mathcal{D}|/b$
the steps-per-epoch with batch size $b$. Then:

| Arm | Teacher forwards | Imitation loss applied |
|---|---|---|
| SEArch | $T_{\textsf{total}} \cdot \frac{|\mathcal{D}|}{b}$ + score batches | every batch |
| CGSE | $0$ | never |

The CGSE arm's "training cost" is therefore exactly the cost of
training a ResNet-20 by SGD on labels alone — there is no per-batch
extra forward pass and no extra loss term.

---

## 12. Summary of equation numbers (paper-facing)

| Eq. | Quantity |
|-----|----------|
| (1) | per-channel descriptor |
| (2) | channel-attention weights |
| (3) | projected teacher feature |
| (4) | per-node distance $D_n$ |
| (5) | imitation loss $\mathcal{L}_{\textsf{im}}$ |
| (6) | cosine $\lambda$-anneal |
| (7) | SEArch arm loss |
| (8) | CGSE arm loss |
| (9) | modification value $\textsf{MV}(n)$ |
| (10) | SEArch selector |
| (11)–(13) | channel covariance + power iteration |
| (14)–(15) | per-stage gradient norm |
| (16) | weight-delta since last mutation |
| (17) | probe descriptor $\phi_n$ |
| (18) | global state $s$ |
| (19) | local descriptor $\ell_c$ |
| (20) | critic score |
| (21) | $\varepsilon$-greedy softmax policy |
| (22) | reward $R_t$ |
| (23) | EMA baseline + advantage |
| (24)–(25) | REINFORCE-with-baseline + entropy |
| (26) | depthwise-separable Conv-3$\times$3 |
| (27) | deepen operator |
| (28) | widen operator |
| (29) | param-budget termination |
| (30) | mutation-count bound |

A LaTeX preamble that supports every equation in this document needs only
the standard `amsmath`, `amssymb`, and `mathtools` packages plus
`\usepackage{bm}` for any bold-symbol substitutions; no custom commands
are required.
