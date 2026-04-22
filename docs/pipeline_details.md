# GoS v1 & v2 — Step-by-step dimensions and v2 motivations

Symbols used throughout:

| Symbol | Meaning | Typical value |
|---|---|---|
| `N` | ways (classes per episode) | 5 or 10 |
| `K` | shots (support per class) | 1 or 5 |
| `Q` | number of query molecules per episode | 16 (3–4 per class) |
| `M` | total nodes in the meta-graph | `N·K + Q` (~40 for 5w5s+Q=16) |
| `d` | embedding dim | 128 |
| `k` | k-NN sparsification | 5 |
| `atoms_s`, `atoms_q` | total atoms in all support / query graphs | ~30 × molecules |
| `bonds_s`, `bonds_q` | total bonds | ~60 × molecules |

---

## Part I — GoS v1 dimensions per step

### Input (per episode)

| Tensor | Shape | Where from |
|---|---|---|
| `support_batch.x` | `[atoms_s, 9]` | `Batch.from_data_list(support_graphs)` |
| `support_batch.edge_index` | `[2, bonds_s]` | PyG |
| `support_batch.edge_attr` | `[bonds_s, 3]` | PyG |
| `support_batch.batch` | `[atoms_s]` | graph-id per atom |
| `support_labels` | `[N·K]` (int64 in 0..N-1) | episode sampler |
| same for `query_batch` | — | |
| `query_labels` | `[Q]` | |

### ① MPNN encoder φ
`models/mpnn_encoder.py`

Every atom starts at `[atoms_s, 9]`, gets projected and then runs through 3 × GINEConv with bond features:

| Op | Input | Output |
|---|---|---|
| Linear atom-embed (9 → d) | `[atoms_s, 9]` | `[atoms_s, d]` |
| Linear bond-embed (3 → d) | `[bonds_s, 3]` | `[bonds_s, d]` |
| GINEConv × 3 + BN + ReLU + Dropout | `[atoms_s, d]` | `[atoms_s, d]` |
| global_mean_pool (atom → graph) | `[atoms_s, d]` | `[N·K, d]` |
| Linear project (d → d) | `[N·K, d]` | `[N·K, d]` |

Same for query → `[Q, d]`. Concatenate:

| Tensor | Shape |
|---|---|
| `h_s` | `[N·K, 128]` |
| `h_q` | `[Q, 128]` |
| `h_all = cat(h_s, h_q)` | `[M, 128]`, where `M = N·K + Q` |

### ② Pairwise affinity W = g(h_i, h_j)

`models/graph_of_shots.py::AffinityFunction`

| Variant | Math | Output |
|---|---|---|
| cosine | `normalize(h)·normalize(h).T` | `[M, M]` |
| bilinear | `h @ W_bil @ h.T`, `W_bil` learnable `[d, d]` | `[M, M]` |
| attention | `(W_q h)(W_k h)^T / √d`, `W_q, W_k: [d, d]` learnable | `[M, M]` |

Every entry `W_ij` is a scalar "how similar are molecules i and j".

### ③ kNN sparsification

`models/graph_of_shots.py::build_meta_graph`

```
W'_ii = -∞                          # mask self-loops
vals, idx = topk(W', k=k, dim=1)    # [M, k], [M, k]
```

Build undirected edges (each kept edge is added twice):

| Tensor | Shape |
|---|---|
| `edge_index` | `[2, 2·M·k]` |
| `edge_weight` | `[2·M·k]`, each ≥ 0 after ReLU+eps |

For 5w5s: M=41, k=5 → 410 directed edges.

### ④ Inject labels into node features

```
label_s = one_hot(support_labels, N)    # [N·K, N]
label_q = zeros(Q, N)                   # [Q, N]
z^(0) = cat( cat(h_s, label_s),
             cat(h_q, label_q) )         # [M, d + N]
```

For 5w5s: `z^(0): [41, 133]`.

### ⑤ Meta-GNN propagation (2-layer weighted GCN)

`models/graph_of_shots.py::MetaGNN`

| Layer | Input | Output | Detail |
|---|---|---|---|
| GCNConv₁ | `[M, d+N]`, `edge_index[2, E]`, `edge_weight[E]` | `[M, d]` | weighted message passing |
| BN + ReLU | `[M, d]` | `[M, d]` | |
| GCNConv₂ | `[M, d]`, edge_index, edge_weight | `[M, d]` | |
| BN | `[M, d]` | `[M, d]` | |

Output `z^(L): [M, 128]`. Split back:

| Tensor | Shape |
|---|---|
| `z_s = z^(L)[:N·K]` | `[N·K, 128]` |
| `z_q = z^(L)[N·K:]` | `[Q, 128]` |

### ⑥ Prototype head

```
prototypes = scatter_mean(z_s, support_labels, dim=0, dim_size=N)
# prototypes: [N, 128]

logits = -cdist(z_q, prototypes, p=2)   # [Q, N]
loss   = F.cross_entropy(logits, query_labels)
```

Predictions: `softmax(logits)` → `[Q, N]`.

---

## Part II — v2 changes, with full motivation

Every v2 knob is toggled via `--v2_*` flags (see `config.py`). Here we
explain **why** each exists, **what v1 fails at**, and **the dimension
delta from v1**.

### M1 · Residual prototype — safety net for the encoder signal

**What v1 did** (the failure mode)
> The prototype is computed only from the meta-GNN outputs `z^(L)`:
> `c^(v1)_n = mean{z^(L)_i : y_i = n}`.

This throws away the original encoder embedding `h_i`. If the meta-GNN
over-smooths, you lose the good signal from the encoder as well.

**Measured evidence**
- `ProtoNet 0.716 > GoS v1 0.572` at 5w5s → the meta-GNN *actively
  hurts* a representation ProtoNet can already classify.
- `A1_alpha0.0` (pure meta-GNN) collapses to **0.500** — chance.

**v2 fix**
```
c^(enc)_n = scatter_mean(h_s,  support_labels, dim=0, dim_size=N)  # [N, d]
c^(gnn)_n = scatter_mean(z_s,  support_labels, dim=0, dim_size=N)  # [N, d]

α         = sigmoid(self.alpha_logit)                # learnable scalar ∈ [0,1]
c_n       = α · c^(enc)_n  +  (1-α) · c^(gnn)_n      # [N, d]
```

*Also* mix the query representations correspondingly so classifier
operates in the same space:
```
z_q^(mix) = α · h_q  +  (1-α) · z_q^(gnn)             # [Q, d]
logits    = -cdist(z_q^(mix), c_n)                    # [Q, N]
```

**Why it works** (safety-net argument)
- `α → 1` recovers **ProtoNet exactly** (c_n = c^(enc)_n, z_q^(mix) = h_q).
  v2 therefore *cannot* be worse than ProtoNet.
- `α → 0` recovers v1.
- The learnable α lets the optimiser choose how much to trust the meta-GNN.

**Measured**: α settles around 0.5–0.7; A1 is flat in [0.3, 1.0] at ~0.71.

### M2 · 1-layer GAT — structurally prevent over-smoothing

**What v1 did**
> 2-layer weighted GCN. Each layer does `D^(-½) A D^(-½) X W`. Stacking
> two of these moves node features toward the stationary distribution
> of the random walk, which is the **global mean** of the graph.

**Why GCN over-smooths here**
- Meta-graph has ~40 nodes and kNN-degree 5–10, so 2 hops reach almost
  every node.
- The normalisation factor concentrates mass at high-degree nodes; one
  or two such hubs drag everyone toward the same point.

**Measured evidence**
- kNN sweep: k=3 > 5 > 10 > 100 (dense) — sparser beats denser, textbook
  over-smoothing signature.

**v2 fix**
```
# v1:  MetaGNN = [GCNConv(d+N → d),  GCNConv(d → d)]
# v2:  MetaGNN = [GATConv(d+N → d, heads=1, concat=False)]         # 1 layer
```
`GATConv` computes attention per node:
```
a_ij = softmax_j( LeakyReLU(a^T [W h_i || W h_j]) )
z_i  = σ( Σ_j a_ij · W h_j )
```

**Why GAT beats GCN here**
- `softmax_j` ⇒ each row of the attention matrix sums to 1 ⇒ no hub
  can dominate.
- 1 layer reaches 1 hop; with kNN-sparsified 40-node episodes that's
  almost every support from any query.
- Self-normalisation ⇒ **sub-linear growth of over-smoothing with
  depth**, so there's no pressure to go deeper.

**Dimension delta**
- `z^(0)` is still `[M, d+N]`.
- Output `z^(L)` is still `[M, d]` — just one GAT layer produces it
  instead of two GCN layers.

**Measured by A2**: GAT L=1 = **0.733** (best cell in 2×3 grid).

### M3 · Bipartite messaging — remove the noise source

**What v1 did**
> The meta-graph was symmetric: every kNN edge was added in both
> directions, so `query → support` and `query → query` messages flowed.

**Why that's bad**
- Query nodes have `[h_q ; 0]` — the label one-hot is all zeros by
  construction. So any message *originating* from a query node carries
  **no label signal** — it's pure noise.
- Query→query propagation especially: averaging noise with noise.

**v2 fix** (in `_apply_bipartite`)
```
# After building undirected edges (both directions), keep only those
# whose source is a support node:
keep_mask = edge_index[0] < N·K          # support nodes are indexed first
edge_index  = edge_index[:, keep_mask]   # [2, E_kept]
edge_weight = edge_weight[keep_mask]      # [E_kept]
```

Result:

| Edge type | v1 | v2 |
|---|:---:|:---:|
| S → S | ✓ | ✓ |
| S → Q | ✓ | ✓ |
| Q → S | ✓ | ✗ |
| Q → Q | ✓ | ✗ |

Supports refine each other and push labels to queries, but queries are
passive receivers.

**Dimension delta**
- `edge_index` goes from `[2, 2·M·k]` to approximately `[2, (N·K)·k + Q·0] ≈ [2, 2·N·K·k]`
  (both S-originated directed edges survive; Q-originated ones die).
- For 5w5s with k=5: 410 → 250 edges.

**Measured by A3**: bipartite **0.7274** vs full 0.7216 → **+0.6 pp**.

### M4 · Contrastive auxiliary loss — shape embedding geometry directly

**What v1 did**
> Only cross-entropy on classifier output:
> `L = F.cross_entropy(logits, query_labels)`.

**Why that's indirect**
- Gradient flows: `loss → logits → z_q & prototypes → z_s → encoder`.
- Nothing directly *pushes* same-class embeddings close; it's mediated
  through the prototype + distance computation.
- Very different embeddings can produce similar prototypes (averaging
  smooths individual variance).

**v2 fix** (InfoNCE on support-query pairs)
```
z_q_n = F.normalize(z_q, dim=1)          # [Q, d]
z_s_n = F.normalize(z_s, dim=1)          # [N·K, d]
sim   = (z_q_n @ z_s_n.T) / τ             # [Q, N·K], τ = 0.1

mask      = (query_labels[:,None] == support_labels[None,:]).float()   # [Q, N·K]
log_denom = torch.logsumexp(sim, dim=1)                                # [Q]
log_num   = torch.logsumexp(sim.masked_fill(mask == 0, -inf), dim=1)   # [Q]
L_con     = -(log_num - log_denom).mean()

L_total   = L_CE  +  λ · L_con           # λ ∈ [0, 1], default 0.5, optimal 0.1
```

**Why it helps**
- Each query's same-class supports are pulled together; other-class
  supports are pushed away.
- Works on the same embedding space classifier uses → aligned.
- Complements (not replaces) L_CE: L_CE cares about *which prototype
  wins*, L_con cares about *relative geometry*.

**Measured by A4**: λ=0.1 is the sweet spot (**0.7291**, +0.8 pp over λ=0).
Too much contrastive signal competes with CE and degrades.

### M5 · Self-supervised encoder pretraining — the heavy hitter

**What v1 did**
> Encoder was trained from scratch via episodic cross-entropy. Each
> episode provides ~16 query supervision signals — tiny data efficiency.

**Why that's a bottleneck**
- ProtoNet baseline hit 0.716 val AUC at 5w5s. When your simplest
  baseline is already at 0.7, the meta-learning strategy is not the
  bottleneck — the **encoder** is.
- Self-supervised pretraining can use **all 110k meta-train molecules**
  without labels, giving the encoder 3+ orders of magnitude more
  training signal.

**v2 fix** — MolCLR-style contrastive SSL (`models/ssl_pretrain.py`)

Before episodic training, run `pretrain_ssl.py`:
```
for molecule G in meta-train pool (110k mols):
    view_a = random_augment(G)         # drop 15% nodes / mask atoms / drop edges
    view_b = random_augment(G)
    z_a = projection_head(φ(view_a))    # [B, 128]
    z_b = projection_head(φ(view_b))    # [B, 128]
    # NT-Xent (SimCLR/MolCLR style): same molecule's two views = close;
    # other molecules in batch = far.
    L = InfoNCE(z_a, z_b, τ=0.1)
```

30 epochs, batch 256. Final loss 1.02 → 0.306
(`logs/v2_rerun/phase0_ssl.log`).

Then episodic training with `--ssl_ckpt checkpoints/ssl_encoder.pt`
loads those weights into φ.

**Why fine-tune and NOT freeze**
- SSL teaches φ "structural similarity" (atoms arranged like this have
  been seen together).
- The episodic task needs *assay-specific similarity* — two structurally
  different molecules can be active for the same assay.
- If you freeze after SSL, φ cannot adapt to the episodic objective.

**Measured by A5**
- scratch: **0.7046**
- SSL + fine-tune: **0.7266**  → **+2.2 pp**  (single biggest lever)
- SSL + frozen: **0.6036**  → **−10 pp**  (don't freeze!)

**Dimension delta**: none at inference — the encoder has the same
architecture. Only the initialization of its weights differs.

---

## Part III — v2 at a glance: side-by-side per-step

| Step | v1 | v2 | Change |
|---|---|---|---|
| 0. Encoder init | random | MolCLR-pretrained | **M5** |
| 1. Encoder forward | 3×GINEConv → Linear | same | — |
| 2. Affinity | cosine / bilinear / attention | same | — |
| 3. kNN | `edge_index [2, 2Mk]` | after M3: `[2, 2·N·K·k]` | **M3 bipartite mask** |
| 4. Label inject | `z^(0) = [M, d+N]` | same | — |
| 5. Meta-GNN | 2-layer weighted GCN | 1-layer GAT | **M2** |
| 6. Prototype | `c = mean(z_s)` | `c = α·mean(h_s) + (1−α)·mean(z_s)` | **M1 residual** |
| Loss | L_CE | L_CE + λ·L_InfoNCE | **M4** |

Composed: **v2 + SSL** gives +1.2 to +1.7 pp over ProtoNet on every
episodic configuration.  Numbers and log paths: [`results.md`](results.md).
