# Graph-of-Shots — Takeaways

*Few-shot molecular graph classification · 15-min talk · 4 MoleculeNet
datasets · 673 tasks · 405 / 134 / 134 split*

---

## 1. The problem is real

Bioassays in drug discovery have **10² – 10³** labeled molecules, and
most rows are missing. Vanilla supervised GNNs overfit and don't
transfer across assays. **Episodic meta-learning** is the natural fix:
train across many small tasks so the model adapts to new assays with
only K labeled molecules.

## 2. ProtoNet is a surprisingly strong starting point

On the merged MoleculeNet pool, bare **ProtoNet hits 0.716 val AUC** at
5-way 5-shot — higher than we expected. A clean metric-based baseline
(no persistent per-class parameters) turns out to be the toughest
competitor, harder to beat than MAML.

## 3. GoS v1 beat MAML but lost to ProtoNet — root cause: over-smoothing

Three independent signals pointed the same way:

| Signal | What we saw | What it means |
|---|---|---|
| k-NN sweep | k = 3 > 5 > 10 > dense | classic GCN over-smoothing |
| Baselines | ProtoNet 0.716 > GoS v1 0.572 | meta-GNN HURTS a good embedding |
| Affinity | cos ≈ bilinear ≈ attention (< 2 pp) | expressivity is not the bottleneck |

**The 2-layer weighted GCN was destroying the class separability the
encoder had already built**, and v1's prototype head used only the
meta-GNN output.

## 4. GoS v2: five targeted modifications — each fixes one failure mode

| Mod | Change | Role | Measured effect |
|---|---|---|---|
| **M1** | Residual prototype  `c = α·c_enc + (1−α)·c_gnn`, α learnable | safety net (α→1 recovers ProtoNet) | α = 0 → 0.500 collapse;  α ∈ [0.3, 1.0] all ≥ 0.71 |
| **M2** | 2-layer GCN → 1-layer GAT | prevent over-smoothing structurally | GAT L = 1  →  0.733  (best cell in 2×3 grid) |
| **M3** | Bipartite masking (cut query → * edges) | remove query-to-query noise | +0.6 pp over full meta-graph |
| **M4** | Contrastive aux loss  `L = L_CE + λ · L_InfoNCE` | shape embedding geometry | best λ = 0.1 (+0.8 pp over λ = 0) |
| **M5** | MolCLR-style SSL pretraining (110k mols, 30 epochs) | stronger encoder | **+2.2 pp** (0.705 → 0.727), the single biggest lever |

## 5. Composed v2 + SSL beats ProtoNet on every config

| Config | ProtoNet | GoS v1 | **v2 + SSL** | Δ |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 0.586 | 0.536 | **0.603** | **+1.7 pp** |
| 5-way 5-shot | 0.716 | 0.572 | **0.728** | **+1.2 pp** |
| 10-way 5-shot | 0.700 | 0.580 | **0.712** | **+1.2 pp** |

Meta-val ROC-AUC at best checkpoint. (Meta-test rerun pending —
`scripts/rerun_v2_main.sh` takes ~30 min on 8×A100.)

---

## Transferable lessons (for anyone doing few-shot + GNN)

1. **Shared classifier heads collapse under episodic training.**
   Class index "0" is a different assay every episode → contradictory
   supervision. Always use **prototype / metric heads**
   (scatter-mean → nearest).

2. **For task-level meta-graphs, sparsification beats expressivity.**
   Varying the affinity kernel (cos / bilinear / attention) moved us
   <2 pp; varying k in kNN moved us 4 pp. Keep the graph sparse
   (k = 3–5) and pick any reasonable affinity.

3. **One message-passing hop is usually enough.** Episodes have ~40
   nodes — more layers just over-smooth. Use **GAT** rather than GCN
   for self-normalisation.

4. **Query nodes carry no label → messages from them are noise.**
   Bipartite masking (keep S→S and S→Q, drop Q→*) is a cheap +0.6 pp.

5. **When your baseline is already 0.70, the bottleneck is the
   encoder.** Self-supervised pretraining on the unlabeled molecule
   pool gave us the biggest single gain (+2.2 pp). But **fine-tune**,
   don't freeze — frozen encoder dropped 10 pp.

6. **Residual safety nets are cheap insurance.** M1 guarantees v2
   can't lose to ProtoNet; everything else is upside. With 5
   composed mods that can each fail, a safety net makes the whole
   thing robust.

---

## What's still open

- **Meta-test numbers for v2** are pending. The current JSONs in
  `logs/v2_rerun/v2_*_test_results.json` are evaluating random-init
  models — see the WARNING lines in `logs/v2_rerun/eval.log`.
  `scripts/rerun_v2_main.sh` fixes this.
- **Contrastive-only objective** — could we skip the meta-graph
  entirely and just do SSL + ProtoNet? A1 shows α = 1 is already
  competitive.
- **Text-augmented encoder** — SMILES tokens or atom descriptions
  as an additional modality.
- **Cross-assay transfer** — explicit task-clustering instead of
  random splits.

---

**Code & logs:** <https://github.com/BDeMo/proj-mol>  ·  full docs in
[`docs/`](README.md).
