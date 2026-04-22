# Graph-of-Shots: Few-Shot Molecular Graph Classification

Course project for **Few-Shot Graph Machine Learning** (Spring 2026).

**Team**: Wenbo Huang, Junchi Yan, Yiqun Yin, Xinyue Xu, Mingjia Shi

---

## TL;DR — Takeaways

*Few-shot molecular graph classification · 4 MoleculeNet datasets · 673
tasks · 405 / 134 / 134 split*

### 1. The problem is real
Bioassays have **10² – 10³** labeled molecules, and most rows are missing. Vanilla supervised GNNs overfit and don't transfer across assays. Episodic meta-learning is the natural fix.

### 2. ProtoNet is a surprisingly strong starting point
Bare ProtoNet hits **0.716 val AUC** at 5-way 5-shot — a clean
metric-based baseline turned out to be our toughest competitor.

### 3. GoS v1 beat MAML but lost to ProtoNet — over-smoothing was the root cause

| Signal | What we saw | What it means |
|---|---|---|
| k-NN sweep | k = 3 > 5 > 10 > dense | classic GCN over-smoothing |
| Baselines | ProtoNet 0.716 > GoS v1 0.572 | meta-GNN **hurts** a good embedding |
| Affinity | cos ≈ bilinear ≈ attention (< 2 pp) | expressivity is **not** the bottleneck |

### 4. GoS v2 — five targeted modifications

| Mod | Change | Role | Effect |
|---|---|---|---|
| **M1** | Residual prototype  `c = α·c_enc + (1−α)·c_gnn` | safety net (α→1 recovers ProtoNet) | α=0 collapses to 0.500;  α ∈ [0.3, 1.0] all ≥ 0.71 |
| **M2** | 2-layer GCN → 1-layer GAT | prevent over-smoothing structurally | GAT L=1 wins at 0.733 |
| **M3** | Bipartite masking (cut query→* edges) | remove query-to-query noise | +0.6 pp |
| **M4** | Contrastive aux loss, λ · L_InfoNCE | shape embedding geometry | best λ = 0.1 (+0.8 pp) |
| **M5** | MolCLR-style SSL pretraining | stronger encoder | **+2.2 pp**, biggest single lever |

### 5. Composed v2 + SSL beats ProtoNet on every config

| Config | ProtoNet | GoS v1 | **v2 + SSL** | Δ |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 0.586 | 0.536 | **0.603** | **+1.7 pp** |
| 5-way 5-shot | 0.716 | 0.572 | **0.728** | **+1.2 pp** |
| 10-way 5-shot | 0.700 | 0.580 | **0.712** | **+1.2 pp** |

Meta-val ROC-AUC at best checkpoint.  (Meta-test rerun pending —
`scripts/rerun_v2_main.sh`, ~30 min on 8×A100.)

### Transferable lessons (for anyone doing few-shot + GNN)

1. **Shared classifier heads collapse under episodic training.** Class
   index "0" is a different assay every episode → contradictory
   supervision. Always use **prototype / metric heads**.
2. **For task-level meta-graphs, sparsification beats expressivity.**
   Affinity choice moved us <2 pp; kNN k moved us 4 pp. Keep graphs
   sparse (k = 3–5).
3. **One hop is usually enough.** Episodes have ~40 nodes — more layers
   over-smooth. Use **GAT** rather than GCN for self-normalisation.
4. **Query → * edges are noise.** Bipartite masking is a cheap +0.6 pp.
5. **When your baseline is already 0.70, the bottleneck is the
   encoder.** SSL pretraining gave us the biggest gain — but
   **fine-tune, don't freeze** (frozen encoder dropped 10 pp).
6. **Residual safety nets are cheap insurance.** M1 guarantees v2
   can't lose to ProtoNet; everything else is upside.

---

## Documentation

| File | What it has |
|---|---|
| [`docs/takeaways.md`](docs/takeaways.md) | One-page project summary (the content above) |
| [`docs/repo_structure.md`](docs/repo_structure.md) | Every path + what lives there |
| [`docs/results.md`](docs/results.md) | Every number with its source log file |
| [`docs/reproduce.md`](docs/reproduce.md) | Step-by-step commands for a fresh clone |
| [`docs/changelog.md`](docs/changelog.md) | Commit log with bug-fix context |

## Project structure

```
.
├── config.py             # All CLI arguments
├── utils.py              # Seeding, metrics, build_exp_name
├── train.py              # Episodic training (auto-resume, SIGINT flush)
├── evaluate.py           # 1000-episode meta-test evaluation
├── pretrain_ssl.py       # M5 SSL pretraining entry point
├── data/                 # MoleculeNet loader + episodic sampler
├── models/
│   ├── mpnn_encoder.py      # 3 × GINEConv + global mean pool
│   ├── proto_net.py         # Prototypical Network baseline
│   ├── maml.py              # FOMAML baseline
│   ├── graph_of_shots.py    # GoS v1 (meta-graph + meta-GNN)
│   ├── graph_of_shots_v2.py # GoS v2 (M1-M4 toggles)
│   └── ssl_pretrain.py      # M5 MolCLR-style SSL
├── scripts/
│   ├── run_all_v2_parallel.sh       # multi-GPU v2 runner
│   ├── rerun_v2_main.sh             # re-run Phase-1 with --exp_name
│   └── rerun_broken_ablations.sh    # re-run A1–A5 ablations
├── logs/
│   ├── v1_stages/            # v1 full-pipeline stage logs
│   ├── v1_final_eval/        # v1 final 1000-ep evaluation
│   ├── v2_initial/           # v2 first run (collision bug)
│   └── v2_rerun/             # v2 current best; v2+SSL beat ProtoNet
├── presentation/             # 13-slide deck + figures + formulas
├── docs/                     # reference documentation
├── requirements.txt
└── .gitignore                # checkpoints/ + data_cache/ ignored
```

## Installation

```bash
git clone https://github.com/BDeMo/proj-mol.git
cd proj-mol
pip install -r requirements.txt
```

**Dependencies**: PyTorch (≥ 2.0), PyTorch Geometric (≥ 2.4), RDKit,
scikit-learn, numpy, matplotlib, tqdm.

## Quick start

### Train and evaluate v2 + SSL end-to-end

```bash
# one command, all 26 jobs across all GPUs, ~1.5 h on 8×A100
bash scripts/run_all_v2_parallel.sh
```

### Manual path

```bash
# 1. SSL pretrain once (~15 min)
python pretrain_ssl.py --epochs 30 --out checkpoints/ssl_encoder.pt

# 2. v2 on 5-way 5-shot
python train.py --method gos_v2 --exp_name v2_5w5s_ssl \
    --datasets tox21 toxcast muv sider --n_way 5 --k_shot 5 \
    --affinity cosine --meta_k 5 --refine_steps 2 \
    --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
    --v2_contrastive_lambda 0.5 \
    --ssl_ckpt checkpoints/ssl_encoder.pt \
    --episodes_train 30000 --patience 20

# 3. evaluate on 1000 meta-test episodes
python evaluate.py --method gos_v2 --exp_name v2_5w5s_ssl \
    --datasets tox21 toxcast muv sider --n_way 5 --k_shot 5 \
    --episodes_test 1000 --ssl_ckpt checkpoints/ssl_encoder.pt
```

More commands in [`docs/reproduce.md`](docs/reproduce.md).

## Key CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `gos` | One of `proto`, `maml`, `gos`, `gos_v2` |
| `--exp_name` | *auto* | Unique checkpoint tag (auto-derived from config if omitted) |
| `--datasets` | `tox21` | Subset of `tox21 toxcast muv sider` |
| `--n_way` / `--k_shot` | 5 / 1 | Episodic config |
| `--affinity` | `cosine` | `cosine`, `bilinear`, `attention` |
| `--meta_k` | 5 | kNN sparsification level |
| `--refine_steps` | 2 | Iterative refinement rounds |
| `--v2_alpha_init` | 0.7 | **M1** residual coefficient initial value |
| `--v2_gnn_type` | `gat` | **M2** `gat` or `gcn` |
| `--v2_bipartite` / `--v2_no_bipartite` | True | **M3** edge masking |
| `--v2_contrastive_lambda` | 0.5 | **M4** InfoNCE weight |
| `--ssl_ckpt PATH` | *none* | **M5** path to SSL-pretrained encoder |
| `--freeze_encoder` | False | Freeze encoder during episodic training |
| `--episodes_train` | 30000 | Max training episodes |
| `--patience` | 20 | Early-stop patience in eval rounds |
| `--resume` / `--no_resume` | True | Auto-resume from `<exp_name>_last.pt` |

## Presentation

- `presentation/Graph-of-Shots.pptx` — 13-slide deck (15 min talk) with full speaker notes on every slide
- `presentation/figures/` — all figures embedded in the deck (`fig_v2_main.png`, `fig_v2_A1.png` through `fig_v2_A5.png`, plus legacy v1 figures)
- `presentation/formulas/` — individual LaTeX-rendered formula PNGs (one per equation)

## Benchmarks used

| Dataset | Molecules | Tasks | Domain |
|---------|----------:|------:|--------|
| Tox21   | 7,823     | 12    | Toxicology |
| ToxCast | 8,579     | 617   | Toxicology |
| MUV     | 93,087    | 17    | Biophysics |
| SIDER   | 1,427     | 27    | Side effects |

Labels are sparse and binary per assay column. We split at the task
level with strict label disjointness.

## What's still open

- **Meta-test AUCs for v2** — the JSONs in `logs/v2_rerun/v2_*_test_results.json` are evaluating random-init models. `scripts/rerun_v2_main.sh` produces the correct numbers.
- **Contrastive-only objective** — could we skip the meta-graph entirely? A1 suggests α=1 is already competitive.
- **Text-augmented encoder** — SMILES tokens or atom descriptions as additional modality.
- **Cross-assay transfer** — explicit task-clustering instead of random splits.

---

Code: <https://github.com/BDeMo/proj-mol>
