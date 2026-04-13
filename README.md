# Graph-of-Shots: Few-Shot Molecular Graph Classification

Course project for **Few-Shot Graph Machine Learning** (Spring 2026).

**Team**: Wenbo Huang, Junchi Yan, Yiqun Yin, Xinyue Xu, Mingjia Shi

## Introduction

Graph-of-Shots is a few-shot molecular classification method that introduces explicit relational modeling among molecular instances within each episode. Unlike traditional instance-centric approaches (ProtoNet, MAML) that embed molecules independently, Graph-of-Shots constructs a **task-level meta-graph** where nodes represent molecular instances and edges encode pairwise structural affinities. A meta-GNN then propagates label information from labeled support molecules to unlabeled queries through these relational pathways.

### Key Ideas

- **Meta-graph construction**: Given molecular embeddings from an MPNN encoder, build a graph connecting all molecules (support + query) in an episode. Edges are weighted by pairwise affinity (cosine, bilinear, or attention).
- **kNN sparsification**: Retain only the k strongest connections per node to reduce noise.
- **Meta-GNN reasoning**: Run message passing over the meta-graph to diffuse support labels to query nodes, then classify queries.
- **End-to-end training**: The molecular encoder, affinity function, and meta-GNN are jointly optimized.

### Benchmarks

Evaluated on four MoleculeNet datasets:
| Dataset  | Molecules | Tasks | Domain       |
|----------|-----------|-------|--------------|
| Tox21    | 7,831     | 12    | Toxicology   |
| ToxCast  | 8,576     | 617   | Toxicology   |
| MUV      | 93,087    | 17    | Biophysics   |
| SIDER    | 1,427     | 27    | Side effects |

## Project Structure

```
.
├── config.py                  # All command-line arguments
├── utils.py                   # Seeding, metrics, utilities
├── train.py                   # Episodic training loop
├── evaluate.py                # Meta-test evaluation
├── ablation.py                # Ablation study driver
├── data/
│   ├── molnet_loader.py       # MoleculeNet dataset loading
│   ├── task_splitter.py       # Meta-train/val/test split
│   └── episode_sampler.py     # N-way K-shot episodic sampler
├── models/
│   ├── mpnn_encoder.py        # GIN-based molecular encoder
│   ├── proto_net.py           # Prototypical Network baseline
│   ├── maml.py                # FOMAML baseline
│   └── graph_of_shots.py      # Graph-of-Shots (core method)
├── scripts/
│   ├── stage1_data_pipeline.sh    # Data loading & verification
│   ├── stage2_baselines.sh        # Train ProtoNet & MAML
│   ├── stage3_graph_of_shots.sh   # Train Graph-of-Shots
│   ├── stage4_evaluate.sh         # Evaluation & ablations
│   └── run_all.sh                 # Full pipeline
└── requirements.txt
```

## Installation

```bash
# Create environment (recommended)
conda create -n gos python=3.10 -y
conda activate gos

# Install dependencies
pip install -r requirements.txt
```

**Dependencies**: PyTorch (>=2.0), PyTorch Geometric (>=2.4), RDKit, scikit-learn, numpy, matplotlib, tqdm.

## Tutorial: Quick Start

### 1. Verify Data Pipeline

Load datasets, inspect task statistics, and test episodic sampling:

```bash
bash scripts/stage1_data_pipeline.sh
```

Or in Python:

```python
from data import load_all_datasets, split_tasks, EpisodeSampler

# Load Tox21
graphs, task_indices, task_ids = load_all_datasets('./data_cache', ['tox21'])

# Split tasks into meta-train/val/test (60/20/20)
splits = split_tasks(task_ids, 0.6, 0.2, seed=42)

# Create a 5-way 1-shot sampler on meta-train tasks
sampler = EpisodeSampler(
    graphs, task_indices, splits['train'],
    n_way=5, k_shot=1, n_query=20, seed=42
)

# Sample an episode
episode = sampler.sample_episode()
print(f"Support: {episode.support_batch.num_graphs} graphs")
print(f"Query: {episode.query_batch.num_graphs} graphs")
```

### 2. Train a Baseline

```bash
# Train Prototypical Network (5-way 1-shot on Tox21)
python train.py --method proto --datasets tox21 --n_way 5 --k_shot 1

# Train MAML
python train.py --method maml --datasets tox21 --n_way 5 --k_shot 1
```

### 3. Train Graph-of-Shots

```bash
# Default: cosine affinity, meta_k=5
python train.py --method gos --datasets tox21 --n_way 5 --k_shot 5

# With bilinear affinity and denser meta-graph
python train.py --method gos --datasets tox21 --affinity bilinear --meta_k 10
```

### 4. Evaluate

```bash
# Evaluate on 1000 meta-test episodes
python evaluate.py --method gos --datasets tox21 --n_way 5 --k_shot 5 --episodes_test 1000
```

### 5. Run Ablation Studies

```bash
# Run all ablations (affinity, k-NN, sample efficiency)
python ablation.py --suite all --datasets tox21

# Or individual suites
python ablation.py --suite affinity --datasets tox21
python ablation.py --suite knn --datasets tox21
python ablation.py --suite sample --datasets tox21
```

## Usage Reference

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `gos` | Method: `proto`, `maml`, or `gos` |
| `--datasets` | `tox21` | Dataset(s): `tox21`, `toxcast`, `muv`, `sider` |
| `--n_way` | `5` | Number of classes per episode |
| `--k_shot` | `1` | Support examples per class |
| `--n_query` | `16` | Total query examples per episode |
| `--affinity` | `cosine` | Affinity function: `cosine`, `bilinear`, `attention` |
| `--meta_k` | `5` | kNN sparsification level for meta-graph |
| `--embed_dim` | `128` | Molecular embedding dimension |
| `--gin_layers` | `3` | Number of GINConv layers in encoder |
| `--episodes_train` | `10000` | Training episodes |
| `--episodes_test` | `1000` | Test episodes for evaluation |
| `--lr` | `0.001` | Learning rate |
| `--patience` | `10` | Early stopping patience (eval rounds) |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### Stage Scripts

All scripts accept environment variables for configuration:

```bash
# Custom dataset and device
DATASETS="tox21 sider" DEVICE=cpu bash scripts/stage2_baselines.sh

# Custom episode config
N_WAY=10 K_SHOT=5 bash scripts/stage3_graph_of_shots.sh

# Run everything end-to-end
bash scripts/run_all.sh "tox21"
```

### Output Structure

```
checkpoints/           # Saved model checkpoints
  proto_5w1s_best.pt
  maml_5w1s_best.pt
  gos_5w1s_cosine_k5_best.pt
logs/                  # Training history (JSON)
  proto_5w1s_history.json
  gos_5w5s_cosine_k5_test_results.json
ablation_results/      # Ablation plots and summary
  convergence_main.png
  sample_efficiency.png
  ablation_summary.json
```

## Method Details

### Molecular Encoder (MPNN)

A 3-layer Graph Isomorphism Network (GIN) with global mean pooling. Each GIN layer wraps a 2-layer MLP with batch normalization. Maps variable-size molecular graphs to fixed 128-d embeddings.

### Prototypical Network Baseline

Computes class prototypes as the mean embedding of support samples per class. Query molecules are classified by negative Euclidean distance to each prototype.

### MAML Baseline

First-order MAML (FOMAML): performs 5 inner-loop SGD steps on the support set to adapt the encoder + linear classifier, then evaluates on the query set. Meta-gradients flow through the parameter cloning operation.

### Graph-of-Shots

1. **Encode**: MPNN maps all episode molecules to embeddings.
2. **Construct meta-graph**: Compute pairwise affinities, apply kNN sparsification.
3. **Label conditioning**: Concatenate one-hot labels to support node features (zeros for queries).
4. **Meta-GNN**: 2-layer GCN with edge weights propagates information.
5. **Classify**: Linear head on query node representations.

Three affinity functions:
- **Cosine**: Parameter-free, `w_ij = cos(h_i, h_j)`
- **Bilinear**: Learned, `w_ij = h_i^T W h_j`
- **Attention**: Scaled dot-product, `w_ij = (W_q h_i)^T (W_k h_j) / sqrt(d)`
