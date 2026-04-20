import argparse


def get_args():
    p = argparse.ArgumentParser(description="Graph-of-Shots: Few-Shot Molecular Graph Classification")

    # Data
    p.add_argument("--datasets", nargs="+", default=["tox21"],
                   choices=["tox21", "toxcast", "muv", "sider"])
    p.add_argument("--data_root", type=str, default="./data_cache")
    p.add_argument("--min_pos", type=int, default=16,
                   help="Minimum positive examples per task to include")

    # Meta-split ratios
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)

    # Episode config
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--n_query", type=int, default=16)

    # Encoder
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--gin_layers", type=int, default=3)
    p.add_argument("--gin_hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    # Graph-of-Shots
    p.add_argument("--affinity", type=str, default="cosine",
                   choices=["cosine", "bilinear", "attention"])
    p.add_argument("--meta_k", type=int, default=5,
                   help="k for kNN sparsification of meta-graph")
    p.add_argument("--meta_gnn_layers", type=int, default=2)
    p.add_argument("--refine_steps", type=int, default=2,
                   help="Iterative refinement rounds in Graph-of-Shots")

    # Method
    p.add_argument("--method", type=str, default="gos",
                   choices=["proto", "maml", "gos"])

    # MAML-specific
    p.add_argument("--maml_inner_lr", type=float, default=0.01)
    p.add_argument("--maml_inner_steps", type=int, default=5)

    # Training
    p.add_argument("--episodes_train", type=int, default=30000)
    p.add_argument("--episodes_val", type=int, default=500)
    p.add_argument("--episodes_test", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (in eval rounds)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--log_dir", type=str, default="./logs")

    return p.parse_args()
