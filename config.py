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
                   choices=["proto", "maml", "gos", "gos_v2"])

    # ── GoS v2 toggles ───────────────────────────────────────────────
    p.add_argument("--v2_alpha_init", type=float, default=0.7,
                   help="M1: residual prototype coefficient initial value")
    p.add_argument("--v2_gnn_type", type=str, default="gat",
                   choices=["gcn", "gat"],
                   help="M2: meta-GNN backbone (gat = shallower / self-normalising)")
    p.add_argument("--v2_bipartite", action="store_true",
                   help="M3: drop edges originating from query nodes")
    p.add_argument("--v2_no_bipartite", dest="v2_bipartite",
                   action="store_false")
    p.set_defaults(v2_bipartite=True)
    p.add_argument("--v2_contrastive_lambda", type=float, default=0.5,
                   help="M4: weight of InfoNCE auxiliary loss (0 disables)")
    p.add_argument("--v2_contrastive_temp", type=float, default=0.1,
                   help="M4: temperature for InfoNCE")
    p.add_argument("--ssl_ckpt", type=str, default=None,
                   help="M5: path to SSL-pretrained encoder state_dict")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="freeze encoder during episodic training")

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

    # Resume
    p.add_argument("--resume", action="store_true",
                   help="auto-resume from ./<save_dir>/<exp_name>_last.pt if present")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="explicit resume checkpoint path (overrides auto)")

    return p.parse_args()
