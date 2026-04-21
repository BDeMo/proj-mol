"""Stand-alone SSL pretraining entry point (M5).

Loads the merged molecule pool, runs MolCLR-style contrastive pretraining
on the MPNN encoder, and saves the state_dict.  The resulting checkpoint
can be passed to train.py via --ssl_ckpt for v2 episodic training.
"""
import argparse
import os
import torch

from models import MPNNEncoder, pretrain_encoder
from data import load_all_datasets
from data.task_splitter import split_tasks
from utils import set_seed, ensure_dir


def main():
    p = argparse.ArgumentParser("SSL pretraining for GoS v2 encoder")
    p.add_argument("--datasets", nargs="+",
                   default=["tox21", "toxcast", "muv", "sider"])
    p.add_argument("--data_root", type=str, default="./data_cache")
    p.add_argument("--min_pos", type=int, default=16)
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--n_way", type=int, default=5,
                   help="only used for split sizing")

    # Encoder
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--gin_layers", type=int, default=3)
    p.add_argument("--gin_hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    # SSL
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)

    # IO
    p.add_argument("--meta_split", type=str, default="train",
                   choices=["train", "all"],
                   help="use only meta-train molecules (no label leakage) or all")
    p.add_argument("--out", type=str, default="./checkpoints/ssl_encoder.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    ensure_dir(os.path.dirname(args.out) or ".")

    print("=== Loading molecules ===")
    all_graphs, task_indices, task_ids = load_all_datasets(
        args.data_root, args.datasets, args.min_pos)

    # To prevent label leakage, restrict pretraining pool to molecules
    # that appear ONLY in meta-train tasks (if meta_split == "train").
    pool = all_graphs
    if args.meta_split == "train":
        splits = split_tasks(
            task_ids, args.train_ratio, args.val_ratio, args.seed,
            min_per_split=args.n_way,
        )
        train_tasks = set(splits["train"])
        val_test_tasks = set(splits["val"]) | set(splits["test"])

        train_mols = set()
        other_mols = set()
        for tid, idx in task_indices.items():
            target = train_mols if tid in train_tasks else other_mols
            target.update(idx["pos"] + idx["neg"])
        # only molecules that never appear in val/test positives stay in the pool
        safe = train_mols - other_mols
        pool = [all_graphs[i] for i in sorted(safe)]
        print(f"Meta-train-only SSL pool: {len(pool)} molecules "
              f"(out of {len(all_graphs)} total).")

    encoder = MPNNEncoder(
        in_dim=9, hidden_dim=args.gin_hidden, out_dim=args.embed_dim,
        num_layers=args.gin_layers, dropout=args.dropout,
    )

    pretrain_encoder(
        encoder, pool,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        temperature=args.temperature, device=device, log_every=50,
    )

    torch.save(encoder.state_dict(), args.out)
    print(f"\nSaved pretrained encoder to {args.out}")


if __name__ == "__main__":
    main()
