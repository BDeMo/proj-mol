"""Episodic training loop for all methods (ProtoNet, MAML, Graph-of-Shots)."""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import get_args
from utils import set_seed, compute_episode_auc, ensure_dir
from data import load_all_datasets, split_tasks, EpisodeSampler
from models import (
    MPNNEncoder, PrototypicalNetwork, MAMLClassifier,
    GraphOfShots, GraphOfShotsV2,
)


def build_model(args):
    """Build the model based on the selected method."""
    encoder = MPNNEncoder(
        in_dim=9,
        hidden_dim=args.gin_hidden,
        out_dim=args.embed_dim,
        num_layers=args.gin_layers,
        dropout=args.dropout,
    )
    if args.method == "proto":
        model = PrototypicalNetwork(encoder)
    elif args.method == "maml":
        model = MAMLClassifier(
            encoder, args.n_way,
            inner_lr=args.maml_inner_lr,
            inner_steps=args.maml_inner_steps,
        )
    elif args.method == "gos":
        model = GraphOfShots(
            encoder,
            affinity_method=args.affinity,
            meta_k=args.meta_k,
            n_way=args.n_way,
            meta_gnn_layers=args.meta_gnn_layers,
            refine_steps=args.refine_steps,
        )
    elif args.method == "gos_v2":
        model = GraphOfShotsV2(
            encoder,
            affinity_method=args.affinity,
            meta_k=args.meta_k,
            n_way=args.n_way,
            meta_gnn_layers=args.meta_gnn_layers,
            refine_steps=args.refine_steps,
            residual_alpha_init=args.v2_alpha_init,
            meta_gnn_type=args.v2_gnn_type,
            bipartite=args.v2_bipartite,
            contrastive_lambda=args.v2_contrastive_lambda,
            contrastive_temp=args.v2_contrastive_temp,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # M5: load SSL-pretrained encoder weights if provided
    if getattr(args, "ssl_ckpt", None):
        import os
        if os.path.exists(args.ssl_ckpt):
            state = torch.load(args.ssl_ckpt, map_location="cpu",
                               weights_only=False)
            model.encoder.load_state_dict(state)
            print(f"Loaded SSL-pretrained encoder from {args.ssl_ckpt}")
        else:
            print(f"WARNING: SSL ckpt not found: {args.ssl_ckpt}")
    if getattr(args, "freeze_encoder", False):
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen.")
    return model


def run_episode(model, episode, args):
    """Run a single episode through the model.

    The actual n_way is inferred from episode labels (the sampler may cap it
    when a split has fewer eligible tasks than args.n_way).

    Returns:
        logits: [Q, n_way] tensor.
        loss: scalar tensor.
        ql: query labels tensor.
        actual_n_way: int, the number of ways in this episode.
    """
    device = next(model.parameters()).device
    sb = episode.support_batch.to(device)
    sl = episode.support_labels.to(device)
    qb = episode.query_batch.to(device)
    ql = episode.query_labels.to(device)

    actual_n_way = int(sl.max().item()) + 1
    logits, loss = model(sb, sl, qb, ql, actual_n_way)
    return logits, loss, ql, actual_n_way


def evaluate(model, sampler, args, num_episodes: int):
    """Evaluate model over multiple episodes.

    Returns:
        mean_auc, ci95, mean_loss, mean_acc
    """
    model.eval()
    aucs = []
    losses = []
    accs = []

    with torch.no_grad():
        for _ in range(num_episodes):
            episode = sampler.sample_episode()
            logits, loss, ql, actual_n_way = run_episode(model, episode, args)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            labels = ql.cpu().numpy()

            auc = compute_episode_auc(probs, labels, actual_n_way)
            acc = (logits.argmax(dim=1) == ql).float().mean().item()

            aucs.append(auc)
            losses.append(loss.item())
            accs.append(acc)

    mean_auc = np.mean(aucs)
    ci95 = 1.96 * np.std(aucs) / np.sqrt(len(aucs))
    return mean_auc, ci95, np.mean(losses), np.mean(accs)


def main():
    args = get_args()
    set_seed(args.seed)
    ensure_dir(args.save_dir)
    ensure_dir(args.log_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Method: {args.method} | {args.n_way}-way {args.k_shot}-shot")

    # 1. Load data
    print("\n=== Loading datasets ===")
    all_graphs, all_task_indices, all_task_ids = load_all_datasets(
        args.data_root, args.datasets, args.min_pos
    )

    # 2. Split tasks (ensure each split has at least n_way tasks)
    print("\n=== Splitting tasks ===")
    splits = split_tasks(
        all_task_ids, args.train_ratio, args.val_ratio, args.seed,
        min_per_split=args.n_way,
    )

    # 3. Create episode samplers
    train_sampler = EpisodeSampler(
        all_graphs, all_task_indices, splits["train"],
        args.n_way, args.k_shot, args.n_query, seed=args.seed,
    )
    val_sampler = EpisodeSampler(
        all_graphs, all_task_indices, splits["val"],
        args.n_way, args.k_shot, args.n_query, seed=args.seed + 1,
    )

    # 4. Build model
    model = build_model(args).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 5. Training loop
    print("\n=== Training ===")
    best_val_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "val_loss": [], "val_acc": []}

    exp_name = f"{args.method}_{args.n_way}w{args.k_shot}s"
    if args.method == "gos":
        exp_name += f"_{args.affinity}_k{args.meta_k}"

    pbar = tqdm(range(1, args.episodes_train + 1), desc="Training")
    running_loss = 0.0

    for ep in pbar:
        model.train()
        episode = train_sampler.sample_episode()
        logits, loss, _, _ = run_episode(model, episode, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()

        if ep % args.eval_every == 0:
            avg_train_loss = running_loss / args.eval_every
            running_loss = 0.0

            val_auc, val_ci, val_loss, val_acc = evaluate(
                model, val_sampler, args, args.episodes_val
            )

            history["train_loss"].append(avg_train_loss)
            history["val_auc"].append(val_auc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            pbar.set_postfix({
                "loss": f"{avg_train_loss:.4f}",
                "val_auc": f"{val_auc:.4f}±{val_ci:.4f}",
                "val_acc": f"{val_acc:.4f}",
            })

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                ckpt_path = os.path.join(args.save_dir, f"{exp_name}_best.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "episode": ep,
                    "val_auc": val_auc,
                }, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at episode {ep} "
                          f"(best val AUC: {best_val_auc:.4f})")
                    break

    # Save training history
    hist_path = os.path.join(args.log_dir, f"{exp_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}")
    print(f"Checkpoint: {os.path.join(args.save_dir, f'{exp_name}_best.pt')}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    main()
