"""Meta-test evaluation: load checkpoint and evaluate over 1000+ episodes."""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import get_args
from utils import set_seed, compute_episode_auc
from data import load_all_datasets, split_tasks, EpisodeSampler
from train import build_model, run_episode


def main():
    args = get_args()

    # Allow overriding checkpoint path
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    extra_args, _ = parser.parse_known_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    print("=== Loading datasets ===")
    all_graphs, all_task_indices, all_task_ids = load_all_datasets(
        args.data_root, args.datasets, args.min_pos
    )

    # Split tasks
    splits = split_tasks(all_task_ids, args.train_ratio, args.val_ratio, args.seed)

    # Meta-test sampler
    test_sampler = EpisodeSampler(
        all_graphs, all_task_indices, splits["test"],
        args.n_way, args.k_shot, args.n_query, seed=args.seed + 2,
    )

    # Build and load model
    model = build_model(args).to(device)

    exp_name = f"{args.method}_{args.n_way}w{args.k_shot}s"
    if args.method == "gos":
        exp_name += f"_{args.affinity}_k{args.meta_k}"

    ckpt_path = extra_args.checkpoint or os.path.join(
        args.save_dir, f"{exp_name}_best.pt"
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from episode {ckpt.get('episode', '?')} "
              f"(val AUC: {ckpt.get('val_auc', '?'):.4f})")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}, evaluating random model.")

    # Evaluate
    print(f"\n=== Meta-Test Evaluation ({args.episodes_test} episodes) ===")
    model.eval()
    aucs = []
    accs = []

    with torch.no_grad():
        for _ in tqdm(range(args.episodes_test), desc="Evaluating"):
            episode = test_sampler.sample_episode()
            logits, loss, ql = run_episode(model, episode, args)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            labels = ql.cpu().numpy()

            auc = compute_episode_auc(probs, labels, args.n_way)
            acc = (logits.argmax(dim=1) == ql).float().mean().item()

            aucs.append(auc)
            accs.append(acc)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ci95 = 1.96 * std_auc / np.sqrt(len(aucs))
    mean_acc = np.mean(accs)
    ci95_acc = 1.96 * np.std(accs) / np.sqrt(len(accs))

    print(f"\n{'='*50}")
    print(f"Method: {args.method} | {args.n_way}-way {args.k_shot}-shot")
    if args.method == "gos":
        print(f"Affinity: {args.affinity} | meta_k: {args.meta_k}")
    print(f"Datasets: {args.datasets}")
    print(f"Episodes: {args.episodes_test}")
    print(f"ROC-AUC: {mean_auc:.4f} ± {ci95:.4f} (std: {std_auc:.4f})")
    print(f"Accuracy: {mean_acc:.4f} ± {ci95_acc:.4f}")
    print(f"{'='*50}")

    # Save results
    results = {
        "method": args.method,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "datasets": args.datasets,
        "affinity": args.affinity if args.method == "gos" else None,
        "meta_k": args.meta_k if args.method == "gos" else None,
        "episodes": args.episodes_test,
        "roc_auc_mean": mean_auc,
        "roc_auc_std": std_auc,
        "roc_auc_ci95": ci95,
        "accuracy_mean": mean_acc,
        "accuracy_ci95": ci95_acc,
    }
    results_path = os.path.join(args.log_dir, f"{exp_name}_test_results.json")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
