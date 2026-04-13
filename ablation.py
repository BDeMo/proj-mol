"""Ablation study driver: runs multiple configurations and collects results."""

import os
import subprocess
import sys
import json
import itertools
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Experiment Configurations ──────────────────────────────────────────────

MAIN_EXPERIMENTS = [
    # Baselines: ProtoNet
    {"method": "proto", "n_way": 5, "k_shot": 1},
    {"method": "proto", "n_way": 5, "k_shot": 5},
    {"method": "proto", "n_way": 10, "k_shot": 5},
    # Baselines: MAML
    {"method": "maml", "n_way": 5, "k_shot": 1},
    {"method": "maml", "n_way": 5, "k_shot": 5},
    {"method": "maml", "n_way": 10, "k_shot": 5},
    # Graph-of-Shots (cosine)
    {"method": "gos", "n_way": 5, "k_shot": 1, "affinity": "cosine"},
    {"method": "gos", "n_way": 5, "k_shot": 5, "affinity": "cosine"},
    {"method": "gos", "n_way": 10, "k_shot": 5, "affinity": "cosine"},
]

AFFINITY_ABLATION = [
    {"method": "gos", "n_way": 5, "k_shot": 5, "affinity": "cosine"},
    {"method": "gos", "n_way": 5, "k_shot": 5, "affinity": "bilinear"},
    {"method": "gos", "n_way": 5, "k_shot": 5, "affinity": "attention"},
]

KNN_ABLATION = [
    {"method": "gos", "n_way": 5, "k_shot": 5, "meta_k": 3},
    {"method": "gos", "n_way": 5, "k_shot": 5, "meta_k": 5},
    {"method": "gos", "n_way": 5, "k_shot": 5, "meta_k": 10},
    {"method": "gos", "n_way": 5, "k_shot": 5, "meta_k": 100},  # ~dense
]

SAMPLE_EFFICIENCY = [
    {"method": "gos", "n_way": 5, "k_shot": 1},
    {"method": "gos", "n_way": 5, "k_shot": 2},
    {"method": "gos", "n_way": 5, "k_shot": 5},
    {"method": "gos", "n_way": 5, "k_shot": 10},
    # Baselines for comparison
    {"method": "proto", "n_way": 5, "k_shot": 1},
    {"method": "proto", "n_way": 5, "k_shot": 2},
    {"method": "proto", "n_way": 5, "k_shot": 5},
    {"method": "proto", "n_way": 5, "k_shot": 10},
]


def config_to_args(config: Dict[str, Any], datasets: List[str]) -> List[str]:
    """Convert a config dict to command-line arguments."""
    args = ["--datasets"] + datasets
    for key, val in config.items():
        args.extend([f"--{key}", str(val)])
    return args


def get_exp_name(config: Dict[str, Any]) -> str:
    name = f"{config['method']}_{config['n_way']}w{config['k_shot']}s"
    if config.get("affinity"):
        name += f"_{config['affinity']}"
    if config.get("meta_k"):
        name += f"_k{config['meta_k']}"
    return name


def run_experiment(config: Dict[str, Any], datasets: List[str], log_dir: str):
    """Train and evaluate a single configuration."""
    exp_name = get_exp_name(config)
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"{'='*60}")

    args = config_to_args(config, datasets)
    common_args = ["--log_dir", log_dir, "--save_dir", os.path.join(log_dir, "checkpoints")]

    # Train
    cmd = [sys.executable, "train.py"] + args + common_args
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  TRAIN FAILED: {result.stderr[-500:]}")
        return None
    print(result.stdout[-300:])

    # Evaluate
    cmd = [sys.executable, "evaluate.py"] + args + common_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  EVAL FAILED: {result.stderr[-500:]}")
        return None
    print(result.stdout[-300:])

    # Load results
    results_path = os.path.join(log_dir, f"{exp_name}_test_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


def print_results_table(results: List[Dict]):
    """Print a formatted results table."""
    if not results:
        print("No results to display.")
        return

    header = f"{'Method':<15} {'Config':<15} {'ROC-AUC':>12} {'Accuracy':>12}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        if r is None:
            continue
        config = f"{r['n_way']}w{r['k_shot']}s"
        auc_str = f"{r['roc_auc_mean']:.4f}±{r['roc_auc_ci95']:.4f}"
        acc_str = f"{r['accuracy_mean']:.4f}±{r['accuracy_ci95']:.4f}"
        method = r["method"]
        if r.get("affinity"):
            method += f"({r['affinity'][:3]})"
        if r.get("meta_k"):
            method += f"[k={r['meta_k']}]"
        print(f"{method:<15} {config:<15} {auc_str:>12} {acc_str:>12}")


def plot_convergence(log_dir: str, configs: List[Dict], title: str, filename: str):
    """Plot val AUC convergence curves for given configs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for config in configs:
        exp_name = get_exp_name(config)
        hist_path = os.path.join(log_dir, f"{exp_name}_history.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            history = json.load(f)
        label = exp_name.replace("_", " ")
        ax.plot(history["val_auc"], label=label)
    ax.set_xlabel("Evaluation Round")
    ax.set_ylabel("Meta-Val ROC-AUC")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, filename), dpi=150)
    plt.close()
    print(f"  Plot saved: {os.path.join(log_dir, filename)}")


def plot_sample_efficiency(results: List[Dict], log_dir: str):
    """Plot performance vs K (number of shots)."""
    methods = {}
    for r in results:
        if r is None:
            continue
        m = r["method"]
        if m not in methods:
            methods[m] = {"k": [], "auc": [], "ci": []}
        methods[m]["k"].append(r["k_shot"])
        methods[m]["auc"].append(r["roc_auc_mean"])
        methods[m]["ci"].append(r["roc_auc_ci95"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, data in methods.items():
        order = np.argsort(data["k"])
        k = np.array(data["k"])[order]
        auc = np.array(data["auc"])[order]
        ci = np.array(data["ci"])[order]
        ax.errorbar(k, auc, yerr=ci, marker="o", label=method, capsize=3)
    ax.set_xlabel("K (shots per class)")
    ax.set_ylabel("Meta-Test ROC-AUC")
    ax.set_title("Sample Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "sample_efficiency.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {os.path.join(log_dir, 'sample_efficiency.png')}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Ablation study driver")
    p.add_argument("--datasets", nargs="+", default=["tox21"])
    p.add_argument("--log_dir", type=str, default="./ablation_results")
    p.add_argument("--suite", type=str, default="main",
                   choices=["main", "affinity", "knn", "sample", "all"])
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    suites = {
        "main": MAIN_EXPERIMENTS,
        "affinity": AFFINITY_ABLATION,
        "knn": KNN_ABLATION,
        "sample": SAMPLE_EFFICIENCY,
    }
    if args.suite == "all":
        configs = MAIN_EXPERIMENTS + AFFINITY_ABLATION + KNN_ABLATION + SAMPLE_EFFICIENCY
        # Deduplicate by exp_name
        seen = set()
        unique = []
        for c in configs:
            name = get_exp_name(c)
            if name not in seen:
                seen.add(name)
                unique.append(c)
        configs = unique
    else:
        configs = suites[args.suite]

    # Run all experiments
    all_results = []
    for config in configs:
        result = run_experiment(config, args.datasets, args.log_dir)
        all_results.append(result)

    # Print summary table
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    print_results_table(all_results)

    # Generate plots
    if args.suite in ("main", "all"):
        plot_convergence(args.log_dir, MAIN_EXPERIMENTS,
                         "Convergence: All Methods", "convergence_main.png")
    if args.suite in ("affinity", "all"):
        plot_convergence(args.log_dir, AFFINITY_ABLATION,
                         "Affinity Function Ablation", "convergence_affinity.png")
    if args.suite in ("sample", "all"):
        sample_results = [r for r in all_results if r and r["n_way"] == 5]
        plot_sample_efficiency(sample_results, args.log_dir)

    # Save all results
    summary_path = os.path.join(args.log_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump([r for r in all_results if r], f, indent=2)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
