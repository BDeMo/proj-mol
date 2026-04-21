import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_episode_auc(probs: np.ndarray, labels: np.ndarray, n_way: int) -> float:
    """Compute ROC-AUC for an episode's query predictions.

    Args:
        probs: [Q, n_way] softmax probabilities.
        labels: [Q] integer class labels in {0, ..., n_way-1}.
        n_way: number of classes.

    Returns:
        ROC-AUC score (macro one-vs-rest).
    """
    if n_way == 2:
        return roc_auc_score(labels, probs[:, 1])
    try:
        return roc_auc_score(
            labels, probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        return 0.5


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_exp_name(args) -> str:
    """Derive a unique experiment name from the training config.

    If ``--exp_name`` is provided, use it verbatim. Otherwise compose a
    name that includes every hyper-parameter the checkpoint format
    depends on, so no two distinct runs collide on ``<exp>_best.pt``.
    """
    if getattr(args, "exp_name", None):
        return args.exp_name

    parts = [args.method, f"{args.n_way}w{args.k_shot}s"]
    if args.method in ("gos", "gos_v2"):
        parts.append(args.affinity)
        parts.append(f"k{args.meta_k}")
        parts.append(f"r{args.refine_steps}")
    if args.method == "gos_v2":
        parts.append(f"a{args.v2_alpha_init}")
        parts.append(f"{args.v2_gnn_type}L{args.meta_gnn_layers}")
        parts.append("bp" if args.v2_bipartite else "full")
        parts.append(f"lam{args.v2_contrastive_lambda}")
        parts.append("ssl" if getattr(args, "ssl_ckpt", None) else "scr")
        if getattr(args, "freeze_encoder", False):
            parts.append("frz")
    return "_".join(parts)
