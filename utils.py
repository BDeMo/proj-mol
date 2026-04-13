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
