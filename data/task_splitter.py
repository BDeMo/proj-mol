"""Task-level meta-train / meta-val / meta-test splitting."""

from typing import Dict, List
import numpy as np


def split_tasks(
    task_ids: List[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split task IDs into meta-train / meta-val / meta-test with strict disjointness.

    Args:
        task_ids: List of task identifier strings.
        train_ratio: Fraction of tasks for meta-train.
        val_ratio: Fraction of tasks for meta-val.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys "train", "val", "test", each mapping to a list of task IDs.
    """
    rng = np.random.RandomState(seed)
    ids = list(task_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }

    for split_name, split_ids in splits.items():
        print(f"  meta-{split_name}: {len(split_ids)} tasks")

    # Verify disjointness
    assert len(set(splits["train"]) & set(splits["val"])) == 0
    assert len(set(splits["train"]) & set(splits["test"])) == 0
    assert len(set(splits["val"]) & set(splits["test"])) == 0

    return splits
