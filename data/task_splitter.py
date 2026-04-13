"""Task-level meta-train / meta-val / meta-test splitting."""

from typing import Dict, List
import numpy as np


def split_tasks(
    task_ids: List[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_per_split: int = 0,
) -> Dict[str, List[str]]:
    """Split task IDs into meta-train / meta-val / meta-test with strict disjointness.

    When ``min_per_split > 0`` the function guarantees every split contains at
    least that many tasks by taking from the train partition first.  If the
    total pool is too small to satisfy all three splits, it distributes as
    evenly as possible.

    Args:
        task_ids: List of task identifier strings.
        train_ratio: Fraction of tasks for meta-train.
        val_ratio: Fraction of tasks for meta-val.
        seed: Random seed for reproducibility.
        min_per_split: Each split will have at least this many tasks.

    Returns:
        Dict with keys "train", "val", "test", each mapping to a list of task IDs.
    """
    rng = np.random.RandomState(seed)
    ids = list(task_ids)
    rng.shuffle(ids)

    n = len(ids)

    # --- compute sizes, respecting min_per_split ---
    n_val = max(int(n * val_ratio), min_per_split)
    n_test = max(int(n * (1.0 - train_ratio - val_ratio)), min_per_split)
    n_train = n - n_val - n_test

    # If train ended up too small, rebalance evenly
    if n_train < min_per_split and min_per_split > 0:
        base = n // 3
        rem = n % 3
        n_train = base + (1 if rem > 0 else 0)
        n_val = base + (1 if rem > 1 else 0)
        n_test = n - n_train - n_val

    # Clamp to valid range
    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    n_test = max(n - n_train - n_val, 1)
    # Re-adjust train in case clamping inflated val+test
    n_train = n - n_val - n_test

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
