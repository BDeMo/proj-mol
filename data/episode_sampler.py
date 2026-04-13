"""Episodic N-way K-shot sampler for few-shot molecular classification."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data, Batch


@dataclass
class Episode:
    """A single few-shot episode."""
    support_batch: Batch       # Batched support graphs
    support_labels: torch.Tensor  # [N*K] integer labels in {0..N-1}
    query_batch: Batch         # Batched query graphs
    query_labels: torch.Tensor    # [n_query] integer labels in {0..N-1}
    task_ids: List[str]        # Which N tasks were sampled


class EpisodeSampler:
    """Sample N-way K-shot episodes from a pool of tasks.

    Each "way" corresponds to one assay's positive molecules.
    For each sampled assay, K positives go to support and n_query // N go to query.
    Molecules positive for multiple sampled assays are excluded to avoid ambiguity.
    """

    def __init__(
        self,
        graphs: List[Data],
        task_indices: Dict[str, Dict[str, List[int]]],
        task_ids: List[str],
        n_way: int,
        k_shot: int,
        n_query: int,
        seed: int = 42,
    ):
        self.graphs = graphs
        self.task_indices = task_indices
        self.task_ids = list(task_ids)
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = np.random.RandomState(seed)
        self.query_per_way = max(1, n_query // n_way)

    def sample_episode(self) -> Episode:
        """Sample a single N-way K-shot episode.

        Returns:
            Episode with batched support/query graphs and labels.
        """
        required_per_way = self.k_shot + self.query_per_way

        # Sample N tasks that each have enough positives
        selected_tasks = self._sample_tasks(required_per_way)

        # Collect positive sets for each selected task
        pos_sets = [
            set(self.task_indices[tid]["pos"]) for tid in selected_tasks
        ]

        # Build per-way candidate pools, excluding molecules positive for other ways
        support_graphs = []
        support_labels = []
        query_graphs = []
        query_labels = []

        for way_idx, tid in enumerate(selected_tasks):
            # Candidates: positive for this task, not positive for other selected tasks
            other_pos = set()
            for j, other_tid in enumerate(selected_tasks):
                if j != way_idx:
                    other_pos |= pos_sets[j]
            candidates = list(pos_sets[way_idx] - other_pos)

            # If not enough after exclusion, allow overlap as fallback
            if len(candidates) < required_per_way:
                candidates = list(pos_sets[way_idx])

            self.rng.shuffle(candidates)
            chosen = candidates[:required_per_way]

            # Split into support and query
            s_indices = chosen[:self.k_shot]
            q_indices = chosen[self.k_shot:self.k_shot + self.query_per_way]

            for idx in s_indices:
                support_graphs.append(self.graphs[idx])
                support_labels.append(way_idx)
            for idx in q_indices:
                query_graphs.append(self.graphs[idx])
                query_labels.append(way_idx)

        # Shuffle support and query independently
        s_perm = self.rng.permutation(len(support_graphs))
        support_graphs = [support_graphs[i] for i in s_perm]
        support_labels = [support_labels[i] for i in s_perm]

        q_perm = self.rng.permutation(len(query_graphs))
        query_graphs = [query_graphs[i] for i in q_perm]
        query_labels = [query_labels[i] for i in q_perm]

        return Episode(
            support_batch=Batch.from_data_list(support_graphs),
            support_labels=torch.tensor(support_labels, dtype=torch.long),
            query_batch=Batch.from_data_list(query_graphs),
            query_labels=torch.tensor(query_labels, dtype=torch.long),
            task_ids=selected_tasks,
        )

    def _sample_tasks(self, required_per_way: int) -> List[str]:
        """Sample N tasks that each have enough positive examples.

        If the split contains fewer eligible tasks than n_way, the episode is
        constructed with the maximum available number of ways (with a warning
        on the first occurrence).
        """
        eligible = [
            tid for tid in self.task_ids
            if len(self.task_indices[tid]["pos"]) >= required_per_way
        ]
        actual_n_way = min(self.n_way, len(eligible))
        if actual_n_way < 2:
            raise ValueError(
                f"Only {len(eligible)} tasks have >= {required_per_way} positives, "
                f"need at least 2 for classification. Try lowering k_shot or "
                f"using more datasets."
            )
        if actual_n_way < self.n_way and not getattr(self, "_warned_nway", False):
            print(f"  [EpisodeSampler] WARNING: only {len(eligible)} eligible "
                  f"tasks in this split, capping n_way from {self.n_way} "
                  f"to {actual_n_way}")
            self._warned_nway = True
        chosen_idx = self.rng.choice(len(eligible), size=actual_n_way, replace=False)
        return [eligible[i] for i in chosen_idx]
