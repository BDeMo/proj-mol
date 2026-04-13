#!/bin/bash
# Stage 1: Data Pipeline and Episodic Task Construction
# Loads MoleculeNet datasets, builds per-task indices, verifies episodic sampling.

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Stage 1: Data Pipeline & Episodic Sampling"
echo "=============================================="

DATASETS="${DATASETS:-tox21 sider}"

python -c "
from data.molnet_loader import load_all_datasets
from data.task_splitter import split_tasks
from data.episode_sampler import EpisodeSampler

datasets = '${DATASETS}'.split()
print(f'Loading datasets: {datasets}')

# Load and merge datasets
all_graphs, all_task_indices, all_task_ids = load_all_datasets(
    './data_cache', datasets, min_pos=16
)

# Split tasks
print('\n--- Task-level meta-splits ---')
splits = split_tasks(all_task_ids, 0.6, 0.2, seed=42, min_per_split=5)

# Report per-dataset statistics
print('\n--- Per-task statistics ---')
for tid in sorted(all_task_ids):
    idx = all_task_indices[tid]
    print(f'  {tid}: {len(idx[\"pos\"]):>5} pos, {len(idx[\"neg\"]):>5} neg')

# Test episodic sampling for each configuration
configs = [(5, 1), (5, 5), (10, 5)]
for n_way, k_shot in configs:
    print(f'\n--- Sampling test: {n_way}-way {k_shot}-shot ---')
    try:
        sampler = EpisodeSampler(
            all_graphs, all_task_indices, splits['train'],
            n_way=n_way, k_shot=k_shot, n_query=n_way * 4, seed=42
        )
        ep = sampler.sample_episode()
        print(f'  Support: {ep.support_batch.num_graphs} graphs '
              f'(expected {n_way * k_shot})')
        print(f'  Query: {ep.query_batch.num_graphs} graphs')
        print(f'  Labels: {ep.support_labels.tolist()}')
        print(f'  Tasks: {ep.task_ids}')
    except ValueError as e:
        print(f'  Skipped: {e}')

print('\n=== Stage 1 Complete ===')
"
