"""Load MoleculeNet datasets and build per-task positive/negative indices."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data


@dataclass
class TaskInfo:
    """Per-task index mapping for a dataset."""
    dataset_name: str
    num_tasks: int
    # task_id -> {"pos": [mol_idx, ...], "neg": [mol_idx, ...]}
    task_indices: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    num_molecules: int = 0


def _slim(g: Data) -> Data:
    """Keep only the tensors needed for encoding; drop `y` / `smiles`.

    Different MoleculeNet datasets have different y widths (Tox21=12,
    ToxCast=617, MUV=17, SIDER=27), so cross-dataset episode batches
    cannot collate `y`.  We store labels in our own task_indices dict,
    so per-graph `y` is redundant anyway.
    """
    kwargs = {"x": g.x, "edge_index": g.edge_index}
    if getattr(g, "edge_attr", None) is not None:
        kwargs["edge_attr"] = g.edge_attr
    return Data(**kwargs)


def load_molnet(root: str, name: str, min_pos: int = 16) -> Tuple[List[Data], TaskInfo]:
    """Load a MoleculeNet dataset and build per-task positive/negative molecule indices.

    Args:
        root: Directory to cache downloaded data.
        name: Dataset name (tox21, toxcast, muv, sider).
        min_pos: Minimum number of positive examples to keep a task.

    Returns:
        graphs: List of PyG Data objects.
        task_info: TaskInfo with per-task positive/negative indices.
    """
    dataset = MoleculeNet(root=root, name=name)
    graphs_raw = list(dataset)
    num_molecules = len(graphs_raw)

    # Determine number of task columns from y (read from RAW graphs).
    sample_y = graphs_raw[0].y
    if sample_y.dim() == 1:
        num_tasks = sample_y.size(0)
    else:
        num_tasks = sample_y.size(1)

    # Build per-task indices using raw y BEFORE slimming.
    task_indices: Dict[str, Dict[str, List[int]]] = {}
    for t in range(num_tasks):
        task_id = f"{name}_{t}"
        pos_indices = []
        neg_indices = []
        for mol_idx, g in enumerate(graphs_raw):
            y = g.y.view(-1)
            val = y[t].item()
            if torch.isnan(torch.tensor(val)):
                continue
            if val == 1.0:
                pos_indices.append(mol_idx)
            elif val == 0.0:
                neg_indices.append(mol_idx)
        if len(pos_indices) >= min_pos:
            task_indices[task_id] = {"pos": pos_indices, "neg": neg_indices}

    # Strip per-graph y/smiles so cross-dataset Batch.from_data_list works.
    graphs = [_slim(g) for g in graphs_raw]

    info = TaskInfo(
        dataset_name=name,
        num_tasks=len(task_indices),
        task_indices=task_indices,
        num_molecules=num_molecules,
    )
    return graphs, info


def load_all_datasets(
    data_root: str,
    dataset_names: List[str],
    min_pos: int = 16,
) -> Tuple[List[Data], Dict[str, Dict[str, List[int]]], List[str]]:
    """Load multiple MoleculeNet datasets and merge task indices.

    Molecule indices are offset so they are globally unique across datasets.

    Returns:
        all_graphs: Combined list of Data objects.
        all_task_indices: Merged task_indices dict with globally unique mol indices.
        all_task_ids: List of all valid task IDs.
    """
    all_graphs: List[Data] = []
    all_task_indices: Dict[str, Dict[str, List[int]]] = {}
    all_task_ids: List[str] = []

    offset = 0
    for name in dataset_names:
        graphs, info = load_molnet(data_root, name, min_pos)
        # Offset molecule indices
        for task_id, indices in info.task_indices.items():
            all_task_indices[task_id] = {
                "pos": [idx + offset for idx in indices["pos"]],
                "neg": [idx + offset for idx in indices["neg"]],
            }
            all_task_ids.append(task_id)
        all_graphs.extend(graphs)
        offset += len(graphs)
        print(f"[{name.upper()}] {info.num_molecules} molecules, "
              f"{info.num_tasks} valid tasks (min_pos={min_pos})")

    print(f"Total: {len(all_graphs)} molecules, {len(all_task_ids)} tasks")
    return all_graphs, all_task_indices, all_task_ids
