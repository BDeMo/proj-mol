"""Load MoleculeNet datasets and build per-task positive/negative indices."""

import os
import time
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


def _cache_path(data_root: str, dataset_names: List[str],
                min_pos: int) -> str:
    key = "-".join(sorted(dataset_names))
    return os.path.join(data_root, f"_merged_{key}_mp{min_pos}.pt")


def _pack_graphs(graphs: List[Data]) -> dict:
    """Flatten a list of PyG Data objects into plain tensors.

    PyG ``Data.__reduce__`` carries a lot of metadata that bloats pickle
    by orders of magnitude (we once generated a 75 GB file for Tox21).
    We store only the three arrays we actually use, concatenated into
    one big tensor per field plus per-graph size arrays so we can split
    on load.
    """
    x_list, ei_list, ea_list = [], [], []
    sizes_x, sizes_e = [], []
    for g in graphs:
        x_list.append(g.x)
        ei_list.append(g.edge_index)
        sizes_x.append(g.x.size(0))
        sizes_e.append(g.edge_index.size(1))
        ea_list.append(getattr(g, "edge_attr", None))
    has_edge_attr = all(e is not None for e in ea_list)
    packed = {
        "x": torch.cat(x_list, dim=0) if x_list else torch.empty(0),
        "edge_index": torch.cat(ei_list, dim=1) if ei_list else torch.empty(2, 0,
                                                                            dtype=torch.long),
        "sizes_x": torch.tensor(sizes_x, dtype=torch.long),
        "sizes_e": torch.tensor(sizes_e, dtype=torch.long),
    }
    if has_edge_attr:
        packed["edge_attr"] = torch.cat(ea_list, dim=0)
    return packed


def _unpack_graphs(packed: dict) -> List[Data]:
    x_all = packed["x"]
    ei_all = packed["edge_index"]
    sizes_x = packed["sizes_x"].tolist()
    sizes_e = packed["sizes_e"].tolist()
    has_ea = "edge_attr" in packed
    ea_all = packed.get("edge_attr", None)
    graphs: List[Data] = []
    xo = eo = 0
    for nx, ne in zip(sizes_x, sizes_e):
        kw = {
            "x": x_all[xo:xo + nx],
            "edge_index": ei_all[:, eo:eo + ne],
        }
        if has_ea:
            kw["edge_attr"] = ea_all[eo:eo + ne]
        graphs.append(Data(**kw))
        xo += nx; eo += ne
    return graphs


def load_all_datasets(
    data_root: str,
    dataset_names: List[str],
    min_pos: int = 16,
    use_cache: bool = True,
) -> Tuple[List[Data], Dict[str, Dict[str, List[int]]], List[str]]:
    """Load multiple MoleculeNet datasets and merge task indices.

    A local cache at ``<data_root>/_merged_<names>_mp<min_pos>.pkl`` stores
    the processed (graphs, task_indices, task_ids) tuple so subsequent
    ``python train.py`` invocations start in ~1 s instead of re-parsing
    every molecule.  Pass ``use_cache=False`` to force a fresh build.

    Returns:
        all_graphs: Combined list of Data objects.
        all_task_indices: Merged task_indices dict with globally unique mol indices.
        all_task_ids: List of all valid task IDs.
    """
    os.makedirs(data_root, exist_ok=True)
    cache = _cache_path(data_root, dataset_names, min_pos)
    if use_cache and os.path.exists(cache):
        t0 = time.time()
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        all_graphs = _unpack_graphs(blob["packed"])
        all_task_indices = blob["task_indices"]
        all_task_ids = blob["task_ids"]
        print(f"[cache] loaded {len(all_graphs)} molecules, "
              f"{len(all_task_ids)} tasks from {cache} "
              f"in {time.time() - t0:.1f}s")
        return all_graphs, all_task_indices, all_task_ids

    all_graphs: List[Data] = []
    all_task_indices: Dict[str, Dict[str, List[int]]] = {}
    all_task_ids: List[str] = []

    offset = 0
    for name in dataset_names:
        graphs, info = load_molnet(data_root, name, min_pos)
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

    if use_cache:
        try:
            blob = {
                "packed": _pack_graphs(all_graphs),
                "task_indices": all_task_indices,
                "task_ids": all_task_ids,
            }
            torch.save(blob, cache)
            size_mb = os.path.getsize(cache) / (1024 * 1024)
            print(f"[cache] saved to {cache}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[cache] save failed ({e}) — continuing without cache")

    return all_graphs, all_task_indices, all_task_ids
