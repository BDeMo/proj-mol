"""Self-supervised MolCLR-style contrastive pretraining for the MPNN encoder.

Two graph augmentations per molecule → encoder → 2-layer MLP head → InfoNCE
loss pulling the two views together.  After training, the projection head is
discarded and the encoder weights are saved for episodic fine-tuning.
"""
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


# ── Augmentations ──────────────────────────────────────────────────────
def _slim(data: Data) -> Data:
    """Return a Data with only the graph-level tensors we need for SSL.

    Stripping per-graph attributes like ``y`` and ``smiles`` avoids
    PyG Batch collate failures (some MoleculeNet items have str attrs,
    others don't after augmentation).
    """
    kwargs = {"x": data.x.float() if data.x is not None else None,
              "edge_index": data.edge_index}
    if getattr(data, "edge_attr", None) is not None:
        kwargs["edge_attr"] = data.edge_attr.float()
    return Data(**kwargs)


def drop_nodes(data: Data, p: float = 0.15) -> Data:
    """Randomly drop a fraction of nodes and their incident edges."""
    d = _slim(data)
    n = d.num_nodes
    if n <= 2:
        return d
    keep = torch.rand(n) > p
    if not keep.any():
        keep[torch.randint(0, n, (1,))] = True
    idx_map = -torch.ones(n, dtype=torch.long)
    idx_map[keep] = torch.arange(int(keep.sum()))

    ei = d.edge_index
    mask = keep[ei[0]] & keep[ei[1]]
    d.x = d.x[keep]
    d.edge_index = idx_map[ei[:, mask]]
    if getattr(d, "edge_attr", None) is not None:
        d.edge_attr = d.edge_attr[mask]
    return d


def mask_atoms(data: Data, p: float = 0.15) -> Data:
    """Zero out feature vectors of a random subset of atoms."""
    d = _slim(data)
    n = d.num_nodes
    if n == 0:
        return d
    mask = torch.rand(n) < p
    d.x = d.x.clone()
    d.x[mask] = 0.0
    return d


def drop_edges(data: Data, p: float = 0.15) -> Data:
    """Randomly remove a fraction of edges."""
    d = _slim(data)
    m = d.edge_index.size(1)
    if m == 0:
        return d
    keep = torch.rand(m) > p
    d.edge_index = d.edge_index[:, keep]
    if getattr(d, "edge_attr", None) is not None:
        d.edge_attr = d.edge_attr[keep]
    return d


AUGMENTATIONS = [drop_nodes, mask_atoms, drop_edges]


def two_views(data: Data) -> tuple[Data, Data]:
    """Sample two independent augmentations."""
    import random
    a1, a2 = random.choice(AUGMENTATIONS), random.choice(AUGMENTATIONS)
    return a1(data), a2(data)


# ── Projection head + InfoNCE ──────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """Symmetric NT-Xent loss for paired views.

    z1, z2: [B, d] — row i of z1 and z2 are two views of the same molecule.
    """
    z = torch.cat([z1, z2], dim=0)  # [2B, d]
    sim = z @ z.T / temperature      # [2B, 2B]
    B = z1.size(0)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))
    # positive pairs: (i, i+B) and (i+B, i)
    labels = torch.arange(2 * B, device=z.device)
    labels = (labels + B) % (2 * B)
    return F.cross_entropy(sim, labels)


# ── Pretraining loop ──────────────────────────────────────────────────
def pretrain_encoder(
    encoder: nn.Module,
    graphs: List[Data],
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.1,
    device: str = "cuda",
    log_every: int = 50,
) -> nn.Module:
    """Run MolCLR-style contrastive pretraining, return the same encoder
    with updated weights in-place.  Projection head is discarded on exit.
    """
    encoder = encoder.to(device)
    proj = ProjectionHead(encoder.out_dim, encoder.out_dim,
                          encoder.out_dim).to(device)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(proj.parameters()), lr=lr)

    N = len(graphs)
    encoder.train(); proj.train()
    print(f"[SSL pretrain] {N} molecules, {epochs} epochs, bs={batch_size}")

    for ep in range(1, epochs + 1):
        perm = torch.randperm(N)
        losses = []
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size].tolist()
            views = []
            for j in idx:
                a, b = two_views(graphs[j])
                if a.num_nodes > 0 and b.num_nodes > 0:
                    views.append((a, b))
            if len(views) < 2:
                continue
            v1, v2 = zip(*views)
            try:
                b1 = Batch.from_data_list(list(v1)).to(device)
                b2 = Batch.from_data_list(list(v2)).to(device)
            except Exception as e:
                # skip any collate failure from PyG on heterogeneous attrs
                continue
            if b1.x.size(0) == 0 or b2.x.size(0) == 0:
                continue
            h1 = proj(encoder(b1))
            h2 = proj(encoder(b2))
            loss = info_nce_loss(h1, h2, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if log_every and len(losses) % log_every == 0:
                print(f"  ep {ep} step {len(losses)}  loss={loss.item():.4f}")
        mean_loss = sum(losses) / max(1, len(losses))
        print(f"[SSL pretrain] epoch {ep}/{epochs}  loss={mean_loss:.4f}")

    return encoder
