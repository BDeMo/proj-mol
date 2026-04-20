"""Graph-of-Shots: Meta-graph construction and meta-GNN for few-shot classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch

from .mpnn_encoder import MPNNEncoder


class AffinityFunction(nn.Module):
    """Pairwise affinity between molecular embeddings.

    Supports three variants:
    - cosine: parameter-free cosine similarity
    - bilinear: learned bilinear kernel h_i^T W h_j
    - attention: scaled dot-product attention with learned projections
    """

    def __init__(self, method: str, embed_dim: int):
        super().__init__()
        self.method = method
        if method == "bilinear":
            self.W = nn.Parameter(torch.empty(embed_dim, embed_dim))
            nn.init.xavier_uniform_(self.W)
        elif method == "attention":
            self.W_q = nn.Linear(embed_dim, embed_dim)
            self.W_k = nn.Linear(embed_dim, embed_dim)
            self.scale = embed_dim ** 0.5

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute pairwise affinity matrix.

        Args:
            h: [M, d] embeddings for all molecules in the episode.

        Returns:
            [M, M] affinity matrix.
        """
        if self.method == "cosine":
            h_norm = F.normalize(h, dim=1)
            return h_norm @ h_norm.T
        elif self.method == "bilinear":
            return h @ self.W @ h.T
        elif self.method == "attention":
            q = self.W_q(h)
            k = self.W_k(h)
            return (q @ k.T) / self.scale
        else:
            raise ValueError(f"Unknown affinity method: {self.method}")


def build_meta_graph(
    affinity_matrix: torch.Tensor, k: int
) -> tuple:
    """Build a kNN-sparsified meta-graph from an affinity matrix.

    Args:
        affinity_matrix: [M, M] pairwise affinities.
        k: Number of nearest neighbors per node.

    Returns:
        edge_index: [2, num_edges] COO format (undirected).
        edge_weight: [num_edges] affinity values.
    """
    M = affinity_matrix.size(0)
    k = min(k, M - 1)

    # Mask self-loops before topk
    aff = affinity_matrix.clone()
    aff.fill_diagonal_(float("-inf"))

    vals, indices = torch.topk(aff, k=k, dim=1)  # [M, k]

    src = torch.arange(M, device=indices.device).unsqueeze(1).expand_as(indices).reshape(-1)
    dst = indices.reshape(-1)

    # Make undirected by adding reverse edges
    edge_index = torch.cat([
        torch.stack([src, dst], dim=0),
        torch.stack([dst, src], dim=0),
    ], dim=1)

    edge_weight = torch.cat([vals.reshape(-1), vals.reshape(-1)])

    # Softmax over each node's k neighbors so weights sum to 1 — keeps
    # GCNConv stable without destroying contrast like sigmoid did.
    edge_weight = F.relu(edge_weight) + 1e-6

    return edge_index, edge_weight


class MetaGNN(nn.Module):
    """GNN operating on the task-level meta-graph.

    Propagates relational information between support and query nodes
    using weighted message passing (GCNConv with edge_weight).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            out_c = out_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(GCNConv(in_c, out_c))
            self.bns.append(nn.BatchNorm1d(out_c))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [M, in_dim] initial node features.
            edge_index: [2, num_edges] graph connectivity.
            edge_weight: [num_edges] edge weights.

        Returns:
            [M, out_dim] updated node representations.
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class GraphOfShots(nn.Module):
    """Graph-of-Shots: few-shot classification via meta-graph reasoning.

    Pipeline per episode:
    1. Encode all molecular graphs to embeddings.
    2. Construct meta-graph with affinity-based edges.
    3. Augment node features with label information (one-hot for support, zeros for query).
    4. Run meta-GNN to propagate label info.
    5. Prototype-based classification of query nodes against support prototypes
       computed from meta-GNN outputs.  Using a prototype head instead of a
       shared Linear(d, n_way) is critical for episodic training: class
       indices are arbitrary per episode, so a persistent classifier collapses
       to chance.
    """

    def __init__(
        self,
        encoder: MPNNEncoder,
        affinity_method: str,
        meta_k: int,
        n_way: int,
        meta_gnn_layers: int = 2,
        distance: str = "euclidean",
        refine_steps: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.meta_k = meta_k
        self.distance = distance
        self.refine_steps = refine_steps

        embed_dim = encoder.out_dim

        # Separate affinity + meta-GNN per refinement step so later rounds can
        # specialize to label-conditioned features.
        self.affinities = nn.ModuleList([
            AffinityFunction(affinity_method, embed_dim)
            for _ in range(refine_steps)
        ])

        meta_in_dim = embed_dim + n_way
        meta_hidden = embed_dim
        self.meta_gnns = nn.ModuleList([
            MetaGNN(meta_in_dim, meta_hidden, embed_dim, meta_gnn_layers)
            for _ in range(refine_steps)
        ])

    # Backwards-compatible aliases: ablation code references .affinity / .meta_gnn
    @property
    def affinity(self):
        return self.affinities[0]

    @property
    def meta_gnn(self):
        return self.meta_gnns[0]

    def forward(
        self,
        support_batch: Batch,
        support_labels: torch.Tensor,
        query_batch: Batch,
        query_labels: torch.Tensor,
        n_way: int,
    ):
        """Forward pass for one episode.

        Iterative refinement: at each step, rebuild the meta-graph from the
        current node features and run another meta-GNN pass.  The label one-hot
        is re-injected at every round so label information persists through
        refinement.
        """
        # 1. Encode all molecules
        h_s = self.encoder(support_batch)  # [N*K, d]
        h_q = self.encoder(query_batch)    # [Q, d]
        n_support = h_s.size(0)
        z = torch.cat([h_s, h_q], dim=0)  # [M, d]

        # Pre-compute label augmentation (constant across refinement rounds)
        label_s = F.one_hot(support_labels, num_classes=self.n_way).float()
        label_q = torch.zeros(h_q.size(0), self.n_way, device=h_q.device)
        label_block = torch.cat([label_s, label_q], dim=0)  # [M, n_way]

        # 2-4. Iterative refinement
        for affinity, meta_gnn in zip(self.affinities, self.meta_gnns):
            aff_matrix = affinity(z)
            edge_index, edge_weight = build_meta_graph(aff_matrix, self.meta_k)
            z_input = torch.cat([z, label_block], dim=1)  # [M, d + n_way]
            z = meta_gnn(z_input, edge_index, edge_weight)  # [M, d]

        z_s = z[:n_support]
        z_q = z[n_support:]

        # 5. Prototype-based classification (no persistent class parameters)
        d = z.size(1)
        prototypes = torch.zeros(n_way, d, device=z.device)
        counts = torch.zeros(n_way, device=z.device)
        prototypes.scatter_add_(0, support_labels.unsqueeze(1).expand_as(z_s), z_s)
        counts.scatter_add_(
            0, support_labels, torch.ones_like(support_labels, dtype=torch.float)
        )
        prototypes = prototypes / counts.unsqueeze(1).clamp(min=1)

        if self.distance == "cosine":
            logits = F.normalize(z_q, dim=1) @ F.normalize(prototypes, dim=1).T
            logits = logits * 10.0
        else:
            logits = -torch.cdist(z_q, prototypes, p=2)

        loss = F.cross_entropy(logits, query_labels)
        return logits, loss
