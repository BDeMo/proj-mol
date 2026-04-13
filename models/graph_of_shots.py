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

    # Normalize weights to [0, 1] via sigmoid for stability
    edge_weight = torch.sigmoid(edge_weight)

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
    5. Classify query nodes.
    """

    def __init__(
        self,
        encoder: MPNNEncoder,
        affinity_method: str,
        meta_k: int,
        n_way: int,
        meta_gnn_layers: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.meta_k = meta_k

        embed_dim = encoder.out_dim
        self.affinity = AffinityFunction(affinity_method, embed_dim)

        # Meta-GNN input: embedding + one-hot label
        meta_in_dim = embed_dim + n_way
        meta_hidden = embed_dim
        self.meta_gnn = MetaGNN(meta_in_dim, meta_hidden, embed_dim, meta_gnn_layers)

        # Classification head
        self.classifier = nn.Linear(embed_dim, n_way)

    def forward(
        self,
        support_batch: Batch,
        support_labels: torch.Tensor,
        query_batch: Batch,
        query_labels: torch.Tensor,
        n_way: int,
    ):
        """Forward pass for one episode.

        ``n_way`` may be smaller than ``self.n_way`` when the episode sampler
        caps the number of ways.  In that case, the one-hot label vector is
        zero-padded to ``self.n_way`` so that layer dimensions remain fixed,
        and the classifier output is sliced to ``[:, :n_way]``.

        Returns:
            logits: [Q, n_way] query classification logits.
            loss: Cross-entropy loss on query predictions.
        """
        # 1. Encode all molecules
        h_s = self.encoder(support_batch)  # [N*K, d]
        h_q = self.encoder(query_batch)    # [Q, d]
        h_all = torch.cat([h_s, h_q], dim=0)  # [M, d]

        n_support = h_s.size(0)

        # 2. Build meta-graph
        aff_matrix = self.affinity(h_all)  # [M, M]
        edge_index, edge_weight = build_meta_graph(aff_matrix, self.meta_k)

        # 3. Augment node features with label info (pad to self.n_way)
        label_s = F.one_hot(support_labels, num_classes=self.n_way).float()  # [N*K, self.n_way]
        label_q = torch.zeros(h_q.size(0), self.n_way, device=h_q.device)

        z_init = torch.cat([
            torch.cat([h_s, label_s], dim=1),
            torch.cat([h_q, label_q], dim=1),
        ], dim=0)  # [M, d + self.n_way]

        # 4. Meta-GNN propagation
        z = self.meta_gnn(z_init, edge_index, edge_weight)  # [M, d]

        # 5. Classify query nodes (slice to actual n_way)
        query_z = z[n_support:]  # [Q, d]
        logits = self.classifier(query_z)[:, :n_way]  # [Q, n_way]

        loss = F.cross_entropy(logits, query_labels)
        return logits, loss
