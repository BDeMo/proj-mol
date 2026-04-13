"""Prototypical Network for few-shot molecular classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from .mpnn_encoder import MPNNEncoder


class PrototypicalNetwork(nn.Module):
    """Metric-based few-shot classifier using class prototypes.

    Classifies query molecules by nearest-prototype matching in embedding space.
    """

    def __init__(self, encoder: MPNNEncoder, distance: str = "euclidean"):
        super().__init__()
        self.encoder = encoder
        self.distance = distance

    def forward(
        self,
        support_batch: Batch,
        support_labels: torch.Tensor,
        query_batch: Batch,
        query_labels: torch.Tensor,
        n_way: int,
    ):
        """Forward pass for one episode.

        Args:
            support_batch: Batched support molecular graphs.
            support_labels: [N*K] integer labels.
            query_batch: Batched query molecular graphs.
            query_labels: [Q] integer labels.
            n_way: Number of classes.

        Returns:
            logits: [Q, n_way] classification logits.
            loss: Cross-entropy loss on query set.
        """
        # Encode support and query molecules
        h_s = self.encoder(support_batch)  # [N*K, d]
        h_q = self.encoder(query_batch)    # [Q, d]

        # Compute class prototypes via scatter mean
        prototypes = torch.zeros(n_way, h_s.size(1), device=h_s.device)
        counts = torch.zeros(n_way, device=h_s.device)
        prototypes.scatter_add_(0, support_labels.unsqueeze(1).expand_as(h_s), h_s)
        counts.scatter_add_(0, support_labels, torch.ones_like(support_labels, dtype=torch.float))
        prototypes = prototypes / counts.unsqueeze(1).clamp(min=1)  # [N, d]

        # Compute distances / similarities
        if self.distance == "euclidean":
            # Negative squared euclidean distance as logits
            dists = torch.cdist(h_q, prototypes, p=2)  # [Q, N]
            logits = -dists
        else:
            # Cosine similarity
            h_q_norm = F.normalize(h_q, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)
            logits = h_q_norm @ proto_norm.T  # [Q, N]
            logits = logits * 10.0  # temperature scaling

        loss = F.cross_entropy(logits, query_labels)
        return logits, loss
