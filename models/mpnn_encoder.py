"""Message Passing Neural Network encoder for molecular graphs."""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Batch


class MPNNEncoder(nn.Module):
    """GIN-based molecular graph encoder.

    Maps each molecular graph to a fixed-dimensional embedding via iterative
    message passing and global mean pooling.
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_c, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Encode a batch of molecular graphs.

        Args:
            batch: PyG Batch with x, edge_index, batch attributes.

        Returns:
            Tensor of shape [num_graphs, out_dim].
        """
        x = batch.x.float()
        edge_index = batch.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)

        # Global mean pooling to get graph-level embeddings
        x = global_mean_pool(x, batch.batch)  # [num_graphs, hidden_dim]
        x = self.proj(x)  # [num_graphs, out_dim]
        return x
