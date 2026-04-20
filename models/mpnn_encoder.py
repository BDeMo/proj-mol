"""Message Passing Neural Network encoder for molecular graphs."""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool
from torch_geometric.data import Batch


class MPNNEncoder(nn.Module):
    """GIN-based molecular graph encoder.

    Maps each molecular graph to a fixed-dimensional embedding via iterative
    message passing and global mean pooling.  When ``use_edge_attr=True`` the
    encoder uses ``GINEConv`` to incorporate bond-level features.
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        edge_dim: int = 3,
        use_edge_attr: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.use_edge_attr = use_edge_attr

        # Project input atom features to hidden_dim so all conv layers share dim.
        self.atom_embed = nn.Linear(in_dim, hidden_dim)

        # Project bond features to match node dim (required by GINEConv).
        if use_edge_attr:
            self.bond_embed = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if use_edge_attr:
                self.convs.append(GINEConv(mlp))
            else:
                self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Encode a batch of molecular graphs.

        Args:
            batch: PyG Batch with x, edge_index, (edge_attr), batch attributes.

        Returns:
            Tensor of shape [num_graphs, out_dim].
        """
        x = self.atom_embed(batch.x.float())
        edge_index = batch.edge_index

        if self.use_edge_attr:
            edge_attr = self.bond_embed(batch.edge_attr.float())
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index, edge_attr)
                x = bn(x)
                x = torch.relu(x)
                x = self.dropout(x)
        else:
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = torch.relu(x)
                x = self.dropout(x)

        x = global_mean_pool(x, batch.batch)
        x = self.proj(x)
        return x
