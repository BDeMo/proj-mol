"""Graph-of-Shots v2 — Five modifications aimed at beating ProtoNet.

M1  Residual prototype:     c_n = α · c_enc + (1-α) · c_gnn, α learnable
M2  Shallower meta-GNN:     1-layer GATConv replaces 2-layer GCNConv
M3  Bipartite messaging:    mask edges originating from query nodes
M4  Contrastive aux loss:   InfoNCE on support-query pairs (weight λ)
M5  SSL pretraining:        (separate module — see models/ssl_pretrain.py)

All five can be toggled independently via constructor args.  Setting
every flag to its "off" value recovers v1 semantics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Batch

from .mpnn_encoder import MPNNEncoder
from .graph_of_shots import AffinityFunction, build_meta_graph


# ── Meta-GNN backbones ─────────────────────────────────────────────────
class MetaGCN(nn.Module):
    """Stack of weighted GCNConv layers (v1-style)."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            ic = in_dim if i == 0 else hidden_dim
            oc = out_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(GCNConv(ic, oc))
            self.bns.append(nn.BatchNorm1d(oc))

    def forward(self, x, edge_index, edge_weight):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class MetaGAT(nn.Module):
    """Graph-attention meta-GNN.  Single-layer attention prevents
    over-smoothing and is sufficient on ~40-node episode graphs."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            ic = in_dim if i == 0 else hidden_dim
            if i == num_layers - 1:
                self.convs.append(GATConv(ic, out_dim, heads=1, concat=False))
                self.bns.append(nn.BatchNorm1d(out_dim))
            else:
                self.convs.append(
                    GATConv(ic, hidden_dim // heads, heads=heads, concat=True))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        # GATConv accepts edge_weight only via edge_attr in newer PyG; fall back
        # to structural attention if not supported.
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x


def build_bipartite_mask(edge_index, n_support, n_query):
    """Return a boolean mask keeping only edges where the SOURCE is a
    support node (i.e. query nodes do NOT send messages)."""
    src = edge_index[0]
    keep = src < n_support  # supports are the first n_support indices
    return keep


# ── GoS v2 module ──────────────────────────────────────────────────────
class GraphOfShotsV2(nn.Module):
    """Upgraded Graph-of-Shots with five toggles.

    Args:
        residual_alpha_init : α init value (0 → pure GNN, 1 → pure encoder).
                               Set > 0 to enable M1; α is always learnable.
        meta_gnn_type       : "gcn" (default, v1) or "gat" (M2).
        bipartite           : if True, drop edges whose source is a query node (M3).
        contrastive_lambda  : weight of InfoNCE auxiliary loss (0 disables M4).
        refine_steps        : iterative refinement rounds.
    """

    def __init__(
        self,
        encoder: MPNNEncoder,
        affinity_method: str,
        meta_k: int,
        n_way: int,
        meta_gnn_layers: int = 2,
        refine_steps: int = 2,
        distance: str = "euclidean",
        # v2 toggles
        residual_alpha_init: float = 0.7,
        meta_gnn_type: str = "gcn",
        bipartite: bool = True,
        contrastive_lambda: float = 0.5,
        contrastive_temp: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.meta_k = meta_k
        self.distance = distance
        self.refine_steps = refine_steps
        self.bipartite = bipartite
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp
        self.meta_gnn_type = meta_gnn_type

        d = encoder.out_dim

        # M1 learnable residual coefficient (stored as logit, sigmoid to [0,1])
        alpha_logit = torch.log(torch.tensor(residual_alpha_init /
                                             max(1e-6, 1 - residual_alpha_init)))
        self.alpha_logit = nn.Parameter(alpha_logit)

        # Affinity + meta-GNN per refinement step
        self.affinities = nn.ModuleList([
            AffinityFunction(affinity_method, d) for _ in range(refine_steps)
        ])
        meta_in = d + n_way
        if meta_gnn_type == "gat":
            self.meta_gnns = nn.ModuleList([
                MetaGAT(meta_in, d, d, num_layers=meta_gnn_layers)
                for _ in range(refine_steps)
            ])
        else:
            self.meta_gnns = nn.ModuleList([
                MetaGCN(meta_in, d, d, num_layers=meta_gnn_layers)
                for _ in range(refine_steps)
            ])

    # alias so evaluate/ablation code that reads .affinity keeps working
    @property
    def affinity(self):
        return self.affinities[0]

    @property
    def meta_gnn(self):
        return self.meta_gnns[0]

    def _apply_bipartite(self, edge_index, edge_weight, n_support):
        keep = edge_index[0] < n_support  # only keep edges FROM support nodes
        return edge_index[:, keep], edge_weight[keep]

    def _prototypes(self, z, labels, n_way):
        d = z.size(1)
        proto = torch.zeros(n_way, d, device=z.device)
        counts = torch.zeros(n_way, device=z.device)
        proto.scatter_add_(0, labels.unsqueeze(1).expand_as(z), z)
        counts.scatter_add_(0, labels,
                            torch.ones_like(labels, dtype=torch.float))
        return proto / counts.unsqueeze(1).clamp(min=1)

    def _logits(self, z_q, prototypes):
        if self.distance == "cosine":
            return 10.0 * (F.normalize(z_q, dim=1) @
                           F.normalize(prototypes, dim=1).T)
        return -torch.cdist(z_q, prototypes, p=2)

    def _contrastive_loss(self, z_sup, y_sup, z_q, y_q):
        """InfoNCE: each query's positives are same-class supports."""
        if z_sup.size(0) == 0 or z_q.size(0) == 0:
            return torch.tensor(0.0, device=z_q.device)
        z_sup = F.normalize(z_sup, dim=1)
        z_q = F.normalize(z_q, dim=1)
        sim = z_q @ z_sup.T / self.contrastive_temp  # [Q, N*K]
        # mask[i, j] = 1 if y_q[i] == y_sup[j]
        mask = (y_q.unsqueeze(1) == y_sup.unsqueeze(0)).float()
        # log-sum-exp denominator
        log_denom = torch.logsumexp(sim, dim=1)  # [Q]
        # numerator = log-sum-exp over positives
        # guard against queries that have no positive support (shouldn't happen)
        sim_pos = sim.masked_fill(mask == 0, float("-inf"))
        log_num = torch.logsumexp(sim_pos, dim=1)
        valid = torch.isfinite(log_num)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z_q.device)
        return -(log_num[valid] - log_denom[valid]).mean()

    def forward(
        self,
        support_batch: Batch,
        support_labels: torch.Tensor,
        query_batch: Batch,
        query_labels: torch.Tensor,
        n_way: int,
    ):
        # 1. Encode
        h_s = self.encoder(support_batch)  # [N*K, d]
        h_q = self.encoder(query_batch)    # [Q, d]
        n_support = h_s.size(0)
        z = torch.cat([h_s, h_q], dim=0)   # [M, d]

        # Label augmentation (fixed across refinement rounds)
        label_s = F.one_hot(support_labels, num_classes=self.n_way).float()
        label_q = torch.zeros(h_q.size(0), self.n_way, device=h_q.device)
        label_block = torch.cat([label_s, label_q], dim=0)

        # 2-4. Iterative refinement
        for aff_fn, gnn in zip(self.affinities, self.meta_gnns):
            aff = aff_fn(z)
            edge_index, edge_weight = build_meta_graph(aff, self.meta_k)
            if self.bipartite:
                edge_index, edge_weight = self._apply_bipartite(
                    edge_index, edge_weight, n_support)
            z_in = torch.cat([z, label_block], dim=1)
            if self.meta_gnn_type == "gat":
                # GATConv uses the graph structure, ignoring edge_weight in this
                # simple setup.  Weighted attention as log-bias is left as a
                # follow-up hook; for now structural sparsification carries signal.
                z = gnn(z_in, edge_index)
            else:
                z = gnn(z_in, edge_index, edge_weight)

        z_s_gnn = z[:n_support]
        z_q_gnn = z[n_support:]

        # 5. Prototype head — residual mix of encoder + meta-GNN prototypes
        proto_gnn = self._prototypes(z_s_gnn, support_labels, n_way)
        proto_enc = self._prototypes(h_s, support_labels, n_way)
        alpha = torch.sigmoid(self.alpha_logit)
        prototypes = alpha * proto_enc + (1 - alpha) * proto_gnn

        # Query representation: also residual-mix so classification matches
        # prototype space.  When α→1 this collapses to ProtoNet.
        z_q_mix = alpha * h_q + (1 - alpha) * z_q_gnn
        logits = self._logits(z_q_mix, prototypes)

        loss_ce = F.cross_entropy(logits, query_labels)
        loss = loss_ce
        if self.contrastive_lambda > 0:
            # Contrastive on the MIXED representation so it shapes the same
            # space classifier uses.
            z_s_mix = alpha * h_s + (1 - alpha) * z_s_gnn
            loss_con = self._contrastive_loss(
                z_s_mix, support_labels, z_q_mix, query_labels)
            loss = loss + self.contrastive_lambda * loss_con
        return logits, loss
