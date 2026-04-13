"""First-order MAML (FOMAML) for few-shot molecular classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from collections import OrderedDict

from .mpnn_encoder import MPNNEncoder


class MAMLClassifier(nn.Module):
    """MAML-based few-shot classifier with first-order approximation.

    Uses functional_call for clean inner-loop adaptation without deepcopy.
    The meta-gradient is approximated by the gradient of the query loss
    w.r.t. the adapted parameters (first-order, no second derivatives).
    """

    def __init__(
        self,
        encoder: MPNNEncoder,
        n_way: int,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, n_way)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def _forward_with_params(
        self, params: dict, batch: Batch
    ) -> torch.Tensor:
        """Run encoder + classifier using the given parameter dict."""
        # Split params back into encoder and classifier
        encoder_params = {
            k[len("encoder."):]: v for k, v in params.items() if k.startswith("encoder.")
        }
        classifier_params = {
            k[len("classifier."):]: v for k, v in params.items() if k.startswith("classifier.")
        }
        h = torch.func.functional_call(self.encoder, encoder_params, (batch,))
        logits = torch.func.functional_call(self.classifier, classifier_params, (h,))
        return logits

    def forward(
        self,
        support_batch: Batch,
        support_labels: torch.Tensor,
        query_batch: Batch,
        query_labels: torch.Tensor,
        n_way: int,
    ):
        """Forward pass for one episode using FOMAML.

        Returns:
            logits: [Q, n_way] classification logits from adapted model.
            loss: Meta-loss for outer-loop optimization.
        """
        # Collect all named parameters
        fast_params = OrderedDict(
            (name, param.clone()) for name, param in self.named_parameters()
        )

        # Inner loop: adapt on support set
        for step in range(self.inner_steps):
            logits_s = self._forward_with_params(fast_params, support_batch)
            loss_s = F.cross_entropy(logits_s, support_labels)
            grads = torch.autograd.grad(
                loss_s, fast_params.values(), create_graph=False
            )
            fast_params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(fast_params.items(), grads)
            )

        # Query loss with adapted parameters
        logits_q = self._forward_with_params(fast_params, query_batch)
        loss_q = F.cross_entropy(logits_q, query_labels)

        # For FOMAML: the meta-gradient is approximated as the gradient of loss_q
        # w.r.t. the original parameters. Since fast_params were cloned (not detached),
        # loss_q.backward() will flow gradients back to self.parameters() through
        # the clone operation. This gives the first-order approximation.
        return logits_q, loss_q
