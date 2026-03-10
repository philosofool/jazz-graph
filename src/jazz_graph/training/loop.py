from jazz_graph.model.model import UnsupervisedJazzModel
from jazz_graph.training.loss import nt_xent_loss


import torch
from torch_geometric.data import HeteroData


from collections.abc import Callable


class UnsupervisedGNNTrainingLogic:
    """Define training step and eval steps."""
    def __init__(self, model: UnsupervisedJazzModel, optimizer, temperature, augment: Callable[[HeteroData], HeteroData]):
        self.device = next(model.parameters()).device
        self.model = model
        self.optimizer = optimizer
        self.temperature = temperature
        self.augment = augment

    def train_step(self, engine, batch: HeteroData) -> dict:
        """SimCLR-style unsuperivised training step on a graph.

        The approach of SimCLR is to perform two random augementations on an input.
        The model then learns embeddings by predicting both augmented graphs.
        The task is to predict which nodes from the augmented batches were the
        same source node. The result is that similar nodes should have nearby
        encodeings while dissimilar nodes have distant ones.
        """
        self.model.train()
        self.optimizer.zero_grad()
        batch.to(self.device)

        h1_dict: dict[str, torch.Tensor] = self.model.encode(self.augment(batch))
        h2_dict: dict[str, torch.Tensor] = self.model.encode(self.augment(batch))
        z1_dict: dict[str, torch.Tensor] = self.model.project(h1_dict)
        z2_dict: dict[str, torch.Tensor] = self.model.project(h2_dict)
        losses = {
            node_type: nt_xent_loss(z1_dict[node_type], z2_dict[node_type], self.temperature)
            for node_type in z1_dict
        }

        total_loss = sum(losses.values())
        total_loss.backward()  # pyright: ignore [reportAttributeAccessIssue]
        self.optimizer.step()
        results = {}
        for node_type in z1_dict:
            results[node_type] = {
                'loss': losses[node_type].item(),
                'h1': h1_dict[node_type].detach(),
                'h2': h2_dict[node_type].detach()
            }
        return results