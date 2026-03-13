from hypothesis import target
from jazz_graph.model.model import UnsupervisedJazzModel
from jazz_graph.training.loss import nt_xent_loss


import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from collections.abc import Callable


class JitterInputs:
    """Provide small random permutations of sorted inputs.

    This is useful for increasing connectivity in batches when a nodes
    should be highly connected for nearby values of a feature such as times
    and date.
    """
    def __init__(self, inputs, jitter):
        self.inputs = inputs.float()
        self.jitter = jitter
        self._perm = torch.argsort(inputs)

    def set_epoch(self, epoch: int):
        # claude suggested the set_epoch interface for this class.
        noise = torch.rand(len(self)) * self.jitter
        self._perm = torch.argsort(self.inputs + noise)

    def __len__(self):
        return len(self.inputs)

    def __iter__(self):
        return iter(self._perm[i] for i in range(len(self)))

# Claude wrote this and suggseted the set_epoch method for distributed training
# as part of the interface.
class NeighborLoaderWithJitter:
    """A Neighborhood loader that uses a jittered feature to increase batch connectivity."""
    def __init__(self, data: HeteroData, target_input: torch.Tensor|tuple[str, torch.Tensor], num_neighbors: list, batch_size: int, jitter=5.0):
        self.data = data
        if isinstance(target_input, tuple):
            if not target_input[0] in data.metadata()[0]:
                raise ValueError("Input tuple node type should be a in data node types.")
            self.sampler = JitterInputs(target_input[1], jitter)
            self.node_type = target_input[0]
        elif isinstance(target, torch.Tensor):
            self.node_type = None
            self.sampler = JitterInputs(target_input, jitter)
        else:
            raise ValueError("Expected tuple or tensor for target input.")

        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self._loader = self._build_loader()

    def _build_loader(self):
        ordered_nodes = torch.tensor(list(iter(self.sampler)))
        if self.node_type is not None:
            ordered_nodes = (self.node_type, ordered_nodes)
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            input_nodes=ordered_nodes,
            shuffle=False,  # sampler already handles order
        )

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)
        self._loader = self._build_loader()  # rebuild with new order

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)

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