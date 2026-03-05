

from collections import Counter
from functools import partial
from math import exp
import jsonlines
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import os

from ignite.engine import Engine, Events
from ignite.handlers import ProgressBar
from ignite.metrics import Metric, RunningAverage

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphConv, SAGEConv, to_hetero, HeteroConv
from torch_geometric import transforms as T
from torch_geometric import seed_everything

from jazz_graph.data.graph_transforms import drop_edge_from_masks, prune_graph_from_masks
from jazz_graph.data.reporting import inspect_degrees
from jazz_graph.training.logging import (
    ExperimentLogger,
    save_embeddings_handler,
    save_checkpoint_handler,
    run_evaluator_handler,
    log_experiment_handler,
    console_logging,
    binary_output_transform
)
from jazz_graph.data.graph_builder import CreateTensors, prune_isolated_nodes, make_jazz_data
from jazz_graph.model.model import JazzModel, LinkPredictionModel, NodeClassifier
from jazz_graph.training.logging import plot_logs

# Define a model

class UnsupervisedJazzModel(nn.Module):
    def __init__(self, gnn_encoder: JazzModel, embeddings_dim, projection_dim):
        super().__init__()
        self.base_model = gnn_encoder
        self.projections = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(embeddings_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )
            for key in ['performance', 'artist', 'song']
        })

    def encode(self, batch):
        return self.base_model(batch.x_dict, batch.edge_index_dict, batch)

    def project(self, x_dict):
        for key, projection in self.projections.items():
            x_dict[key] = projection(x_dict[key])
        x_dict = {key: F.normalize(tensor, dim=-1) for key, tensor in x_dict.items()}
        return x_dict

    def forward(self, batch):
        x_dict = self.encode(batch)
        x_dict = self.project(x_dict)
        return x_dict

# Definte augmentation Strategies

def drop_random_nodes_and_edges(data: HeteroData):
    out = data.clone()
    drop_edge_augmentation(data, out)
    # drop_node_augmentation(data, out)  # This would be complex: graphs need to align their node indecies in loss.
    return out

def drop_node_augmentation(src_graph: HeteroData, dst_graph: HeteroData, drop_node_prob: float = .1):
    node_types, edge_types = src_graph.metadata()
    masks = {
        node_type: torch.rand(src_graph[node_type].num_nodes) > drop_node_prob
        for node_type in node_types
    }
    prune_graph_from_masks(src_graph, masks)

def drop_edge_augmentation(graph: HeteroData, dst_graph, drop_edge_prob: float = .2):
    edge_types = graph.metadata()[1]
    edge_masks = {
        edge_type: torch.rand(graph[edge_type].edge_index.size(1)) > drop_edge_prob
        for edge_type in edge_types
    }
    return drop_edge_from_masks(graph, edge_masks, dst_graph)

# NT_Xent loss. Used in SimCLR.

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5):
    """
    NT-Xent loss for SimCLR.

    For each sample i:
    - Positive pair: (z1[i], z2[i]) - same image, different augmentations
    - Negative pairs: All other samples in the batch

    Goal: Pull positives together, push negatives apart
    """
    batch_size = z1.size(0)

    # Concatenate both views: [2N, dim]
    z = torch.cat([z1, z2], dim=0)

    # Compute similarity matrix: [2N, 2N]
    # We expect normalized embeddings in z, so dot product and cosine similarity are same:
    # sim[i,j] = cosine_similarity(z[i], z[j]) / temperature
    sim_matrix = torch.mm(z, z.t()) / temperature

    # For each z1[i], the positive is z2[i] (at index i+N)
    # For each z2[i], the positive is z1[i] (at index i)
    labels = torch.cat([
        torch.arange(batch_size) + batch_size,   # z1[i] matches z2[i]
        torch.arange(batch_size)                 # z2[i] matches z1[i]
    ])

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    # NOTE: internally, cross-entropy loss is in log space: entropy(-inf) == 0
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

    # Cross-entropy: treat it as a classification problem
    # "Which of the 2N-1 other samples is the positive pair?"
    loss = F.cross_entropy(sim_matrix, labels)

    return loss

# Some metrics

def alignment_loss(z1, z2):
    """
    Measures how close positive pairs are.
    Lower is better (pairs are aligned).
    """
    return (z1 - z2).pow(2).sum(dim=1).mean()


def uniformity_loss(z, t=2):
    """
    Measures how uniformly distributed embeddings are on unit sphere.
    Lower is better (more uniform).
    """
    sq_dist = torch.pdist(z, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().log()


class AlignmentLoss(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self.alignment_sum = 0.0
        self.count = 0
        super().__init__(output_transform, device)

    @torch.no_grad()
    def update(self, output) -> None:
        z1, z2 = output
        # Sum of squared distances
        self.alignment_sum += (z1 - z2).pow(2).sum(dim=1).mean().item()
        self.count += z1.size(0)

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.alignment_sum / self.count

    def reset(self):
        self.alignment_sum = 0.0
        self.count = 0


class UniformityLoss(Metric):
    def __init__(self, t=2, output_transform=lambda x: x, device="cpu"):
        self.uniformity_sum = 0.0
        self.count = 0
        self.t = t
        super().__init__(output_transform, device)

    def reset(self):
        self.uniformity_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, output):
        z1, z2 = output

        # Compute uniformity for both views
        for z in [z1, z2]:
            sq_dist = torch.pdist(z, p=2).pow(2)
            uniformity = sq_dist.mul(-self.t).exp().mean().log()
            self.uniformity_sum += uniformity.item()
            self.count += 1

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.uniformity_sum / self.count

# Thanks to Claude.ai for taking AlignmentLoss as template and giving this
# evaluator of embedding variance.
class EmbeddingStd(Metric):
    """
    Metric to track average standard deviation per dimension of embeddings.
    Low values indicate potential collapse.
    """
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self.sum_embeddings = None
        self.sum_squared_embeddings = None
        self.count = 0
        super().__init__(output_transform, device)

    @torch.no_grad()
    def reset(self):
        self.sum_embeddings = None
        self.sum_squared_embeddings = None
        self.count = 0

    @torch.no_grad()
    def update(self, output) -> None:
        """
        Args:
            output: Embeddings tensor of shape [batch_size, embedding_dim]
        """
        embeddings = torch.concat(output)
        batch_size = embeddings.size(0)

        # Initialize on first batch
        if self.sum_embeddings is None:
            embed_dim = embeddings.size(1)
            self.sum_embeddings = torch.zeros(embed_dim, device=embeddings.device)
            self.sum_squared_embeddings = torch.zeros(embed_dim, device=embeddings.device)

        # Accumulate statistics
        self.sum_embeddings += embeddings.sum(dim=0)
        self.sum_squared_embeddings += (embeddings ** 2).sum(dim=0)  # pyright: ignore [reportOperatorIssue]
        self.count += batch_size

    @torch.no_grad()
    def compute(self) -> float:
        """
        Compute average standard deviation per dimension.

        Using: std = sqrt(E[X^2] - E[X]^2)
        """
        if self.count == 0:
            return 0.0

        # Compute mean per dimension
        mean = self.sum_embeddings / self.count   # pyright: ignore [reportOptionalOperand]

        # Compute variance per dimension
        mean_squared = self.sum_squared_embeddings / self.count   # pyright: ignore [reportOptionalOperand]
        variance = mean_squared - (mean ** 2)

        # Standard deviation per dimension
        std_per_dim = torch.sqrt(variance.clamp(min=1e-8))  # Clamp to avoid NaN

        # Average across dimensions
        avg_std = std_per_dim.mean()

        return avg_std.item()


# # Usage:
# from ignite.engine import Engine

# # Attach to engine
# embedding_std_metric = EmbeddingStd(output_transform=lambda out: out['embeddings'])
# embedding_std_metric.attach(evaluator, 'embedding_std')

# # Or if your output is directly embeddings:
# embedding_std_metric = EmbeddingStd()
# embedding_std_metric.attach(evaluator, 'embedding_std')

# # Access after evaluation
# evaluator.run(val_loader)
# avg_std = evaluator.state.metrics['embedding_std']
# print(f"Average embedding std: {avg_std:.4f}")

# # Warning threshold
# if avg_std < 0.01:
#     print("WARNING: Possible embedding collapse!")

# Define training.

class UnsupervisedGNNTrainingLogic:
    """Define training step and eval steps."""
    def __init__(self, model: UnsupervisedJazzModel, optimizer, temperature):
        self.device = next(model.parameters()).device
        self.model = model
        self.optimizer = optimizer
        self.temperature = temperature
        # self.criterion = partial(nt_xent_loss, temperature=temperature)
        self.augment = drop_random_nodes_and_edges

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
            node_type:  nt_xent_loss(z1_dict[node_type], z2_dict[node_type], self.temperature)
            for node_type in z1_dict
        }

        # print(f"z1 norms: {torch.norm(z1_dict['performance'], dim=1).mean():.3f}")
        # print(f"z2 norms: {torch.norm(z2_dict['performance'], dim=1).mean():.3f}")

        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()
        results = {}
        for node_type in z1_dict:
            results[node_type] = {
                'loss': losses[node_type].item(),
                'h1': h1_dict[node_type].detach(),
                'h2': h2_dict[node_type].detach()
            }
        return results

    # def eval_step(self, engine, batch: HeteroData) -> dict:
    #     """Complete one pass over a batch of data with no-grad and return results."""
    #     self.model.eval()
    #     batch.to(self.device)

    #     batch_size = batch['performance'].batch_size
    #     with torch.no_grad():
    #         y_pred = self.model(batch)[:batch_size]
    #         y_true = batch['performance'].y[:batch_size].to(torch.float)
    #         loss = self.criterion(y_pred, y_true.to(torch.float))
    #     return {'y_pred': y_pred, 'y_true': y_true}


def make_trainer(model, temperature: float):
    # criterion = nn.BCEWithLogitsLoss()
    node_types = ['performance', 'artist', 'song']
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config['lr'])
    trainer_logic = UnsupervisedGNNTrainingLogic(model, optimizer, temperature)

    experiment_logger = ExperimentLogger(root='/workspace/experiments', run_name=f'gnn_simCLR_{os.path.basename(models_dir)}', config=experiment_config)

    trainer = Engine(trainer_logic.train_step)

    def create_metrics_for_node_type(node_type):
        """Create metrics for a specific node type."""
        return {
            f"{node_type}_loss": RunningAverage(
                output_transform=lambda out: out[node_type]['loss']
            ),
            f"{node_type}_alignment": AlignmentLoss(
                output_transform=lambda out: (out[node_type]['h1'], out[node_type]['h2'])
            ),
            f"{node_type}_uniformity": UniformityLoss(
                output_transform=lambda out: (out[node_type]['h1'], out[node_type]['h2'])
            ),
            f"{node_type}_embedding_std": EmbeddingStd(
                output_transform=lambda out: (out[node_type]['h1'], out[node_type]['h2'])
            ),
        }

    # Create all metrics
    metrics = {}
    for node_type in node_types:
        metrics.update(create_metrics_for_node_type(node_type))

    # Attach
    for name, metric in metrics.items():
        metric.attach(trainer, name)

    progress_bar = ProgressBar()
    progress_bar.attach(trainer, metric_names=[f'{node_type}_loss' for node_type in node_types])

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator_handler, train_evaluator, train_loader, "Training")
    trainer.add_event_handler(Events.EPOCH_COMPLETED, console_logging, 'Training', trainer)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator_handler, dev_evaluator, dev_loader)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_experiment_handler, experiment_logger, 'train', trainer)
    # dev_evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_experiment_handler, experiment_logger, 'dev', trainer)
    # dev_evaluator.add_event_handler(Events.EPOCH_COMPLETED, console_logging, 'Valiation', trainer)
    trainer.add_event_handler(Events.COMPLETED, save_embeddings_handler, experiment_logger, model)
    trainer.add_event_handler(Events.COMPLETED, experiment_logger.save_checkpoint)
    return trainer

def train_indecies(mask):
    """Helper to take a mask array and return the relevant nodes."""
    # I think there's a torch_geometric helper for this.
    num_nodes = mask.shape[0]
    all_node_indicies = torch.arange(num_nodes)
    return all_node_indicies[mask]


if __name__ == '__main__':


    models_dir = '/workspace/local_data/graph_parquet_proto'
    assert os.path.exists(models_dir)

    experiment_config = {
        'data_config': {
            'dataset': models_dir,
            # Sampling is important here. We want to make sure that the neighborhodd
            # for high degree nodes doesn't become too noisy. Artist and song nodes can
            # be very high degree. Basially, get a sizable neighborhood for each node to
            # reduce this noise.
            'sampling': {'num_neighbors': [25, 15, 8]}
        },
        'model': {
            'hidden_dim': 128,
            'embed_dim': 64,
            'projection_dim': 64,
            # NOTE: sage seems to need high regularization, >.4
            #       gat does better with less, ~=.2
            'dropout': 0.2,
            'model_type': 'gat'
        },
        'dataset': models_dir,
        'lr': .03,
        'batch_size': 128,
        'temperature': .5
    }
    model_config = experiment_config['model']
    data_config = experiment_config['data_config']

    create = CreateTensors(models_dir)
    data = make_jazz_data(create)

    model = UnsupervisedJazzModel(
        JazzModel(
            data['performance'].num_nodes,
            data['artist'].num_nodes,
            data['song'].num_nodes,
            hidden_dim=model_config['hidden_dim'],
            embed_dim=model_config['embed_dim'],
            dropout=model_config['dropout'],
            metadata=data.metadata(),
            model_type=model_config['model_type']
        ),
        embeddings_dim=model_config['hidden_dim'],
        projection_dim=model_config['projection_dim']
    )
    num_neighbors = data_config['sampling']['num_neighbors']
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=experiment_config['batch_size'],
        input_nodes=('performance', train_indecies(data['performance'].train_mask)),
        shuffle=True
    )
    # dev_loader = NeighborLoader(
    #     data,
    #     sampling,
    #     batch_size=128,
    #     input_nodes=('performance', train_indicies(data['performance'].dev_mask)),
    #     shuffle=True
    # )
    trainer = make_trainer(model, experiment_config['temperature'])
    trainer.run(train_loader, max_epochs=2)
