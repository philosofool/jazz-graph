

from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path
from math import exp
import jsonlines
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import os

from ignite.engine import Engine, Events
from ignite.handlers import ProgressBar
from ignite.metrics import RunningAverage

import torch
from torch import layer_norm, seed
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphConv, SAGEConv, to_hetero, HeteroConv
from torch_geometric import transforms as T
from torch_geometric import seed_everything

from jazz_graph.data.reporting import inspect_degrees
from jazz_graph.etl.transforms import map_array, map_by_index
from jazz_graph.metrics.embedding_metrics import UniformityLoss
from jazz_graph.metrics.embedding_metrics import EmbeddingStd
from jazz_graph.metrics.embedding_metrics import AlignmentLoss
from jazz_graph.model.model import UnsupervisedJazzModel
from jazz_graph.training.views import drop_random_nodes_and_edges
from jazz_graph.training.logging import (
    ExperimentLogger,
    load_model,
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
from jazz_graph.training.views import MatchAlbumAugmentation, performance_album_map

# NT_Xent loss. Used in SimCLR.
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
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

    # We expect normalized embeddings in z, so dot product and cosine similarity are same:
    # sim[i,j] = cosine_similarity(z[i], z[j]) / temperature
    sim_matrix = torch.mm(z, z.t()) / temperature

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


# Thanks to Claude.ai for taking AlignmentLoss as template and giving this
# evaluator of embedding variance.

class UnsupervisedGNNTrainingLogicMatchAlbum:
    """Define training step and eval steps."""
    def __init__(self, model: UnsupervisedJazzModel, optimizer, temperature):
        self.device = next(model.parameters()).device
        self.model = model
        self.optimizer = optimizer
        self.temperature = temperature
        # self.map_nodes = map_nodes

    def map_nodes(self, batch):
        return performance_album_map(batch)

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

        h1_dict: dict[str, torch.Tensor] = self.model.encode(batch)
        # h2_dict: dict[str, torch.Tensor] = self.model.encode(self.augment(batch))
        z1_dict: dict[str, torch.Tensor] = self.model.project(h1_dict)
        album_match_idx = self.map_nodes(batch)
        h2_dict = h1_dict
        z2_dict: dict[str, torch.Tensor] = z1_dict
        h2_dict['performance'] = h1_dict['performance'][album_match_idx]
        z2_dict['performance'] = z2_dict['performance'][album_match_idx]
        loss = nt_xent_loss(z1_dict['performance'], z2_dict['performance'], self.temperature)
        loss.backward()
        self.optimizer.step()
        # get some metrics
        with torch.no_grad():
            losses = {
                node_type: nt_xent_loss(z1_dict[node_type], z2_dict[node_type], self.temperature)
                for node_type in ['artist', 'song']
            }
        losses['performance'] = loss
        # total_loss = sum(losses.values())
        # total_loss.backward()  # pyright: ignore [reportAttributeAccessIssue]
        results = {}
        for node_type in z1_dict:
            results[node_type] = {
                'loss': losses[node_type].item(),
                'h1': h1_dict[node_type].detach(),
                'h2': h2_dict[node_type].detach()
            }
        return results

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

def console_logging(evaluator: Engine, step_name: str, trainer: Engine):
    metrics = evaluator.state.metrics
    print(f"{step_name} - Epoch[{trainer.state.epoch:03}]")
    def order_keys(key):
        if 'performance' in key:
            return 0
        if 'artist' in key:
            return 1
        if 'song' in key:
            return 2
        return 3
    metrics = sorted(metrics.items(),  key=lambda x: order_keys(x[0]))
    for i, (metric, value) in enumerate(metrics):
        if i % 4 == 0 and i != 0:
            print()
        print(f"  Avg. {metric}: {value:.3f}", end='; ')

    print()

def make_trainer(model, optimizer, experiment_logger: ExperimentLogger):
    node_types = ['performance', 'artist', 'song']
    experiment_config = experiment_logger.load_config()
    if experiment_config is None:
        raise ValueError("Logger must have an experiment config.")
    model_dir = experiment_config['data_config']['dataset']

    trainer_logic = UnsupervisedGNNTrainingLogicMatchAlbum(
        model,
        optimizer,
        experiment_config['temperature']
        # make_match_album_augmentation(models_dir)
    )

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

    metrics = {}
    for node_type in node_types:
        metrics.update(create_metrics_for_node_type(node_type))

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    progress_bar = ProgressBar()
    progress_bar.attach(trainer, metric_names=[f'{node_type}_loss' for node_type in node_types])

    trainer.add_event_handler(Events.EPOCH_COMPLETED, console_logging, 'Training', trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_experiment_handler, experiment_logger, 'train', trainer)
    # trainer.add_event_handler(Events.COMPLETED, save_embeddings_handler, experiment_logger, model)
    trainer.add_event_handler(Events.COMPLETED, save_checkpoint_handler, experiment_logger, model, optimizer)
    return trainer

def train_indecies(mask):
    """Helper to take a mask array and return the relevant nodes."""
    # I think there's a torch_geometric helper for this.
    num_nodes = mask.shape[0]
    all_node_indicies = torch.arange(num_nodes)
    return all_node_indicies[mask]

def make_match_album_augmentation(path_to_node_data) -> MatchAlbumAugmentation:
    graph = make_jazz_data(CreateTensors(path_to_node_data))
    data = pd.read_parquet(Path(path_to_node_data) / 'performance_nodes.parquet')
    albums = data.release_group_id.unique()
    albums_lookup = {album: i for i, album in enumerate(albums)}
    rec_to_album_lookup = data[['recording_id', 'release_group_id']].set_index('recording_id').release_group_id

    print(data.head())
    print(graph['performance'].x[:5])
    performance_ids = graph['performance'].x[:, 1].numpy()
    album_in_graph = rec_to_album_lookup.loc[performance_ids]
    rec_to_album_index = map_array(album_in_graph, albums_lookup)
    rec_to_album_edge = np.stack(
        [np.arange(rec_to_album_index.size), rec_to_album_index], axis=0)
    assert rec_to_album_edge.shape[0] == 2, "Stack should create 2 x num_nodes."
    rec_to_album_edge = torch.from_numpy(rec_to_album_edge)
    return MatchAlbumAugmentation(recording_album_edge=rec_to_album_edge)


if __name__ == '__main__':

    seed_everything(42)
    models_dir = '/workspace/local_data/graph_parquet_proto'
    assert os.path.exists(models_dir)
    create = CreateTensors(models_dir)
    data = make_jazz_data(create)

    make_match_album_augmentation(models_dir)
    # assert False
    run_to_load: str|None = None

    if run_to_load:
        if run_to_load == 'most_recent':
            run_to_load = max(Path("/workspace/experiments").iterdir(), key=lambda p: p.stat().st_mtime)
        experiment_logger = ExperimentLogger.from_run_dir(run_to_load)
        experiment_config = experiment_logger.load_config()
        if experiment_config is None:
            raise ValueError("Only experiment loggers with configs can be used.")
    else:
        experiment_config = {
            'data_config': {
                'dataset': models_dir,
                # Sampling is important here. We want to make sure that the neighborhodd
                # for high degree nodes doesn't become too noisy. Artist and song nodes can
                # be very high degree. Basially, get a sizable neighborhood for each node to
                # reduce this noise.
                'sampling': {'num_neighbors': [25, 15, 8, 8]}
            },
            'model': {
                'num_performances': data['performance'].num_nodes,
                'num_artists': data['artist'].num_nodes,
                'num_songs': data['song'].num_nodes,
                'hidden_dim': 128,
                'embed_dim': 64,
                'projection_dim': 64,
                'dropout': 0.0005,
                'model_type': 'sage',
                'num_layers': 4,
            },
            'drop_edge_prob': .5,
            'dataset': models_dir,
            'lr': .001,
            'batch_size': 256,
            'temperature': .2
        }
        experiment_logger = ExperimentLogger(root='/workspace/experiments', run_name=f'gnn_simCLR_{os.path.basename(models_dir)}', config=experiment_config)

    model_config = experiment_config['model']
    data_config = experiment_config['data_config']


    model = UnsupervisedJazzModel(
        JazzModel(
            num_performances=model_config['num_performances'],
            num_artists=model_config['num_artists'],
            num_songs=model_config['num_songs'],
            hidden_dim=model_config['hidden_dim'],
            embed_dim=model_config['embed_dim'],
            dropout=model_config['dropout'],
            metadata=data.metadata(),
            num_layers=model_config['num_layers'],
            model_type=model_config['model_type']
        ),
        embeddings_dim=model_config['hidden_dim'],
        projection_dim=model_config['projection_dim']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config['lr'])

    if run_to_load:
        checkpoint = experiment_logger.load_checkpoint()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    num_neighbors = data_config['sampling']['num_neighbors']
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=experiment_config['batch_size'],
        input_nodes=('artist', torch.arange(data['artist'].num_nodes)),
        shuffle=True
    )
    # dev_loader = NeighborLoader(
    #     data,
    #     sampling,
    #     batch_size=128,
    #     input_nodes=('performance', train_indicies(data['performance'].dev_mask)),
    #     shuffle=True
    # )
    trainer = make_trainer(model, optimizer, experiment_logger)
    trainer.run(train_loader, max_epochs=35)
