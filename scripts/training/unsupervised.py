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
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphConv, SAGEConv, to_hetero, HeteroConv
from torch_geometric import transforms as T
from torch_geometric import seed_everything

from jazz_graph.data.fetch import fetch_recording_traits
from jazz_graph.data.reporting import inspect_degrees
from jazz_graph.etl.transforms import map_array, map_by_index
from jazz_graph.metrics.embedding_metrics import AlignmentLoss, UniformityLoss, MultiPositiveAlignment, EmbeddingStd
from jazz_graph.model.model import UnsupervisedJazzModel
from jazz_graph.training.inspect import analyze_model_embeddings
from jazz_graph.training.views import drop_random_nodes_and_edges
from jazz_graph.training.logging import (
    ExperimentLogger,
    load_model
)
from jazz_graph.data.graph_builder.graph_builder import CreateTensors, prune_isolated_nodes, make_jazz_data
from jazz_graph.model.model import JazzModel, LinkPredictionModel, NodeClassifier
from jazz_graph.training.logging import plot_logs
from jazz_graph.training.views import MatchAlbumAugmentation, performance_album_map
from jazz_graph.training.loss import nt_xent_loss_with_masking
from jazz_graph.training.loop import NeighborLoaderWithJitter, UnsupervisedGNNTrainingLogic, UnsupervisedGNNTrainingLogicMatchAlbum, binary_output_transform, console_logging, log_experiment_handler, run_evaluator_handler, save_checkpoint_handler, save_embeddings_handler, console_logging_self_supervised



def make_album_match_trainer(model, optimizer, experiment_logger: ExperimentLogger):
    """Make a trainer that matches on semantic similarity by album."""
    node_types = ['performance']
    experiment_config = experiment_logger.load_config()
    if experiment_config is None:
        raise ValueError("Logger must have an experiment config.")
    models_dir = experiment_config['data_config']['dataset']

    trainer_logic = UnsupervisedGNNTrainingLogicMatchAlbum(
        model,
        optimizer,
        experiment_config['temperature']
    )

    trainer = Engine(trainer_logic.train_step)

    def create_metrics_for_node_type(node_type):
        """Create metrics for a specific node type."""
        return {
            f"{node_type}_loss": RunningAverage(
                output_transform=lambda out: out[node_type]['loss']
            ),
            f"{node_type}_alignment": MultiPositiveAlignment(
                output_transform=lambda out: (out[node_type]['z1'], out[node_type]['mask'])
            ),
            f"{node_type}_uniformity": UniformityLoss(
                output_transform=lambda out: out[node_type]['z1']
            ),
            f"{node_type}_embedding_std": EmbeddingStd(
                output_transform=lambda out: out[node_type]['z1'])
        }

    metrics = {}
    metrics.update(create_metrics_for_node_type('performance'))
    for name, metric in metrics.items():
        metric.attach(trainer, name)

    progress_bar = ProgressBar()
    progress_bar.attach(trainer, metric_names=[f'{node_type}_loss' for node_type in node_types])

    trainer.add_event_handler(Events.EPOCH_COMPLETED, console_logging, 'Training', trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_experiment_handler, experiment_logger, 'train', trainer)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), make_analyze_embeddings(models_dir), model)
    # trainer.add_event_handler(Events.COMPLETED, save_embeddings_handler, experiment_logger, model)
    trainer.add_event_handler(Events.COMPLETED, save_checkpoint_handler, experiment_logger, model, optimizer)
    return trainer

def make_trainer(model, optimizer, experiment_logger: ExperimentLogger):
    """Build a trainer suitable for self-supervized learning with pairs as similarities.

    For example, use with a drop_edge augmentation.
    """
    node_types = ['performance', 'artist', 'song']
    experiment_config = experiment_logger.load_config()
    if experiment_config is None:
        raise ValueError("Logger must have an experiment config.")
    if experiment_config.get('drop_edge_prob', None) is None:
        raise ValueError("Expected drop edge probability in config.")
    models_dir = experiment_config['data_config']['dataset']

    trainer_logic = UnsupervisedGNNTrainingLogic(
        model,
        optimizer,
        experiment_config['temperature'],
        augment=lambda data: drop_random_nodes_and_edges(data, experiment_config['drop_edge_prob'])
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
                output_transform=lambda out: out[node_type]['z1']
            ),
            f"{node_type}_embedding_std": EmbeddingStd(
                output_transform=lambda out: out[node_type]['z1'])
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
    # trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), make_analyze_embeddings(models_dir), model)
    # trainer.add_event_handler(Events.COMPLETED, save_embeddings_handler, experiment_logger, model)
    trainer.add_event_handler(Events.COMPLETED, save_checkpoint_handler, experiment_logger, model, optimizer)
    return trainer

def train_indecies(mask):
    """Helper to take a mask array and return the relevant nodes."""
    # I think there's a torch_geometric helper for this.
    num_nodes = mask.shape[0]
    all_node_indicies = torch.arange(num_nodes)
    return all_node_indicies[mask]

def make_analyze_embeddings(models_dir) -> Callable:

    assert os.path.exists(models_dir)
    create = CreateTensors(models_dir)
    data = make_jazz_data(create)

    def analyze(engine, model):
        analyze_model_embeddings(model, data)

    return analyze


if __name__ == '__main__':
    random_seed = 42
    seed_everything(random_seed)
    models_dir = '/workspace/local_data/graph_parquet'
    assert os.path.exists(models_dir)
    create = CreateTensors(models_dir)
    data = make_jazz_data(create)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_to_load: str|None = None

    if run_to_load:
        assert ' ' not in run_to_load, "Expected no spaces."
        if run_to_load == 'most_recent':
            run_to_load = max(Path("/workspace/experiments").iterdir(), key=lambda p: p.stat().st_mtime)
        experiment_logger = ExperimentLogger.from_run_dir(run_to_load)
        experiment_config = experiment_logger.load_config()
        if experiment_config is None:
            raise ValueError("Only experiment loggers with configs can be used.")
        print(f"Initializing existing model from checkpoint at {run_to_load}. (See config file for details.)")
    else:
        experiment_config = {
            'random_seed': random_seed,
            'data_config': {
                'dataset': models_dir,
                # Sampling is important here. We want to make sure that the neighborhodd
                # for high degree nodes doesn't become too noisy. Artist and song nodes can
                # be very high degree. Basially, get a sizable neighborhood for each node to
                # reduce this noise.
                'sampling': {'num_neighbors': [20, 15, 8]},
                'input_node_type': 'performance'
            },
            'model': {
                'num_performances': data['performance'].num_nodes,
                'num_artists': data['artist'].num_nodes,
                'num_songs': data['song'].num_nodes,
                'hidden_dim': 128,
                'embed_dim': 64,
                'projection_dim': 64,
                'dropout': 0.0,
                'model_type': 'sage',
                'num_layers': 3,
            },
            'drop_edge_prob': None,
            'dataset': models_dir,
            'lr': .003,
            'batch_size': 128,
            'temperature': .25
        }
        print(f"Initializing new model with configuration:\n{experiment_config}")
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
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config['lr'])

    if run_to_load:
        checkpoint = experiment_logger.load_checkpoint()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    num_neighbors = data_config['sampling']['num_neighbors']
    input_node_type = data_config['input_node_type']

    year_feature = data['performance'].x[:, 0]
    trainer = make_album_match_trainer(model, optimizer, experiment_logger)
    # trainer = make_album_match_trainer(model, optimizer, experiment_logger)

    if False:
        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=experiment_config['batch_size'],
            input_nodes=(input_node_type, torch.arange(data[input_node_type].num_nodes)),
            # shuffle=True
            #input_nodes=ordered_nodes,
            shuffle=True,  # sampler already handles order
        )
    else:
        train_loader = NeighborLoaderWithJitter(
            data,
            (input_node_type, year_feature),
            num_neighbors=num_neighbors,
            batch_size=experiment_config['batch_size'],
        )

        trainer.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: train_loader.set_epoch(engine.state.epoch)
        )

    trainer.run(train_loader, max_epochs=3)
