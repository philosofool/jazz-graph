import os
from pathlib import Path

from ignite.engine import Engine, Events
from ignite.handlers import ProgressBar, EarlyStopping
from ignite.metrics import RunningAverage

import torch
from torch_geometric.nn import MetaPath2Vec
from torch_geometric import seed_everything

from jazz_graph.data.graph_builder.graph_builder import CreateTensors, make_jazz_data
from jazz_graph.training.logging import ExperimentLogger
from jazz_graph.training.loop import console_logging, log_experiment_handler, save_checkpoint_handler


# A metapath cycle Performance -> Artist -> Performance -> Song -> Performance.
# Any cycle of (src_node_type, relation, dst_node_type) edge triples present in
# the graph's edge_index_dict can be substituted here. metapath[0][0] must
# match metapath[-1][-1] so that walk_length can extend past one cycle, and
# the node type used to start random walks (and thus the node type embedded
# with the most signal) is metapath[0][0].

# NOTE: this is a bit unusual; path is not theoretically motivated.
# Songs that are standards can connect in surprising ways.
# Empirically, this worked pretty well, but expect future revisions.

DEFAULT_METAPATH: list[tuple[str, str, str]] = [
    ('performance', 'rev_performs', 'artist'),
    ('artist', 'performs', 'performance'),
    ('performance', 'performing', 'song'),
    ('song', 'rev_performing', 'performance'),
]


class MetaPath2VecTrainingLogic:
    """Define the training step for a MetaPath2Vec model."""

    def __init__(self, model: MetaPath2Vec, optimizer):
        self.device = next(model.parameters()).device
        self.model = model
        self.optimizer = optimizer

    def train_step(self, engine, batch):
        pos_rw, neg_rw = batch
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


def make_trainer(
    model: MetaPath2Vec,
    optimizer,
    experiment_logger: ExperimentLogger,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
) -> Engine:
    trainer_logic = MetaPath2VecTrainingLogic(model, optimizer)
    trainer = Engine(trainer_logic.train_step)

    RunningAverage(output_transform=lambda out: out['loss']).attach(trainer, 'loss')

    progress_bar = ProgressBar()
    progress_bar.attach(trainer, metric_names=['loss'])

    trainer.add_event_handler(Events.EPOCH_COMPLETED, console_logging, 'Training', trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_experiment_handler, experiment_logger, 'train', trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_checkpoint_handler, experiment_logger, model, optimizer)

    def score_function(engine: Engine):
        return engine.state.metrics['loss']

    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        score_function=score_function,
        trainer=trainer,
        mode='min',
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
    return trainer


if __name__ == '__main__':
    random_seed = 42
    seed_everything(random_seed)
    models_dir = '/workspace/local_data/graph_parquet'
    assert os.path.exists(models_dir)
    create = CreateTensors(models_dir)
    data = make_jazz_data(create)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_to_load: str | None = None

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
            'data_config': {'dataset': models_dir},
            # Any cycle of edge types from data.edge_index_dict can be used here
            # to change what kind of similarity the embeddings capture.
            'metapath': DEFAULT_METAPATH,
            'embedding_dim': 64,
            'walk_length': 16,
            'context_size': 7,
            'walks_per_node': 3,
            'num_negative_samples': 3,
            'batch_size': 128,
            'lr': .01,
            'epochs': 100,
            # Stop once the epoch loss (RunningAverage over all batches) hasn't
            # improved by more than min_delta for `patience` epochs in a row.
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 0.001,
        }
        print(f"Initializing new model with configuration:\n{experiment_config}")
        experiment_logger = ExperimentLogger(
            root='/workspace/experiments',
            run_name=f'metapath2vec_{os.path.basename(models_dir)}',
            config=experiment_config
        )

    metapath = [tuple(edge_type) for edge_type in experiment_config['metapath']]
    num_nodes_dict = {
        'performance': data['performance'].num_nodes,
        'artist': data['artist'].num_nodes,
        'song': data['song'].num_nodes,
    }

    model = MetaPath2Vec(
        data.edge_index_dict,
        embedding_dim=experiment_config['embedding_dim'],
        metapath=metapath,
        walk_length=experiment_config['walk_length'],
        context_size=experiment_config['context_size'],
        walks_per_node=experiment_config['walks_per_node'],
        num_negative_samples=experiment_config['num_negative_samples'],
        num_nodes_dict=num_nodes_dict,
        sparse=True,
    )
    model.to(device)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=experiment_config['lr'])

    if run_to_load:
        checkpoint = experiment_logger.load_checkpoint()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loader = model.loader(batch_size=experiment_config['batch_size'], shuffle=True)

    trainer = make_trainer(
        model,
        optimizer,
        experiment_logger,
        early_stopping_patience=experiment_config['early_stopping_patience'],
        early_stopping_min_delta=experiment_config['early_stopping_min_delta'],
    )
    trainer.run(loader, max_epochs=experiment_config['epochs'])

    experiment_logger.save_metapath2vec_embeddings(model)
