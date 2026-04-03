from collections.abc import Callable

from ignite.engine import Engine, Events
from ignite.handlers import ProgressBar, EarlyStopping
from ignite.metrics import RunningAverage

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader


from jazz_graph.training.views import performance_album_map
from jazz_graph.metrics.embedding_metrics import AlignmentLoss, UniformityLoss, MultiPositiveAlignment, EmbeddingStd
from jazz_graph.model.model import UnsupervisedJazzModel
from jazz_graph.training.logging import ExperimentLogger
from jazz_graph.training.loss import nt_xent_loss, nt_xent_loss_with_masking


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
        elif isinstance(target_input, torch.Tensor):
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

class Augment:
    def __init__(self, batch, augment):
        self.augment = augment
        self.batch = batch

    def view(self):
        self.augment(self.batch)


class UnsupervisedGNNTrainingLogic:
    """Define training step and eval steps."""
    def __init__(self, model: UnsupervisedJazzModel, optimizer, temperature, augment: Callable[[HeteroData], HeteroData]):
        self.device = next(model.parameters()).device
        self.model = model
        self.optimizer = optimizer
        self.temperature = temperature
        self.augment = augment

    # def augment(self, batch):
    #     ...

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
                'h2': h2_dict[node_type].detach(),
                'z1': z2_dict[node_type].detach(),
                'z2': z2_dict[node_type].detach()
            }
        return results


class UnsupervisedGNNTrainingLogicMatchAlbum:
    """Define training step and eval steps in a SimCLR style learning over albums."""
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
        z1_dict: dict[str, torch.Tensor] = self.model.project(h1_dict)
        album_ids = batch['performance'].album_id
        matching_album_mask = album_ids.reshape(-1, 1) == album_ids.reshape(1, -1)

        loss = nt_xent_loss_with_masking(z1_dict['performance'], matching_album_mask, self.temperature)
        loss.backward()
        self.optimizer.step()
        results = {'performance': {'loss': loss.item(), 'z1': z1_dict['performance'].detach(), 'mask': matching_album_mask}}
        return results

class DualLossUnsupervisedTraining:
    def __init__(self, model: UnsupervisedJazzModel, optimizer, temperature, augment: Callable[[HeteroData], HeteroData]):
        self.device = next(model.parameters()).device
        self.model = model
        self.optimizer = optimizer
        self.temperature = temperature
        self.augment = augment
        self.alpha = .5

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

        h1_dict_album: dict[str, torch.Tensor] = self.model.encode(batch)
        z1_dict_album: dict[str, torch.Tensor] = self.model.project(h1_dict_album)
        album_ids = batch['performance'].album_id
        matching_album_mask = album_ids.reshape(-1, 1) == album_ids.reshape(1, -1)

        loss_album = nt_xent_loss_with_masking(z1_dict_album['performance'], matching_album_mask, self.temperature)

        h1_dict_ablation: dict[str, torch.Tensor] = self.model.encode(self.augment(batch))
        h2_dict_ablation: dict[str, torch.Tensor] = self.model.encode(self.augment(batch))
        z1_dict_ablation: dict[str, torch.Tensor] = self.model.project(h1_dict_ablation)
        z2_dict_ablation: dict[str, torch.Tensor] = self.model.project(h2_dict_ablation)
        losses_ablation = {
            node_type: nt_xent_loss(z1_dict_ablation[node_type], z2_dict_ablation[node_type], self.temperature)
            for node_type in z1_dict_ablation
        }

        total_loss = self.alpha * sum(losses_ablation.values()) / len(z1_dict_ablation) + (1 - self.alpha) * loss_album
        total_loss.backward()  # pyright: ignore [reportAttributeAccessIssue]
        self.optimizer.step()

        results_ablation = {}
        for node_type in z1_dict_ablation:
            results_ablation[node_type] = {
                'loss': losses_ablation[node_type].item(),
                'h1': h1_dict_ablation[node_type].detach(),
                'h2': h2_dict_ablation[node_type].detach(),
                'z1': z1_dict_ablation[node_type].detach(),
                'z2': z2_dict_ablation[node_type].detach()
            }


        results = {
            'total_loss': total_loss.item(),
            'album': {'performance': {'loss': loss_album.item(), 'z1': z1_dict_album['performance'].detach(), 'mask': matching_album_mask}},
            'ablation': results_ablation
        }
        return results

    def decay_loss_weight(self, *args, **kwargs):
        alpha = self.alpha * .8
        if alpha > .1:
            self.alpha = alpha
        else:
            self.alpha = .1

    def trainer(self, experiment_logger):
        """Build a trainer suitable for self-supervized learning with pairs as similarities.

        For example, use with a drop_edge augmentation.
        """
        node_types = ['performance', 'artist', 'song']
        trainer = Engine(self.train_step)


        def create_metrics_for_node_ablation(node_type):
            """Create metrics for a specific node type."""
            return {
                f"{node_type}_loss": RunningAverage(
                    output_transform=lambda out: out['ablation'][node_type]['loss']
                ),
                f"{node_type}_alignment": AlignmentLoss(
                    output_transform=lambda out: (out['ablation'][node_type]['z1'], out['ablation'][node_type]['z2'])
                ),
            f"{node_type}_uniformity": UniformityLoss(
                    output_transform=lambda out: out['ablation'][node_type]['z1']
                ),
                f"{node_type}_embedding_std": EmbeddingStd(
                    output_transform=lambda out: out['ablation'][node_type]['z1'])
            }

        def create_metrics_for_node_album(node_type):
            return {
                f"{node_type}_loss_album": RunningAverage(
                    output_transform=lambda out: out['album'][node_type]['loss']
                ),
                f"{node_type}_alignment_album": MultiPositiveAlignment(
                    output_transform=lambda out: (out['album'][node_type]['z1'], out['album'][node_type]['mask'])
                ),
                f"{node_type}_uniformity_album": UniformityLoss(
                    output_transform=lambda out: out['album'][node_type]['z1']
                ),
                f"{node_type}_embedding_std_album": EmbeddingStd(
                    output_transform=lambda out: out['album'][node_type]['z1'])
            }

        metrics = {}
        for node_type in node_types:
            metrics.update(create_metrics_for_node_ablation(node_type))
        metrics.update(create_metrics_for_node_album('performance'))
        metrics.update({'total_loss': RunningAverage(
                    output_transform=lambda out: out['total_loss']
        )})

        for name, metric in metrics.items():
            metric.attach(trainer, name)

        progress_bar = ProgressBar()
        progress_bar.attach(trainer, metric_names=[f'{node_type}_loss' for node_type in node_types])

        def score_function(engine: Engine):
            if engine.state.epoch <= 2:
                return -1e6 + engine.state.epoch
            return -engine.state.output['ablation']['performance']['loss']    # pyright: ignore

        handler = EarlyStopping(
            patience=5,
            min_delta=0.02,
            score_function=score_function,
            trainer=trainer
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, console_logging_self_supervised, 'Training', trainer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_experiment_handler, experiment_logger, 'train', trainer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, save_checkpoint_handler, experiment_logger, self.model, self.optimizer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.decay_loss_weight)
        return trainer


def run_evaluator_handler(trainer: Engine, evaluator: Engine, loader: LinkNeighborLoader):
    evaluator.run(loader)


def console_logging(evaluator: Engine, step_name: str, trainer: Engine):
    metrics = evaluator.state.metrics
    print(f"{step_name} - Epoch[{trainer.state.epoch:03}]")
    for metric, value in metrics.items():
        print(f"  Avg. {metric}: {value:.3f}", end='; ')
    else:
        print()


def console_logging_self_supervised(evaluator: Engine, step_name: str, trainer: Engine):
    """Console logging for unsupervised experiments."""
    metrics = evaluator.state.metrics.copy()
    print(f"{step_name} - Epoch[{trainer.state.epoch:03}]")
    def order_keys(key):
        if 'total_l' in key:
            return -1
        if 'performance' in key:
            return 0
        if 'artist' in key:
            return 1
        if 'song' in key:
            return 2
        return 3
    value = metrics.pop('total_loss')
    print(f"  Avg. total loss: {value:.3f}")
    metrics = sorted(metrics.items(),  key=lambda x: order_keys(x[0]))
    for i, (metric, value) in enumerate(metrics):
        if i % 4 == 0 and i != 0:
            # add space.
            print()
        print(f"  Avg. {metric}: {value:.3f}", end='; ')

    print()


def save_embeddings_handler(engine, logger: ExperimentLogger, model):
    """Save embeddings at end of training."""
    logger.save_embeddings(model)


def save_checkpoint_handler(engine, logger: ExperimentLogger, model, optimizer):
    """Save checkpoint at end of epoch."""
    logger.save_checkpoint(
        model,
        optimizer=optimizer,
        name="last.pt"
    )


def log_experiment_handler(engine, logger: ExperimentLogger, split, trainer: Engine):
    """Log experiment results (usually each epoch) to files."""
    metrics = engine.state.metrics
    logger.log_metrics(trainer.state.epoch, metrics, split)


def binary_output_transform(output: dict[str, torch.Tensor]) -> tuple:
    """Return y_true and y_pred as binary classifications."""
    y_pred = (output["y_pred"] > 0).long()
    y_true = output["y_true"]
    return y_pred, y_true
