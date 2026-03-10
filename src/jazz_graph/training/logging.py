from __future__ import annotations
import json
import jsonlines
import os
from datetime import datetime
from pathlib import Path
import subprocess

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_geometric.loader import LinkNeighborLoader
    from ignite.engine import Engine
    from jazz_graph.model.model import NodeClassifier, LinkPredictionModel, JazzModel


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        return None

def is_working_tree_dirty(path=".", untracked_only=True) -> bool:
    """
    Returns True if the git working tree has tracked or untracked changes.
    Returns False if the tree is clean.
    """
    cmd = ["git", "status", "--porcelain"]
    if not untracked_only:
        cmd.append("--untracked-files=no")
    result = subprocess.run(
        cmd,
        cwd=path,
        capture_output=True,
        text=True,
        check=True
    )
    return bool(result.stdout.strip())


# Thanks to ChatGPT for creating this.
class ExperimentLogger:
    """Log experiment runs."""
    def __init__(self, root="experiments", run_name=None, config=None):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = run_name or "run"
        self.run_dir = Path(root) / f"{timestamp}_{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.metrics_path = self.run_dir / "metrics.jsonl"

        if config:
            self.save_config(config)

    def save_config(self, config_dict):
        commit = {'commit': get_git_commit()}
        config_dict = config_dict | commit
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def log_metrics(self, epoch, metrics, split="train"):
        record = {
            "epoch": epoch,
            "split": split,
            **metrics
        }
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_checkpoint(self, model, optimizer, name="last.pt"):
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / name
        optmizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(model.state_dict(), path)
        print(f"Saved checkpoint to {path}")

    def save_embeddings(self, model: NodeClassifier | LinkPredictionModel):
        base_model: JazzModel = model.base_model

        # Save embeddings as tensors
        torch.save({
            'performance': base_model.performance_embed.cpu(),
            'artist': base_model.artist_embed.cpu(),
            'song': base_model.song_embed.cpu(),
        }, self.run_dir / "embeddings.pt")

        # Save node ID mappings (critical for recommendations!)
        metadata = {
            'num_performances': base_model.performance_embed.weight.shape[0],
            'num_artists': base_model.artist_embed.weight.shape[0],
            'num_songs': base_model.song_embed.weight.shape[0],
            'embedding_dim': base_model.performance_embed.weight.shape[1],
        }

        with open(self.run_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved embeddings to {self.run_dir}")

    @classmethod
    def from_run_dir(cls, run_dir):
        """
        Create a logger instance from an existing run directory.

        Useful for loading a previously saved experiment to continue logging
        or to access its config/metrics.

        Args:
            run_dir: Path to existing run directory (str or Path)

        Returns:
            ExperimentLogger instance pointing to the existing run
        """
        logger = cls.__new__(cls)  # Create instance without calling __init__
        logger.run_dir = Path(run_dir)

        if not logger.run_dir.exists():
            raise ValueError(f"Run directory does not exist: {run_dir}")

        logger.metrics_path = logger.run_dir / "metrics.jsonl"
        return logger

    def load_config(self) -> dict | None:
        """Load config from run directory."""
        config_path = self.run_dir / "config.json"

        if not config_path.exists():
            print(f"No config found at {config_path}")
            return None

        with open(config_path, 'r') as f:
            config = json.load(f)

        return config

    def load_metrics(self):
        """Load all metrics from run directory as list of dicts."""
        if not self.metrics_path.exists():
            print(f"No metrics found at {self.metrics_path}")
            return []

        metrics = []
        with open(self.metrics_path, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))

        return metrics

    def load_checkpoint(self, name="last.pt"):
        """Load checkpoint from run directory."""
        ckpt_path = self.run_dir / "checkpoints" / name

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        return checkpoint


## Handlers

def run_evaluator_handler(trainer: Engine, evaluator: Engine, loader: LinkNeighborLoader):
    evaluator.run(loader)

def console_logging(evaluator: Engine, step_name: str, trainer: Engine):
    metrics = evaluator.state.metrics
    print(f"{step_name} - Epoch[{trainer.state.epoch:03}]")
    for metric, value in metrics.items():
        print(f"  Avg. {metric}: {value:.3f}", end='; ')
    else:
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

## VISUALIZATION of logs.

def update_metrics(metrics, new_metrics):
    for key, value in new_metrics.items():
        metrics[key].append(value)

def flat_logs(logs_path: Path | str) -> dict:
    print(logs_path)
    logs_path = Path(logs_path)
    from collections import defaultdict
    splits = defaultdict(lambda: defaultdict(list))
    with jsonlines.open(logs_path / 'metrics.jsonl') as f:
        for log in f:
            # log = json.loads(f)
            split = log.pop('split')
            log.pop('epoch')
            record = splits[split]
            update_metrics(record, log)
    return splits

def plot_logs(logs_path):
    import matplotlib.pyplot as plt
    logs = flat_logs(logs_path)
    n_row = len(next(iter(logs.values())))
    fig, ax = plt.subplots(n_row, 1)
    fig.set_size_inches(5, n_row * 3)
    for split, log in logs.items():
        i = 0
        for metric, values in log.items():
            axes = ax[i]
            axes.plot(values, label=f"{split} {metric}")
            i += 1
    fig.legend()


def load_embeddings(embedding_path):
    """Load embeddings for recommendation."""
    embedding_path = Path(embedding_path)
    embeddings = torch.load(embedding_path / "embeddings.pt", weights_only=False)
    with open(embedding_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    return embeddings, metadata


def load_model(model_path):
    model_path = Path(model_path)
    model = torch.load(model_path / "checkpoints" / "last.pt")
    return model


# Credit: Claude.ai
def find_most_recent_run(root="experiments", run_name_pattern=None, return_all=False):
    """
    Find the most recent experiment run directory.

    Args:
        root: Root experiments directory
        run_name_pattern: Optional string to filter run names
        return_all: If True, return list of all runs sorted by timestamp (newest first)

    Returns:
        Path to most recent run, or list of Paths if return_all=True
    """
    root_path = Path(root)

    if not root_path.exists():
        print(f"Directory {root} does not exist")
        return [] if return_all else None

    # Get all subdirectories
    run_dirs = [d for d in root_path.iterdir() if d.is_dir()]

    # Filter by pattern
    if run_name_pattern:
        run_dirs = [d for d in run_dirs if run_name_pattern in d.name]

    if not run_dirs:
        print(f"No runs found in {root}" + (f" matching '{run_name_pattern}'" if run_name_pattern else ""))
        return [] if return_all else None

    # Parse and sort by timestamp
    runs_with_timestamps = []
    for dir_path in run_dirs:
        try:
            # Extract YYYY-MM-DD_HH-MM-SS from start of dirname
            parts = dir_path.name.split('_')
            if len(parts) >= 2:
                timestamp_str = f"{parts[0]}_{parts[1]}"
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                runs_with_timestamps.append((timestamp, dir_path))
        except ValueError:
            # Skip directories that don't match the timestamp format
            continue

    if not runs_with_timestamps:
        print(f"No valid timestamped runs found")
        return [] if return_all else None

    # Sort by timestamp (newest first)
    runs_with_timestamps.sort(reverse=True, key=lambda x: x[0])

    if return_all:
        return [path for _, path in runs_with_timestamps]
    else:
        return runs_with_timestamps[0][1]


# Helper to load checkpoint from most recent run
def load_most_recent_checkpoint(model, root="experiments", run_name_pattern=None,
                                checkpoint_name="last.pt"):
    """Load checkpoint from most recent run."""
    recent_run = find_most_recent_run(root, run_name_pattern)

    if not recent_run:
        print("No recent run found")
        return None

    checkpoint_path = recent_run / "checkpoints" / checkpoint_name

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        # Try artifacts directory
        checkpoint_path = recent_run / "artifacts" / checkpoint_name

    if not checkpoint_path.exists():
        print(f"No checkpoint found in {recent_run}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    print(f"  Run: {recent_run.name}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")

    return checkpoint
