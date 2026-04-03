import json
from pathlib import Path
from jazz_graph.training.logging import is_valid_experiment

def write_valid_experiment(base: Path):
    """Helper to stamp out a valid experiment directory."""
    (base / "config.json").write_text(json.dumps({"lr": 0.01}))
    (base / "metrics.jsonl").write_text('{"loss": 1.0}\n{"loss": 0.5}\n')
    (base / "checkpoints").mkdir()

def test_valid_experiment(tmp_path):
    write_valid_experiment(tmp_path)
    assert is_valid_experiment(tmp_path) is True

def test_missing_config(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"loss": 1.0}\n{"loss": 0.5}\n')
    (tmp_path / "checkpoints").mkdir()
    assert is_valid_experiment(tmp_path) is False

def test_missing_metrics(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"lr": 0.01}))
    (tmp_path / "checkpoints").mkdir()
    assert is_valid_experiment(tmp_path) is False

def test_too_few_epochs(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"lr": 0.01}))
    (tmp_path / "metrics.jsonl").write_text('{"loss": 1.0}\n')  # only 1 epoch
    (tmp_path / "checkpoints").mkdir()
    assert is_valid_experiment(tmp_path) is False
