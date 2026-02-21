import json
import os
from datetime import datetime
from pathlib import Path
import subprocess

import torch

# Thanks to ChatGPT for creating this.


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        return None


class JSONRunLogger:
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

    def save_checkpoint(self, model, name="last.pt"):
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / name
        torch.save(model.state_dict(), path)