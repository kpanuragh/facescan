"""Training metrics logger."""

import json
from pathlib import Path


class TrainingLogger:
    """Log training metrics to JSON for analysis."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch": []
        }

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Log metrics for an epoch."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["learning_rate"].append(lr)

    def save(self):
        """Save logs to JSON."""
        with open(self.log_dir / "training_log.json", "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training logs saved to {self.log_dir / 'training_log.json'}")
