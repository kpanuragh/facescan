"""Publication-quality result visualizations."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class ResultPlotter:
    """Create publication-quality plots for rPPG results."""

    def __init__(self, output_dir: Path = "results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def bland_altman(self, mean_vals: np.ndarray, diff: np.ndarray,
                     mean_diff: float, upper_loa: float, lower_loa: float,
                     metric_name: str, unit: str):
        """Plot Bland-Altman agreement."""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(mean_vals, diff, alpha=0.6, s=30)
        ax.axhline(mean_diff, color="r", linestyle="-",
                   label=f"Mean: {mean_diff:.2f}")
        ax.axhline(upper_loa, color="r", linestyle="--",
                   label=f"±1.96 SD: [{lower_loa:.2f}, {upper_loa:.2f}]")
        ax.axhline(lower_loa, color="r", linestyle="--")
        ax.axhline(0, color="k", linestyle="-", alpha=0.3, linewidth=0.5)

        ax.set_xlabel(f"Mean {metric_name} ({unit})", fontsize=12)
        ax.set_ylabel(f"Difference {metric_name} ({unit})", fontsize=12)
        ax.set_title(f"Bland-Altman Plot: {metric_name}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"bland_altman_{metric_name.lower()}.png", dpi=300)
        plt.close()
        print(f"Saved: bland_altman_{metric_name.lower()}.png")

    def confidence_calibration(self, confidence: np.ndarray, error: np.ndarray, metric_name: str):
        """Plot confidence vs actual error."""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(confidence, error, alpha=0.6, s=30)
        ax.set_xlabel("Predicted Confidence", fontsize=12)
        ax.set_ylabel("Absolute Error", fontsize=12)
        ax.set_title(f"Confidence Calibration: {metric_name}", fontsize=14)
        ax.grid(alpha=0.3)

        # Add trend line if data available
        if len(confidence) > 1:
            z = np.polyfit(confidence, error, 1)
            p = np.poly1d(z)
            x_line = np.linspace(confidence.min(), confidence.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Trend")
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f"confidence_calibration_{metric_name.lower()}.png", dpi=300)
        plt.close()
        print(f"Saved: confidence_calibration_{metric_name.lower()}.png")

    def training_curves(self, epochs: list, train_loss: list, val_loss: list):
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, train_loss, label="Training Loss", marker="o", markersize=3)
        ax.plot(epochs, val_loss, label="Validation Loss", marker="s", markersize=3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Progress", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300)
        plt.close()
        print(f"Saved: training_curves.png")
