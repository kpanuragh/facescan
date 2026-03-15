"""Create reproducible train/val/test data splits."""

import json
from pathlib import Path
from typing import Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Create reproducible subject-level train/val/test splits."""

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, random_seed: int = 42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.random_seed = random_seed

    def create_splits(self, num_subjects: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create subject-level splits to prevent data leakage.

        Args:
            num_subjects: Total number of subjects in dataset

        Returns:
            train_subj: Subject indices for training
            val_subj: Subject indices for validation
            test_subj: Subject indices for testing
        """
        np.random.seed(self.random_seed)

        # Assign subjects to splits (subject-level, not frame-level)
        subjects = np.arange(num_subjects)

        train_subj, temp_subj = train_test_split(
            subjects,
            test_size=1 - self.train_ratio,
            random_state=self.random_seed
        )

        val_subj, test_subj = train_test_split(
            temp_subj,
            test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
            random_state=self.random_seed
        )

        return train_subj, val_subj, test_subj

    def save_splits(self, splits: Tuple[np.ndarray, np.ndarray, np.ndarray], output_dir: Path):
        """Save split assignments to JSON."""
        train_subj, val_subj, test_subj = splits
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits_dict = {
            "train": train_subj.tolist(),
            "validation": val_subj.tolist(),
            "test": test_subj.tolist(),
            "meta": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "random_seed": self.random_seed
            }
        }

        with open(output_dir / "splits.json", "w") as f:
            json.dump(splits_dict, f, indent=2)

        print(f"Splits saved to {output_dir / 'splits.json'}")
        print(f"  Train: {len(train_subj)} subjects ({self.train_ratio*100:.0f}%)")
        print(f"  Val:   {len(val_subj)} subjects ({self.val_ratio*100:.0f}%)")
        print(f"  Test:  {len(test_subj)} subjects ({self.test_ratio*100:.0f}%)")

    @staticmethod
    def load_splits(splits_file: Path) -> dict:
        """Load splits from JSON file."""
        with open(splits_file, "r") as f:
            return json.load(f)


# Convenience function for simple usage
def create_splits(dataset, train_ratio: float = 0.7, val_ratio: float = 0.15, random_seed: int = 42, total_samples: int = None) -> Tuple[List, List, List]:
    """
    Create train/val/test splits from dataset.

    Args:
        dataset: HuggingFace dataset or similar object with __len__
        train_ratio: Fraction of data for training (default 0.7)
        val_ratio: Fraction of data for validation (default 0.15)
        random_seed: Random seed for reproducibility
        total_samples: Override dataset size (for streaming datasets)

    Returns:
        train_indices, val_indices, test_indices: Lists of indices for each split
    """
    # Handle streaming datasets or when size is unknown
    if total_samples is None:
        try:
            num_samples = len(dataset)
            # If dataset is streaming, it may report size as 1 (first batch)
            # Default to MCD-rPPG size if streaming
            if hasattr(dataset, 'streaming') and dataset.streaming and num_samples <= 1:
                num_samples = 3600  # MCD-rPPG has 3600 samples
            elif num_samples <= 1:
                num_samples = 3600  # Fallback to MCD-rPPG size
        except:
            num_samples = 3600  # Default to MCD-rPPG size
    else:
        num_samples = total_samples

    splitter = DatasetSplitter(train_ratio=train_ratio, val_ratio=val_ratio, random_seed=random_seed)
    train_subj, val_subj, test_subj = splitter.create_splits(num_samples)

    return train_subj.tolist(), val_subj.tolist(), test_subj.tolist()
