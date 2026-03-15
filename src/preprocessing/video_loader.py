"""Load and cache MCD-rPPG dataset from HuggingFace."""

import os
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import cv2
import json


class VideoLoader:
    """Load and cache MCD-rPPG dataset from HuggingFace."""

    def __init__(self, dataset_id: str = "kyegorov/mcd_rppg", cache_dir: str = None):
        self.dataset_id = dataset_id
        self.cache_dir = cache_dir or "./data/raw"
        self.dataset = None

    def download_dataset(self):
        """Download MCD-rPPG dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required. Install with: pip install datasets")

        print(f"Downloading {self.dataset_id}...")
        try:
            # Try loading with trust_remote_code for custom processing
            self.dataset = load_dataset(
                self.dataset_id,
                cache_dir=self.cache_dir,
                streaming=False,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Standard loading failed: {e}")
            print("Retrying with streaming mode...")
            # Fallback: use streaming mode for incremental loading
            self.dataset = load_dataset(
                self.dataset_id,
                cache_dir=self.cache_dir,
                streaming=True,
                trust_remote_code=True
            )

        print(f"Dataset downloaded: {len(self.dataset) if not self.dataset.streaming else 'streaming'} samples")
        return self.dataset

    def load_video_frames(self, video_path: str, frame_limit: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Load video frames from file.

        Args:
            video_path: Path to video file
            frame_limit: Maximum number of frames to load

        Returns:
            frames: (T, H, W, 3) numpy array in RGB
            metadata: dict with video info
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1

            if frame_limit and frame_count >= frame_limit:
                break

        cap.release()

        metadata = {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "loaded_frames": len(frames),
        }

        return np.array(frames), metadata

    def get_ground_truth(self, sample: Dict) -> Dict:
        """Extract ground truth signals from MCD-rPPG sample."""
        ground_truth = {
            "ppg": sample.get("ppg"),                 # PPG signal (100 Hz)
            "ecg": sample.get("ecg"),                 # ECG for HR reference
            "respiratory": sample.get("respiratory"), # Respiratory signal
            "spo2": sample.get("spo2"),               # Blood oxygen saturation
            "timestamp": sample.get("timestamp")
        }
        return ground_truth

    def load_dataset(self):
        """
        Load MCD-rPPG dataset from HuggingFace.

        Returns:
            dataset: HuggingFace dataset object with all samples
        """
        if self.dataset is None:
            self.download_dataset()
        return self.dataset
