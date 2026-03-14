"""Frame normalization and augmentation for rPPG analysis."""

import numpy as np
import cv2


class FrameNormalizer:
    """Normalize and augment frames for rPPG analysis."""

    def __init__(self, clamp_value: float = 5.0):
        self.clamp_value = clamp_value

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame using mean-std normalization per channel.

        Args:
            frame: (H, W, 3) RGB frame

        Returns:
            normalized: (H, W, 3) normalized frame
        """
        frame = frame.astype(np.float32)

        # Per-channel normalization
        normalized = np.zeros_like(frame)
        for c in range(3):
            channel = frame[:, :, c]
            mean = channel.mean()
            std = channel.std()
            if std > 1e-6:
                normalized[:, :, c] = (channel - mean) / std
            else:
                normalized[:, :, c] = channel - mean

        # Clamp to [-clamp_value, clamp_value]
        normalized = np.clip(normalized, -self.clamp_value, self.clamp_value)

        return normalized

    def augment_brightness(self, frame: np.ndarray, factor: float = 0.1) -> np.ndarray:
        """Add random brightness shift."""
        shift = np.random.uniform(-factor, factor)
        augmented = frame.astype(np.float32) * (1 + shift)
        return np.clip(augmented, 0, 255).astype(np.uint8)

    def augment_contrast(self, frame: np.ndarray, factor: float = 0.1) -> np.ndarray:
        """Add random contrast adjustment."""
        factor = 1 + np.random.uniform(-factor, factor)
        augmented = (frame.astype(np.float32) * factor)
        return np.clip(augmented, 0, 255).astype(np.uint8)

    def augment_flip(self, frame: np.ndarray) -> np.ndarray:
        """Randomly flip frame horizontally."""
        if np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)
        return frame

    def augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        frame = self.augment_flip(frame)
        frame = self.augment_brightness(frame)
        frame = self.augment_contrast(frame)
        return frame

    def process_frame(self, frame: np.ndarray, augment: bool = True) -> np.ndarray:
        """Augment (if training) and normalize frame."""
        if augment:
            frame = self.augment_frame(frame)
        return self.normalize_frame(frame)
