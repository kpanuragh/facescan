"""rPPG model architecture with shared feature extractor and task-specific heads."""

import torch
import torch.nn as nn
from typing import Tuple


class SharedFeatureExtractor(nn.Module):
    """Shared CNN feature extractor for rPPG using 3D convolutions."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim

        # Input: (B, T, C, H, W) = (B, 30, 3, 128, 128)
        # Use 3D convolutions to capture temporal + spatial patterns
        self.features = nn.Sequential(
            # Conv3D block 1: 30 frames → 15 frames
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            # Conv3D block 2: 15 frames → 8 frames
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            # Conv3D block 3: 8 frames → 4 frames
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            # Global average pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # FC layers to output feature vector
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) = (B, 30, 3, 128, 128)

        Returns:
            features: (B, output_dim)
        """
        # Rearrange to (B, C, T, H, W) for Conv3D
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TaskHead(nn.Module):
    """Task-specific head for metric prediction + confidence."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # [value, confidence_logit]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) features

        Returns:
            value: (B,) metric value
            confidence: (B,) confidence score [0, 1]
        """
        out = self.net(x)
        value = out[:, 0]
        confidence = torch.sigmoid(out[:, 1])  # Sigmoid to [0, 1]
        return value, confidence


class rPPGModel(nn.Module):
    """Complete rPPG model with shared extractor + 3 task heads."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.extractor = SharedFeatureExtractor(output_dim=feature_dim)
        self.hr_head = TaskHead(input_dim=feature_dim, hidden_dim=hidden_dim)
        self.rr_head = TaskHead(input_dim=feature_dim, hidden_dim=hidden_dim)
        self.spo2_head = TaskHead(input_dim=feature_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, 
                                                   torch.Tensor, torch.Tensor,
                                                   torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) video frames

        Returns:
            hr_val, hr_conf, rr_val, rr_conf, spo2_val, spo2_conf
        """
        features = self.extractor(x)
        hr_val, hr_conf = self.hr_head(features)
        rr_val, rr_conf = self.rr_head(features)
        spo2_val, spo2_conf = self.spo2_head(features)

        return hr_val, hr_conf, rr_val, rr_conf, spo2_val, spo2_conf
