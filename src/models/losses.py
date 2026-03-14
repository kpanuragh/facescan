"""Multi-task loss with confidence calibration for rPPG model."""

import torch
import torch.nn as nn
from typing import Tuple


class MultiTaskLoss(nn.Module):
    """Multi-task loss with confidence calibration."""

    def __init__(self, weights: dict = None):
        super().__init__()
        self.weights = weights or {
            "hr": 1.0,
            "rr": 1.0,
            "spo2": 1.0,
            "confidence": 0.5
        }
        self.mse_loss = nn.MSELoss()

    def calibration_loss(self, confidence: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        Binary cross-entropy between confidence and normalized error.

        Lower confidence should correspond to higher error.
        Higher confidence should correspond to lower error.
        """
        # Normalize error to [0, 1] range
        # Assume max reasonable error is ~10 bpm for HR
        error_normalized = torch.clamp(error / 10.0, 0, 1)

        # Target: confidence should be high when error is low
        # So target = 1 - error_normalized
        target = 1 - error_normalized
        kl_loss = nn.functional.binary_cross_entropy(confidence, target.detach())
        return kl_loss

    def forward(self, outputs: Tuple, targets: Tuple) -> torch.Tensor:
        """
        Calculate multi-task loss.

        Args:
            outputs: (hr_val, hr_conf, rr_val, rr_conf, spo2_val, spo2_conf)
            targets: (hr_target, rr_target, spo2_target)

        Returns:
            total_loss: scalar loss value
        """
        hr_val, hr_conf, rr_val, rr_conf, spo2_val, spo2_conf = outputs
        hr_target, rr_target, spo2_target = targets

        # MSE losses for each metric
        loss_hr = self.mse_loss(hr_val, hr_target)
        loss_rr = self.mse_loss(rr_val, rr_target)
        loss_spo2 = self.mse_loss(spo2_val, spo2_target)

        # Confidence calibration losses
        hr_error = torch.abs(hr_val - hr_target)
        rr_error = torch.abs(rr_val - rr_target)
        spo2_error = torch.abs(spo2_val - spo2_target)

        cal_loss_hr = self.calibration_loss(hr_conf, hr_error)
        cal_loss_rr = self.calibration_loss(rr_conf, rr_error)
        cal_loss_spo2 = self.calibration_loss(spo2_conf, spo2_error)

        # Weighted total loss
        total_loss = (
            self.weights["hr"] * loss_hr +
            self.weights["rr"] * loss_rr +
            self.weights["spo2"] * loss_spo2 +
            self.weights["confidence"] * (cal_loss_hr + cal_loss_rr + cal_loss_spo2)
        )

        return total_loss
