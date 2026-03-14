"""Confidence score calibration analysis."""

import numpy as np


class ConfidenceAnalysis:
    """Analyze confidence score calibration."""

    @staticmethod
    def calibration_error(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> dict:
        """
        Compute calibration error by confidence level.

        Returns dict with performance metrics for high/medium/low confidence predictions.
        """
        high_conf = confidence > 0.9
        med_conf = (confidence >= 0.5) & (confidence <= 0.9)
        low_conf = confidence < 0.5

        results = {}
        for partition, mask, label in [
            ("high", high_conf, "High (>0.9)"),
            ("medium", med_conf, "Medium (0.5-0.9)"),
            ("low", low_conf, "Low (<0.5)")
        ]:
            if mask.sum() > 0:
                mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
                results[partition] = {
                    "mae": mae,
                    "count": int(mask.sum()),
                    "label": label
                }

        return results

    @staticmethod
    def expected_calibration_error(confidence: np.ndarray, accuracy: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).

        accuracy: binary array (1 if prediction correct, 0 if incorrect)
        """
        bin_sums = np.zeros(n_bins)
        bin_true = np.zeros(n_bins)
        bin_total = np.zeros(n_bins)

        for i in range(len(confidence)):
            bin_idx = int(confidence[i] * n_bins) % n_bins
            bin_sums[bin_idx] += confidence[i]
            bin_true[bin_idx] += accuracy[i]
            bin_total[bin_idx] += 1

        ece_val = 0.0
        for i in range(n_bins):
            if bin_total[i] > 0:
                avg_conf = bin_sums[i] / bin_total[i]
                acc = bin_true[i] / bin_total[i]
                ece_val += (bin_total[i] / len(confidence)) * abs(avg_conf - acc)

        return ece_val
