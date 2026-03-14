"""Demographic bias analysis for rPPG model."""

import numpy as np
import pandas as pd


class DemographicAnalysis:
    """Analyze performance across demographic groups."""

    @staticmethod
    def analyze_by_skin_tone(y_true: np.ndarray, y_pred: np.ndarray, skin_tones: np.ndarray) -> pd.DataFrame:
        """
        Compute metrics by Fitzpatrick skin tone.

        Args:
            y_true: ground truth values
            y_pred: predicted values
            skin_tones: array with values like "I-II", "III-IV", "V-VI"

        Returns:
            DataFrame with metrics per skin tone
        """
        unique_tones = np.unique(skin_tones)
        results = []

        for tone in unique_tones:
            mask = skin_tones == tone
            if mask.sum() == 0:
                continue

            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            mae = np.mean(np.abs(y_pred_group - y_true_group))
            rmse = np.sqrt(np.mean((y_pred_group - y_true_group) ** 2))
            r = np.corrcoef(y_true_group, y_pred_group)[0, 1] if len(y_true_group) > 1 else 0

            results.append({
                "skin_tone": str(tone),
                "count": int(mask.sum()),
                "mae": mae,
                "rmse": rmse,
                "correlation": r
            })

        return pd.DataFrame(results)
