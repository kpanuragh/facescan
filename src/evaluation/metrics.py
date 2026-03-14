"""Clinical evaluation metrics for rPPG model."""

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


class EvaluationMetrics:
    """Compute clinical evaluation metrics."""

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """Pearson correlation coefficient and p-value."""
        r, p_value = stats.pearsonr(y_true, y_pred)
        return r, p_value

    @staticmethod
    def bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute Bland-Altman agreement statistics."""
        mean_val = (y_true + y_pred) / 2
        diff = y_pred - y_true

        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        return {
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "upper_loa": upper_loa,
            "lower_loa": lower_loa,
            "mean": mean_val,
            "diff": diff
        }
