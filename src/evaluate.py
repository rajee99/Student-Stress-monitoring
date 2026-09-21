"""
Evaluation Diagnostics, Classification Metrics, and Signal Attribution Engine.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


class AssessmentDiagnostics:
    """Computes test set classification reports, confusion matrices, and feature importances."""

    @staticmethod
    def calculate_normalized_confusion_matrix(
        ground_truth_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> np.ndarray:
        """Derive a percentage-normalized contingency matrix."""
        raw_contingency = confusion_matrix(ground_truth_labels, predicted_labels)
        normalized_matrix = raw_contingency.astype('float') / raw_contingency.sum(axis=1)[:, np.newaxis]
        return normalized_matrix

    @staticmethod
    def rank_feature_influences(
        fitted_estimator: Any,
        feature_identifiers: List[str]
    ) -> Optional[pd.Series]:
        """Extract and rank the relative importance scores of input survey features."""
        if hasattr(fitted_estimator, 'feature_importances_'):
            importance_coefficients = fitted_estimator.feature_importances_
            return pd.Series(importance_coefficients, index=feature_identifiers).sort_values(ascending=False)
        elif hasattr(fitted_estimator, 'coef_'):
            raw_coefficients = fitted_estimator.coef_
            mean_magnitudes = (
                np.abs(raw_coefficients).mean(axis=0)
                if raw_coefficients.ndim > 1 else np.abs(raw_coefficients)
            )
            return pd.Series(mean_magnitudes, index=feature_identifiers).sort_values(ascending=False)
        return None

    @staticmethod
    def pick_champion_model(
        benchmark_results_dict: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Identify the optimal classifier maximizing holdout accuracy and cross-validation stability.
        """
        champion_name = max(
            benchmark_results_dict.keys(),
            key=lambda name: (
                benchmark_results_dict[name]['holdout_accuracy'],
                benchmark_results_dict[name]['cross_val_mean'],
                -benchmark_results_dict[name]['cross_val_std']
            )
        )
        return champion_name, benchmark_results_dict[champion_name]

    @staticmethod
    def build_classification_text_report(
        actual_targets: np.ndarray,
        predicted_targets: np.ndarray,
        category_names: Optional[List[str]] = None
    ) -> str:
        """Format a detailed tabular precision/recall/F1 breakdown."""
        return classification_report(actual_targets, predicted_targets, target_names=category_names, digits=4)
