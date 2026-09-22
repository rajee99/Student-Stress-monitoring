"""
Streamlined Visualization Engine for the Paper's 5 Exact Models.
Generates:
1. benchmark_and_confusion.png (Paper Baseline vs Fine-Tuned Accuracy + Confusion Matrices)
2. feature_importance_ranking.png (Ranked explanatory stress determinants)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def configure_visual_theme() -> None:
    """Establish a modern aesthetic palette and clean typography."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        'figure.autolayout': False,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'figure.titleweight': 'bold',
        'font.family': 'sans-serif'
    })


class DiagnosticPlotter:
    """Renders and persists concise analytical dashboards for the 5 paper models."""

    def __init__(self, export_directory: Path):
        self.export_directory = export_directory
        self.export_directory.mkdir(parents=True, exist_ok=True)
        configure_visual_theme()

    def render_benchmark_and_confusion(
        self,
        baseline_results_d1: Dict[str, Dict[str, Any]],
        finetuned_results_d1: Dict[str, Dict[str, Any]],
        baseline_results_d2: Dict[str, Dict[str, Any]],
        finetuned_results_d2: Dict[str, Dict[str, Any]],
        confusion_matrix_d1: np.ndarray,
        labels_d1: List[str],
        confusion_matrix_d2: np.ndarray,
        labels_d2: List[str],
        output_filename: str = "benchmark_and_confusion.png"
    ) -> Path:
        """
        Unified 4-panel dashboard comparing Paper Baseline vs Fine-Tuned models and Confusion Matrices.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Helper to plot baseline vs fine-tuned
        model_keys = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM RBF', 'MLP Neural Net']

        # Panel (0, 0): Dataset 1 (Stress Level) Baseline vs Fine-Tuned
        ax_b1 = axes[0, 0]
        base_acc_d1 = [
            list(baseline_results_d1.values())[i]['holdout_accuracy'] * 100 for i in range(len(model_keys))
        ]
        fine_acc_d1 = [
            list(finetuned_results_d1.values())[i]['holdout_accuracy'] * 100 for i in range(len(model_keys))
        ]
        
        x_idx = np.arange(len(model_keys))
        bar_w = 0.35
        ax_b1.bar(x_idx - bar_w/2, base_acc_d1, bar_w, label='Paper Baseline Test Acc (%)', color='#e74c3c', alpha=0.85)
        ax_b1.bar(x_idx + bar_w/2, fine_acc_d1, bar_w, label='Fine-Tuned Test Acc (%)', color='#2ecc71', alpha=0.95)
        
        ax_b1.set_title("Dataset 1 (Stress Level): Paper Baseline vs. Fine-Tuned Accuracy")
        ax_b1.set_ylabel("Test Accuracy (%)")
        ax_b1.set_xticks(x_idx)
        ax_b1.set_xticklabels(model_keys, rotation=25, ha='right')
        ax_b1.set_ylim([80, 102])
        ax_b1.legend(loc='lower right')
        ax_b1.grid(axis='y', linestyle='--', alpha=0.7)

        # Panel (0, 1): Dataset 2 (Stress Type) Baseline vs Fine-Tuned
        ax_b2 = axes[0, 1]
        base_acc_d2 = [
            list(baseline_results_d2.values())[i]['holdout_accuracy'] * 100 for i in range(len(model_keys))
        ]
        fine_acc_d2 = [
            list(finetuned_results_d2.values())[i]['holdout_accuracy'] * 100 for i in range(len(model_keys))
        ]
        
        ax_b2.bar(x_idx - bar_w/2, base_acc_d2, bar_w, label='Paper Baseline Test Acc (%)', color='#e74c3c', alpha=0.85)
        ax_b2.bar(x_idx + bar_w/2, fine_acc_d2, bar_w, label='Fine-Tuned Test Acc (%)', color='#2ecc71', alpha=0.95)
        
        ax_b2.set_title("Dataset 2 (Stress Type): Paper Baseline vs. Fine-Tuned Accuracy")
        ax_b2.set_ylabel("Test Accuracy (%)")
        ax_b2.set_xticks(x_idx)
        ax_b2.set_xticklabels(model_keys, rotation=25, ha='right')
        ax_b2.set_ylim([85, 102])
        ax_b2.legend(loc='lower right')
        ax_b2.grid(axis='y', linestyle='--', alpha=0.7)

        # Panel (1, 0): Dataset 1 Normalized Confusion Matrix
        ax_cm1 = axes[1, 0]
        sns.heatmap(
            confusion_matrix_d1, annot=True, fmt=".1%", cmap="Blues", cbar=True,
            xticklabels=labels_d1, yticklabels=labels_d1, ax=ax_cm1
        )
        ax_cm1.set_title("Dataset 1: Confusion Matrix (Fine-Tuned Champion)")
        ax_cm1.set_xlabel("Predicted Label")
        ax_cm1.set_ylabel("Ground Truth")

        # Panel (1, 1): Dataset 2 Normalized Confusion Matrix
        ax_cm2 = axes[1, 1]
        clean_d2_labels = [
            (label.split(' - ')[0] if ' - ' in label else label) for label in labels_d2
        ]
        short_labels = [(l[:14] + '..') if len(l) > 16 else l for l in clean_d2_labels]
        sns.heatmap(
            confusion_matrix_d2, annot=True, fmt=".1%", cmap="Greens", cbar=True,
            xticklabels=short_labels, yticklabels=short_labels, ax=ax_cm2
        )
        ax_cm2.set_title("Dataset 2: Confusion Matrix (Fine-Tuned Champion)")
        ax_cm2.set_xlabel("Predicted Label")
        ax_cm2.set_ylabel("Ground Truth")
        ax_cm2.set_xticklabels(ax_cm2.get_xticklabels(), rotation=25, ha='right')

        plt.suptitle("Paper Models Comparative Analysis: Baseline vs Fine-Tuned", fontsize=15, fontweight='bold')
        plt.tight_layout()

        destination_path = self.export_directory / output_filename
        plt.savefig(destination_path, dpi=300, bbox_inches='tight')
        plt.close()
        return destination_path

    def render_feature_importance(
        self,
        importance_series_d1: Optional[pd.Series],
        importance_series_d2: Optional[pd.Series],
        output_filename: str = "feature_importance_ranking.png"
    ) -> Path:
        """
        Side-by-side ranked influence plots for explanatory factors.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for idx, (importance_data, panel_title, color_palette_name, target_axis) in enumerate([
            (importance_series_d1, "Dataset 1: Key Stress Level Determinants (Random Forest)", "viridis", axes[0]),
            (importance_series_d2, "Dataset 2: Key Stress Type Determinants (Gradient Boosting)", "plasma", axes[1])
        ]):
            if importance_data is not None:
                top_features = importance_data.head(10).sort_values(ascending=True)
                try:
                    palette_func = plt.colormaps[color_palette_name]
                except (AttributeError, KeyError):
                    palette_func = plt.get_cmap(color_palette_name)
                bar_colors = palette_func(np.linspace(0.4, 0.9, len(top_features)))

                trimmed_labels = [
                    (lbl[:32] + '...') if len(lbl) > 35 else lbl for lbl in top_features.index
                ]

                target_axis.barh(trimmed_labels, top_features.values, color=bar_colors, edgecolor='black', linewidth=0.7)
                target_axis.set_title(panel_title)
                target_axis.set_xlabel("Gini Feature Importance / Attribution Score")
                target_axis.grid(axis='x', linestyle='--', alpha=0.6)

        plt.suptitle("Top Predictive Indicators of Student Stress (Paper Models)", fontsize=15, fontweight='bold')
        plt.tight_layout()

        destination_path = self.export_directory / output_filename
        plt.savefig(destination_path, dpi=300, bbox_inches='tight')
        plt.close()
        return destination_path
