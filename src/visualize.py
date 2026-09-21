"""
Streamlined Visualization Engine.
Produces 2 focused, publication-grade diagnostic figures:
1. benchmark_and_confusion.png (Train, Validation, and Test accuracy comparison + Confusion Matrices)
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
    """Renders and persists concise analytical dashboards."""

    def __init__(self, export_directory: Path):
        self.export_directory = export_directory
        self.export_directory.mkdir(parents=True, exist_ok=True)
        configure_visual_theme()

    def render_benchmark_and_confusion(
        self,
        results_dataset_1: Dict[str, Dict[str, Any]],
        results_dataset_2: Dict[str, Dict[str, Any]],
        confusion_matrix_d1: np.ndarray,
        labels_d1: List[str],
        confusion_matrix_d2: np.ndarray,
        labels_d2: List[str],
        output_filename: str = "benchmark_and_confusion.png"
    ) -> Path:
        """
        Unified 4-panel dashboard combining Train/Val/Test performance benchmarks and confusion matrices.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Panel (0, 0): Dataset 1 Model Benchmark
        ax_b1 = axes[0, 0]
        models_1 = list(results_dataset_1.keys())[:8]  # Top 8 models for clean visibility
        train_1 = [results_dataset_1[m]['train_accuracy'] * 100 for m in models_1]
        val_1 = [results_dataset_1[m]['validation_accuracy'] * 100 for m in models_1]
        test_1 = [results_dataset_1[m]['holdout_accuracy'] * 100 for m in models_1]
        
        x_idx1 = np.arange(len(models_1))
        bar_w = 0.26
        ax_b1.bar(x_idx1 - bar_w, train_1, bar_w, label='Train Acc (%)', color='#95a5a6', alpha=0.85)
        ax_b1.bar(x_idx1, val_1, bar_w, label='Val Acc (5-Fold CV) (%)', color='#2980b9', alpha=0.9)
        ax_b1.bar(x_idx1 + bar_w, test_1, bar_w, label='Test Acc (%)', color='#27ae60', alpha=0.95)
        
        ax_b1.set_title("Dataset 1: Train vs. Validation vs. Test Accuracy")
        ax_b1.set_ylabel("Accuracy (%)")
        ax_b1.set_xticks(x_idx1)
        ax_b1.set_xticklabels(models_1, rotation=35, ha='right')
        ax_b1.set_ylim([75, 104])
        ax_b1.legend(loc='lower right')
        ax_b1.grid(axis='y', linestyle='--', alpha=0.7)

        # Panel (0, 1): Dataset 2 Model Benchmark
        ax_b2 = axes[0, 1]
        models_2 = list(results_dataset_2.keys())[:8]
        train_2 = [results_dataset_2[m]['train_accuracy'] * 100 for m in models_2]
        val_2 = [results_dataset_2[m]['validation_accuracy'] * 100 for m in models_2]
        test_2 = [results_dataset_2[m]['holdout_accuracy'] * 100 for m in models_2]
        
        x_idx2 = np.arange(len(models_2))
        ax_b2.bar(x_idx2 - bar_w, train_2, bar_w, label='Train Acc (%)', color='#95a5a6', alpha=0.85)
        ax_b2.bar(x_idx2, val_2, bar_w, label='Val Acc (5-Fold CV) (%)', color='#2980b9', alpha=0.9)
        ax_b2.bar(x_idx2 + bar_w, test_2, bar_w, label='Test Acc (%)', color='#27ae60', alpha=0.95)
        
        ax_b2.set_title("Dataset 2: Train vs. Validation vs. Test Accuracy")
        ax_b2.set_ylabel("Accuracy (%)")
        ax_b2.set_xticks(x_idx2)
        ax_b2.set_xticklabels(models_2, rotation=35, ha='right')
        ax_b2.set_ylim([85, 104])
        ax_b2.legend(loc='lower right')
        ax_b2.grid(axis='y', linestyle='--', alpha=0.7)

        # Panel (1, 0): Dataset 1 Normalized Confusion Matrix
        ax_cm1 = axes[1, 0]
        sns.heatmap(
            confusion_matrix_d1, annot=True, fmt=".1%", cmap="Blues", cbar=True,
            xticklabels=labels_d1, yticklabels=labels_d1, ax=ax_cm1
        )
        ax_cm1.set_title("Dataset 1: Confusion Matrix (Stress Level)")
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
        ax_cm2.set_title("Dataset 2: Confusion Matrix (Stress Type)")
        ax_cm2.set_xlabel("Predicted Label")
        ax_cm2.set_ylabel("Ground Truth")
        ax_cm2.set_xticklabels(ax_cm2.get_xticklabels(), rotation=25, ha='right')

        plt.suptitle("Model Evaluation: Train, Validation & Test Diagnostics", fontsize=15, fontweight='bold')
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
            (importance_series_d1, "Dataset 1: Key Stress Level Determinants", "viridis", axes[0]),
            (importance_series_d2, "Dataset 2: Key Stress Type Determinants", "plasma", axes[1])
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
                target_axis.set_xlabel("Attribution / Gini Importance")
                target_axis.grid(axis='x', linestyle='--', alpha=0.6)

        plt.suptitle("Top Predictive Indicators of Student Stress", fontsize=15, fontweight='bold')
        plt.tight_layout()

        destination_path = self.export_directory / output_filename
        plt.savefig(destination_path, dpi=300, bbox_inches='tight')
        plt.close()
        return destination_path
