"""
Visualization Engine for Stress Classification Pipeline.
Generates:
1. unseen_data_performance.png (Unseen Test Accuracy comparisons + Unseen Confusion Matrices)
2. benchmark_and_confusion.png (Train vs Validation vs Unseen Test breakdown)
3. feature_importance_ranking.png (Ranked explanatory stress determinants)
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
    """Renders and persists analytical dashboards for the classification models."""

    def __init__(self, export_directory: Path):
        self.export_directory = export_directory
        self.export_directory.mkdir(parents=True, exist_ok=True)
        configure_visual_theme()

    def render_unseen_performance_dashboard(
        self,
        unseen_metrics_d1: Dict[str, Dict[str, Any]],
        unseen_metrics_d2: Dict[str, Dict[str, Any]],
        unseen_cm_d1: np.ndarray,
        labels_d1: List[str],
        unseen_cm_d2: np.ndarray,
        labels_d2: List[str],
        output_filename: str = "unseen_data_performance.png"
    ) -> Path:
        """
        Dedicated Dashboard Visualizing Out-of-Sample Performance on 100% Unseen Test Data.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # -------------------------------------------------------------
        # Panel (0, 0): Dataset 1 Unseen Accuracy Bar Chart
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        models_1 = list(unseen_metrics_d1.keys())
        accs_1 = [unseen_metrics_d1[m]['holdout_accuracy'] * 100 for m in models_1]
        correct_1 = [unseen_metrics_d1[m].get('correct_count', 0) for m in models_1]
        total_1 = unseen_metrics_d1[models_1[0]].get('total_count', 220)

        bars1 = ax1.bar(
            models_1, accs_1, color=['#27ae60' if a == max(accs_1) else '#3498db' for a in accs_1],
            edgecolor='black', linewidth=1.1, alpha=0.9
        )
        ax1.set_title(f"Dataset 1 (Stress Level): Accuracy on Unseen Data (N={total_1})")
        ax1.set_ylabel("Unseen Test Accuracy (%)")
        ax1.set_ylim([75, 100])
        ax1.set_xticklabels(models_1, rotation=25, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        for bar, corr, acc in zip(bars1, correct_1, accs_1):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{acc:.2f}%\n({corr}/{total_1})",
                ha='center', va='bottom', fontweight='bold', fontsize=9
            )

        # -------------------------------------------------------------
        # Panel (0, 1): Dataset 2 Unseen Accuracy Bar Chart
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        models_2 = list(unseen_metrics_d2.keys())
        accs_2 = [unseen_metrics_d2[m]['holdout_accuracy'] * 100 for m in models_2]
        correct_2 = [unseen_metrics_d2[m].get('correct_count', 0) for m in models_2]
        total_2 = unseen_metrics_d2[models_2[0]].get('total_count', 164)

        bars2 = ax2.bar(
            models_2, accs_2, color=['#27ae60' if a == max(accs_2) else '#3498db' for a in accs_2],
            edgecolor='black', linewidth=1.1, alpha=0.9
        )
        ax2.set_title(f"Dataset 2 (Stress Type): Accuracy on Unseen Data (N={total_2})")
        ax2.set_ylabel("Unseen Test Accuracy (%)")
        ax2.set_ylim([85, 102])
        ax2.set_xticklabels(models_2, rotation=25, ha='right')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        for bar, corr, acc in zip(bars2, correct_2, accs_2):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc:.2f}%\n({corr}/{total_2})",
                ha='center', va='bottom', fontweight='bold', fontsize=9
            )

        # -------------------------------------------------------------
        # Panel (1, 0): Dataset 1 Unseen Normalized Confusion Matrix
        # -------------------------------------------------------------
        ax3 = axes[1, 0]
        sns.heatmap(
            unseen_cm_d1, annot=True, fmt=".1%", cmap="Blues", cbar=True,
            xticklabels=labels_d1, yticklabels=labels_d1, ax=ax3, linewidths=0.5
        )
        ax3.set_title("Dataset 1: Unseen Data Confusion Matrix (Top Model)")
        ax3.set_xlabel("Predicted Class")
        ax3.set_ylabel("True Ground Truth")

        # -------------------------------------------------------------
        # Panel (1, 1): Dataset 2 Unseen Normalized Confusion Matrix
        # -------------------------------------------------------------
        ax4 = axes[1, 1]
        clean_d2_labels = [
            (label.split(' - ')[0] if ' - ' in label else label) for label in labels_d2
        ]
        short_labels = [(l[:14] + '..') if len(l) > 16 else l for l in clean_d2_labels]
        sns.heatmap(
            unseen_cm_d2, annot=True, fmt=".1%", cmap="Greens", cbar=True,
            xticklabels=short_labels, yticklabels=short_labels, ax=ax4, linewidths=0.5
        )
        ax4.set_title("Dataset 2: Unseen Data Confusion Matrix (Top Model)")
        ax4.set_xlabel("Predicted Class")
        ax4.set_ylabel("True Ground Truth")
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=25, ha='right')

        plt.suptitle("Generalization Diagnostics on 100% Unseen Held-Out Test Data", fontsize=15, fontweight='bold')
        plt.tight_layout()

        destination_path = self.export_directory / output_filename
        plt.savefig(destination_path, dpi=300, bbox_inches='tight')
        plt.close()
        return destination_path

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
        Unified 4-panel dashboard comparing model benchmark results and confusion matrices.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Panel (0, 0): Dataset 1 (Stress Level)
        ax_b1 = axes[0, 0]
        models_1 = list(finetuned_results_d1.keys())
        fine_acc_d1 = [finetuned_results_d1[m]['holdout_accuracy'] * 100 for m in models_1]
        val_acc_d1 = [finetuned_results_d1[m]['validation_accuracy'] * 100 for m in models_1]
        
        x_idx1 = np.arange(len(models_1))
        bar_w = 0.35
        ax_b1.bar(x_idx1 - bar_w/2, val_acc_d1, bar_w, label='5-Fold CV Acc (%)', color='#3498db', alpha=0.85)
        ax_b1.bar(x_idx1 + bar_w/2, fine_acc_d1, bar_w, label='Unseen Test Acc (%)', color='#2ecc71', alpha=0.95)
        
        ax_b1.set_title("Dataset 1 (Stress Level): Model Generalization Benchmark")
        ax_b1.set_ylabel("Accuracy (%)")
        ax_b1.set_xticks(x_idx1)
        short_names_1 = [m.replace(' (Novel Custom 1)', '').replace(' (Novel Custom 2)', '').replace(' (Kernel Benchmark)', '') for m in models_1]
        ax_b1.set_xticklabels(short_names_1, rotation=20, ha='right')
        ax_b1.set_ylim([75, 102])
        ax_b1.legend(loc='lower right')
        ax_b1.grid(axis='y', linestyle='--', alpha=0.7)

        # Panel (0, 1): Dataset 2 (Stress Type)
        ax_b2 = axes[0, 1]
        models_2 = list(finetuned_results_d2.keys())
        fine_acc_d2 = [finetuned_results_d2[m]['holdout_accuracy'] * 100 for m in models_2]
        val_acc_d2 = [finetuned_results_d2[m]['validation_accuracy'] * 100 for m in models_2]
        
        x_idx2 = np.arange(len(models_2))
        ax_b2.bar(x_idx2 - bar_w/2, val_acc_d2, bar_w, label='5-Fold CV Acc (%)', color='#3498db', alpha=0.85)
        ax_b2.bar(x_idx2 + bar_w/2, fine_acc_d2, bar_w, label='Unseen Test Acc (%)', color='#2ecc71', alpha=0.95)
        
        ax_b2.set_title("Dataset 2 (Stress Type): Model Generalization Benchmark")
        ax_b2.set_ylabel("Accuracy (%)")
        ax_b2.set_xticks(x_idx2)
        short_names_2 = [m.replace(' (Novel Custom 1)', '').replace(' (Novel Custom 2)', '').replace(' (Kernel Benchmark)', '') for m in models_2]
        ax_b2.set_xticklabels(short_names_2, rotation=20, ha='right')
        ax_b2.set_ylim([80, 102])
        ax_b2.legend(loc='lower right')
        ax_b2.grid(axis='y', linestyle='--', alpha=0.7)

        # Panel (1, 0): Dataset 1 Confusion Matrix
        ax_cm1 = axes[1, 0]
        sns.heatmap(
            confusion_matrix_d1, annot=True, fmt=".1%", cmap="Blues", cbar=True,
            xticklabels=labels_d1, yticklabels=labels_d1, ax=ax_cm1
        )
        ax_cm1.set_title("Dataset 1 (Stress Level): Unseen Normalized Confusion Matrix")
        ax_cm1.set_xlabel("Predicted Label")
        ax_cm1.set_ylabel("Ground Truth")

        # Panel (1, 1): Dataset 2 Confusion Matrix
        ax_cm2 = axes[1, 1]
        clean_d2_labels = [
            (label.split(' - ')[0] if ' - ' in label else label) for label in labels_d2
        ]
        short_labels = [(l[:14] + '..') if len(l) > 16 else l for l in clean_d2_labels]
        sns.heatmap(
            confusion_matrix_d2, annot=True, fmt=".1%", cmap="Greens", cbar=True,
            xticklabels=short_labels, yticklabels=short_labels, ax=ax_cm2
        )
        ax_cm2.set_title("Dataset 2 (Stress Type): Unseen Normalized Confusion Matrix")
        ax_cm2.set_xlabel("Predicted Label")
        ax_cm2.set_ylabel("Ground Truth")
        ax_cm2.set_xticklabels(ax_cm2.get_xticklabels(), rotation=20, ha='right')

        plt.suptitle("Non-Tree & Non-Regression Classifiers: Generalization Analysis", fontsize=15, fontweight='bold')
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
            (importance_series_d1, "Dataset 1: Key Stress Level Determinants (Top Model)", "viridis", axes[0]),
            (importance_series_d2, "Dataset 2: Key Stress Type Determinants (Top Model)", "plasma", axes[1])
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
