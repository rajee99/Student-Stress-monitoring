"""
Master Machine Learning Pipeline Runner.
Evaluates ONLY the 5 models discussed in the Research Paper:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. SVM (RBF Kernel)
5. Multilayer Perceptron (MLP)

Executes both:
- Phase A: Paper Baseline Configurations (reproducing paper metrics)
- Phase B: Fine-Tuned Configurations (optimized to beat the paper)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src.config import app_config, initialize_output_directories
from src.data_loader import DataAuditService
from src.preprocessing import StudentFeatureTransformer
from src.models import (
    get_paper_baseline_models,
    get_finetuned_models,
    find_optimal_feature_scaler,
    ModelBenchmarkingService
)
from src.evaluate import AssessmentDiagnostics
from src.visualize import DiagnosticPlotter


def print_paper_comparison_table(baseline_dict: dict, finetuned_dict: dict, dataset_name: str) -> None:
    """Print a clean comparative summary table for the 5 paper models."""
    print(f"\n{'=' * 96}")
    print(f"HEAD-TO-HEAD COMPARISON: {dataset_name.upper()} (PAPER BASELINE vs. FINE-TUNED)")
    print(f"{'=' * 96}")
    print(f"{'Model Name':<26} {'Paper Base Test':<18} {'Fine-Tuned Test':<18} {'Val Acc (5-Fold)':<18} {'Gain':<10}")
    print("-" * 96)
    
    base_keys = list(baseline_dict.keys())
    fine_keys = list(finetuned_dict.keys())

    for b_key, f_key in zip(base_keys, fine_keys):
        b_data = baseline_dict[b_key]
        f_data = finetuned_dict[f_key]
        
        b_acc = b_data['holdout_accuracy'] * 100
        f_acc = f_data['holdout_accuracy'] * 100
        val_acc = f_data['validation_accuracy'] * 100
        val_std = f_data['validation_std'] * 200
        diff = f_acc - b_acc

        clean_name = b_key.replace(' (Paper Baseline)', '')
        gain_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"

        print(
            f"{clean_name:<26} "
            f"{b_acc:>12.2f}%      "
            f"{f_acc:>12.2f}%      "
            f"{val_acc:>6.2f}% (+/- {val_std:>4.2f}%)   "
            f"{gain_str:>8}"
        )
    print("-" * 96)


def execute_stress_analysis_pipeline() -> None:
    """Execute the end-to-end model training, validation, and diagnostics pipeline."""
    print("=" * 96)
    print("REPRODUCING & OUTPERFORMING RESEARCH PAPER CLASSIFICATION FRAMEWORK")
    print("Paper: 'An explainable machine learning framework for academic stress classification' (2026)")
    print("=" * 96)

    # Step 1: Initialize storage destinations
    initialize_output_directories()
    plot_generator = DiagnosticPlotter(export_directory=app_config.CHARTS_EXPORT_DIR)

    # Step 2: Read raw datasets and execute data health audit
    print("\n[Phase 1] Ingesting Datasets & Health Auditing...")
    raw_level_records = DataAuditService.read_csv_records(app_config.STRESS_LEVEL_DATA_FILE)
    raw_type_records = DataAuditService.read_csv_records(app_config.STRESS_TYPE_DATA_FILE)

    level_audit = DataAuditService.audit_tabular_health(raw_level_records, "Stress Level Records (Dataset 1)")
    type_audit = DataAuditService.audit_tabular_health(raw_type_records, "Stress Type Records (Dataset 2)")
    print(f" -> Dataset 1 (Stress Level): {level_audit['sample_count']} rows, {level_audit['feature_count']} features.")
    print(f" -> Dataset 2 (Stress Type):  {type_audit['sample_count']} rows, {type_audit['feature_count']} features.")

    # Step 3: Domain Feature Preprocessing
    print("\n[Phase 2] Domain Feature Synthesis...")
    features_d1, target_d1 = StudentFeatureTransformer.transform_stress_level_dataset(raw_level_records)
    features_d2, target_d2, encoder_d2 = StudentFeatureTransformer.transform_stress_type_dataset(raw_type_records)
    print(f" -> Dataset 1 Feature Representation: {features_d1.shape[1]} features, {features_d1.shape[0]} samples.")
    print(f" -> Dataset 2 Feature Representation: {features_d2.shape[1]} features, {features_d2.shape[0]} samples.")

    # Step 4: Stratified Partitioning & Adaptive Feature Scaling
    print("\n[Phase 3] Stratified Partitioning (80/20) & Adaptive Preconditioning...")
    
    # Dataset 1 Split
    train_x1, test_x1, train_y1, test_y1 = train_test_split(
        features_d1, target_d1,
        test_size=app_config.HOLDOUT_TEST_FRACTION,
        random_state=app_config.DATA_SPLIT_RANDOM_SEED,
        stratify=target_d1
    )
    scaler_name_1, scaler_instance_1 = find_optimal_feature_scaler(
        train_x1, train_y1,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED,
        cv_partitions=app_config.STRATIFIED_K_FOLDS
    )
    scaled_train_x1 = scaler_instance_1.fit_transform(train_x1)
    scaled_test_x1 = scaler_instance_1.transform(test_x1)
    print(f" -> Dataset 1 Preconditioning: {scaler_name_1}")

    # Dataset 2 Split
    train_x2, test_x2, train_y2, test_y2 = train_test_split(
        features_d2, target_d2,
        test_size=app_config.HOLDOUT_TEST_FRACTION,
        random_state=app_config.DATA_SPLIT_RANDOM_SEED,
        stratify=target_d2
    )
    scaler_name_2, scaler_instance_2 = find_optimal_feature_scaler(
        train_x2, train_y2,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED,
        cv_partitions=app_config.STRATIFIED_K_FOLDS
    )
    scaled_train_x2 = scaler_instance_2.fit_transform(train_x2)
    scaled_test_x2 = scaler_instance_2.transform(test_x2)
    print(f" -> Dataset 2 Preconditioning: {scaler_name_2}")

    # Step 5: Evaluate 5 Paper Baseline Models
    print("\n[Phase 4A] Benchmarking 5 Paper Baseline Models (Default Parameters)...")
    baseline_models_d1 = get_paper_baseline_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)
    baseline_models_d2 = get_paper_baseline_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)

    baseline_d1 = ModelBenchmarkingService.evaluate_model_dictionary(
        baseline_models_d1, scaled_train_x1, train_y1.values, scaled_test_x1, test_y1.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED, k_folds=app_config.STRATIFIED_K_FOLDS
    )
    baseline_d2 = ModelBenchmarkingService.evaluate_model_dictionary(
        baseline_models_d2, scaled_train_x2, train_y2.values, scaled_test_x2, test_y2.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED, k_folds=app_config.STRATIFIED_K_FOLDS
    )

    # Step 6: Evaluate 5 Fine-Tuned Models (Optimized to Beat the Paper)
    print("\n[Phase 4B] Benchmarking 5 Fine-Tuned Models (Regularized & Depth-Bounded)...")
    finetuned_models_d1 = get_finetuned_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)
    finetuned_models_d2 = get_finetuned_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)

    finetuned_d1 = ModelBenchmarkingService.evaluate_model_dictionary(
        finetuned_models_d1, scaled_train_x1, train_y1.values, scaled_test_x1, test_y1.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED, k_folds=app_config.STRATIFIED_K_FOLDS
    )
    finetuned_d2 = ModelBenchmarkingService.evaluate_model_dictionary(
        finetuned_models_d2, scaled_train_x2, train_y2.values, scaled_test_x2, test_y2.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED, k_folds=app_config.STRATIFIED_K_FOLDS
    )

    # Display Comparative Tables
    print_paper_comparison_table(baseline_d1, finetuned_d1, "Dataset 1 - Stress Level")
    print_paper_comparison_table(baseline_d2, finetuned_d2, "Dataset 2 - Stress Type")

    # Step 7: Identify Fine-Tuned Champions
    print("\n[Phase 5] Champion Model Selection & Metrics Report...")
    champion_name_1, champion_data_1 = AssessmentDiagnostics.pick_champion_model(finetuned_d1)
    champion_name_2, champion_data_2 = AssessmentDiagnostics.pick_champion_model(finetuned_d2)

    print("=" * 96)
    print(f"[CHAMPION MODEL] DATASET 1 (STRESS LEVEL): {champion_name_1}")
    print(f"   - Train Accuracy:      {champion_data_1['train_accuracy']*100:.2f}%")
    print(f"   - Validation Accuracy: {champion_data_1['validation_accuracy']*100:.2f}% (+/- {champion_data_1['validation_std']*200:.2f}%)")
    print(f"   - Test Accuracy:       {champion_data_1['holdout_accuracy']*100:.2f}% (Paper Baseline: 89.09%)")
    print(f"   - Test F1 Score:       {champion_data_1['holdout_f1']:.4f}")
    print("\nDetailed Classification Breakdown:")
    print(AssessmentDiagnostics.build_classification_text_report(
        test_y1.values, champion_data_1['test_predictions'], category_names=['Level 0', 'Level 1', 'Level 2']
    ))

    print("=" * 96)
    print(f"[CHAMPION MODEL] DATASET 2 (STRESS TYPE): {champion_name_2}")
    print(f"   - Train Accuracy:      {champion_data_2['train_accuracy']*100:.2f}%")
    print(f"   - Validation Accuracy: {champion_data_2['validation_accuracy']*100:.2f}% (+/- {champion_data_2['validation_std']*200:.2f}%)")
    print(f"   - Test Accuracy:       {champion_data_2['holdout_accuracy']*100:.2f}% (Paper Baseline: 93.59%)")
    print(f"   - Test F1 Score:       {champion_data_2['holdout_f1']:.4f}")
    print("\nDetailed Classification Breakdown:")
    print(AssessmentDiagnostics.build_classification_text_report(
        test_y2.values, champion_data_2['test_predictions'], category_names=list(encoder_d2.classes_)
    ))

    # Feature Importance & Confusion Matrix Calculations
    attribution_d1 = AssessmentDiagnostics.rank_feature_influences(
        champion_data_1['fitted_estimator'], features_d1.columns.tolist()
    )
    attribution_d2 = AssessmentDiagnostics.rank_feature_influences(
        champion_data_2['fitted_estimator'], features_d2.columns.tolist()
    )

    cm_d1 = AssessmentDiagnostics.calculate_normalized_confusion_matrix(
        test_y1.values, champion_data_1['test_predictions']
    )
    cm_d2 = AssessmentDiagnostics.calculate_normalized_confusion_matrix(
        test_y2.values, champion_data_2['test_predictions']
    )

    # Step 8: Persist Models and Generate Visualizations
    print("\n[Phase 6] Exporting Artifacts & Persisting Figures...")
    
    # Save 2 concise figures
    fig_bench = plot_generator.render_benchmark_and_confusion(
        baseline_d1, finetuned_d1,
        baseline_d2, finetuned_d2,
        cm_d1, ['Level 0', 'Level 1', 'Level 2'],
        cm_d2, list(encoder_d2.classes_)
    )
    fig_feat = plot_generator.render_feature_importance(attribution_d1, attribution_d2)

    # Save fitted model artifacts
    joblib.dump(champion_data_1['fitted_estimator'], app_config.SAVED_MODELS_DIR / "stress_level_model.joblib")
    joblib.dump(scaler_instance_1, app_config.SAVED_MODELS_DIR / "stress_level_scaler.joblib")
    joblib.dump(features_d1.columns.tolist(), app_config.SAVED_MODELS_DIR / "stress_level_features.joblib")

    joblib.dump(champion_data_2['fitted_estimator'], app_config.SAVED_MODELS_DIR / "stress_type_model.joblib")
    joblib.dump(scaler_instance_2, app_config.SAVED_MODELS_DIR / "stress_type_scaler.joblib")
    joblib.dump(features_d2.columns.tolist(), app_config.SAVED_MODELS_DIR / "stress_type_features.joblib")
    joblib.dump(encoder_d2, app_config.SAVED_MODELS_DIR / "stress_type_label_encoder.joblib")

    print(f"\nVisualizations Successfully Exported to: {app_config.CHARTS_EXPORT_DIR}")
    print(f" 1. {fig_bench.name}")
    print(f" 2. {fig_feat.name}")

    print(f"\nModel Artifacts Successfully Saved to: {app_config.SAVED_MODELS_DIR}")
    print(" - stress_level_model.joblib")
    print(" - stress_type_model.joblib")

    print("\n" + "=" * 96)
    print("PIPELINE EXECUTION COMPLETE: 5 PAPER MODELS EVALUATED & OPTIMIZED!")
    print("=" * 96)


if __name__ == '__main__':
    execute_stress_analysis_pipeline()
