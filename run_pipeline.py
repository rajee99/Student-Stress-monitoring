"""
Master Machine Learning Pipeline Runner.
Orchestrates data loading, transformation, adaptive modeling, evaluation, and visualization.
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
from src.models import ModelBenchmarkingService, find_optimal_feature_scaler
from src.evaluate import AssessmentDiagnostics
from src.visualize import DiagnosticPlotter


def print_performance_table(benchmark_dict: dict, dataset_name: str) -> None:
    """Print a clean comparative summary table for all models."""
    print(f"\n{'=' * 88}")
    print(f"MODEL BENCHMARK SUMMARY TABLE: {dataset_name.upper()}")
    print(f"{'=' * 88}")
    print(f"{'Model Name':<28} {'Train Acc':<12} {'Val Acc (5-Fold CV)':<22} {'Test Acc':<12} {'Test F1':<10}")
    print("-" * 88)
    for model_name, data in benchmark_dict.items():
        print(
            f"{model_name:<28} "
            f"{data['train_accuracy']*100:>8.2f}%   "
            f"{data['validation_accuracy']*100:>7.2f}% (+/- {data['validation_std']*200:>4.2f}%)   "
            f"{data['holdout_accuracy']*100:>7.2f}%   "
            f"{data['holdout_f1']:>8.4f}"
        )
    print("-" * 88)


def execute_stress_analysis_pipeline() -> None:
    """Execute the end-to-end model training, validation, and diagnostics pipeline."""
    print("=" * 88)
    print("STUDENT MENTAL HEALTH & STRESS PREDICTION PIPELINE")
    print("=" * 88)

    # Step 1: Initialize storage destinations
    initialize_output_directories()
    plot_generator = DiagnosticPlotter(export_directory=app_config.CHARTS_EXPORT_DIR)

    # Step 2: Read raw datasets and execute data health audit
    print("\n[Phase 1] Data Ingestion & Health Checks...")
    raw_level_records = DataAuditService.read_csv_records(app_config.STRESS_LEVEL_DATA_FILE)
    raw_type_records = DataAuditService.read_csv_records(app_config.STRESS_TYPE_DATA_FILE)

    level_audit = DataAuditService.audit_tabular_health(raw_level_records, "Stress Level Records")
    type_audit = DataAuditService.audit_tabular_health(raw_type_records, "Stress Type Records")
    print(f" -> Level Dataset: {level_audit['sample_count']} rows, {level_audit['feature_count']} cols.")
    print(f" -> Type Dataset:  {type_audit['sample_count']} rows, {type_audit['feature_count']} cols.")

    # Step 3: Feature Engineering & Domain Index Generation
    print("\n[Phase 2] Domain Feature Engineering...")
    features_d1, target_d1 = StudentFeatureTransformer.transform_stress_level_dataset(raw_level_records)
    features_d2, target_d2, encoder_d2 = StudentFeatureTransformer.transform_stress_type_dataset(raw_type_records)
    print(f" -> Dataset 1 Processed: {features_d1.shape[1]} engineered features across {features_d1.shape[0]} rows.")
    print(f" -> Dataset 2 Processed: {features_d2.shape[1]} engineered features across {features_d2.shape[0]} rows.")

    # Step 4: Stratified Partitioning & Adaptive Feature Scaling
    print("\n[Phase 3] Stratified Partitioning & Adaptive Scaling...")
    
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
    print(f" -> Dataset 1 Selected Scaler: {scaler_name_1}")

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
    print(f" -> Dataset 2 Selected Scaler: {scaler_name_2}")

    # Step 5: Multi-Model Benchmark & Cross-Validation
    print("\n[Phase 4] Cross-Validation Benchmarking (18+ Models)...")
    benchmark_d1 = ModelBenchmarkingService.run_portfolio_benchmark(
        scaled_train_x1, train_y1.values, scaled_test_x1, test_y1.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED,
        k_folds=app_config.STRATIFIED_K_FOLDS
    )
    benchmark_d2 = ModelBenchmarkingService.run_portfolio_benchmark(
        scaled_train_x2, train_y2.values, scaled_test_x2, test_y2.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED,
        k_folds=app_config.STRATIFIED_K_FOLDS
    )

    # Print clean benchmark tables
    print_performance_table(benchmark_d1, "Dataset 1 - Stress Level")
    print_performance_table(benchmark_d2, "Dataset 2 - Stress Type")

    # Step 6: Identify Champions and Diagnostic Evaluation
    print("\n[Phase 5] Champion Model Selection & Metrics Report...")
    champion_name_1, champion_data_1 = AssessmentDiagnostics.pick_champion_model(benchmark_d1)
    champion_name_2, champion_data_2 = AssessmentDiagnostics.pick_champion_model(benchmark_d2)

    print("=" * 88)
    print(f"[CHAMPION MODEL] DATASET 1: {champion_name_1}")
    print(f"   - Train Accuracy:      {champion_data_1['train_accuracy']*100:.2f}%")
    print(f"   - Validation Accuracy: {champion_data_1['validation_accuracy']*100:.2f}% (+/- {champion_data_1['validation_std']*200:.2f}%)")
    print(f"   - Test Accuracy:       {champion_data_1['holdout_accuracy']*100:.2f}%")
    print(f"   - Test F1 Score:       {champion_data_1['holdout_f1']:.4f}")
    print("\nDetailed Classification Breakdown:")
    print(AssessmentDiagnostics.build_classification_text_report(
        test_y1.values, champion_data_1['test_predictions'], category_names=['Level 0', 'Level 1', 'Level 2']
    ))

    print("=" * 88)
    print(f"[CHAMPION MODEL] DATASET 2: {champion_name_2}")
    print(f"   - Train Accuracy:      {champion_data_2['train_accuracy']*100:.2f}%")
    print(f"   - Validation Accuracy: {champion_data_2['validation_accuracy']*100:.2f}% (+/- {champion_data_2['validation_std']*200:.2f}%)")
    print(f"   - Test Accuracy:       {champion_data_2['holdout_accuracy']*100:.2f}%")
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

    # Step 7: Persist Models and Generate Visualizations
    print("\n[Phase 6] Exporting Artifacts & Persisting Figures...")
    
    # Save 2 concise figures
    fig_bench = plot_generator.render_benchmark_and_confusion(
        benchmark_d1, benchmark_d2,
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

    print("\n" + "=" * 88)
    print("PIPELINE EXECUTION COMPLETE & VERIFIED!")
    print("=" * 88)


if __name__ == '__main__':
    execute_stress_analysis_pipeline()
