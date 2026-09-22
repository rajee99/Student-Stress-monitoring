"""
Master Machine Learning Pipeline Runner (Novel Non-Tree, Non-Regression Branch).

Evaluates:
1. Standard Non-Tree Baselines: SVM RBF, MLP Neural Net
2. Custom Novel Classifiers:
   - KernelManifoldAttentionClassifier (KMAC - Metric Learning & Prototype Attention)
   - ResidualGatedFeatureClassifier (RGFN - Tabular Squeeze-and-Excitation & Hyperspherical Cosine Head)

Strict Constraints Adhered To:
- ZERO Tree Models (No Random Forest, Decision Tree, Gradient Boosting, XGBoost, etc.)
- ZERO Regression Models (No Logistic Regression, Linear Regression, Ridge, etc.)

Features:
- Stratified 5-Fold Cross-Validation on Training Set (80%)
- Full Evaluation & Accuracy Measurement on 100% UNSEEN Test Data (20%)
- Automated Visualizations of Unseen Data Performance & Feature Importance
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
from sklearn.metrics import accuracy_score, f1_score

from src.config import app_config, initialize_output_directories
from src.data_loader import DataAuditService
from src.preprocessing import StudentFeatureTransformer
from src.models import (
    get_non_tree_baseline_models,
    get_novel_non_tree_models,
    find_optimal_feature_scaler,
    ModelBenchmarkingService
)
from src.evaluate import AssessmentDiagnostics
from src.visualize import DiagnosticPlotter


def print_unseen_performance_table(unseen_dict: dict, dataset_name: str) -> None:
    """Print a clean summary table of model accuracy on 100% unseen test data."""
    print(f"\n{'=' * 96}")
    print(f"UNSEEN TEST DATA ACCURACY BREAKDOWN: {dataset_name.upper()}")
    print(f"{'=' * 96}")
    print(f"{'Model Name':<42} {'Unseen Accuracy':<18} {'Correct / Total':<18} {'Val Acc (5-Fold)':<18} {'Weighted F1':<12}")
    print("-" * 108)
    
    for model_name, data in unseen_dict.items():
        acc = data['holdout_accuracy'] * 100
        correct = data.get('correct_count', 0)
        total = data.get('total_count', 0)
        val_acc = data['validation_accuracy'] * 100
        val_std = data['validation_std'] * 200
        f1 = data['holdout_f1']

        print(
            f"{model_name:<42} "
            f"{acc:>12.2f}%      "
            f"{correct}/{total:<13} "
            f"{val_acc:>6.2f}% (+/- {val_std:>4.2f}%)   "
            f"{f1:>10.4f}"
        )
    print("-" * 108)


def execute_stress_analysis_pipeline() -> None:
    """Execute the end-to-end model training, validation, and diagnostics pipeline."""
    print("=" * 108)
    print("STUDENT STRESS PREDICTION: NOVEL NON-TREE CLASSIFIERS & UNSEEN DATA BENCHMARK")
    print("=" * 108)

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

    # Step 4: Stratified Partitioning (80% Train / 20% Held-Out Unseen Test)
    print("\n[Phase 3] Stratified Partitioning (80% Train / 20% Unseen Test)...")
    
    # Dataset 1 Split
    train_x1, unseen_x1, train_y1, unseen_y1 = train_test_split(
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
    scaled_unseen_x1 = scaler_instance_1.transform(unseen_x1)
    print(f" -> Dataset 1 Preconditioning: {scaler_name_1} | Train={len(train_x1)}, Unseen Test={len(unseen_x1)}")

    # Dataset 2 Split
    train_x2, unseen_x2, train_y2, unseen_y2 = train_test_split(
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
    scaled_unseen_x2 = scaler_instance_2.transform(unseen_x2)
    print(f" -> Dataset 2 Preconditioning: {scaler_name_2} | Train={len(train_x2)}, Unseen Test={len(unseen_x2)}")

    # Step 5: Evaluate Non-Tree Baseline Models
    print("\n[Phase 4A] Benchmarking Non-Tree Baselines...")
    baseline_models_d1 = get_non_tree_baseline_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)
    baseline_models_d2 = get_non_tree_baseline_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)

    baseline_d1 = ModelBenchmarkingService.evaluate_model_dictionary(
        baseline_models_d1, scaled_train_x1, train_y1.values, scaled_unseen_x1, unseen_y1.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED, k_folds=app_config.STRATIFIED_K_FOLDS
    )
    baseline_d2 = ModelBenchmarkingService.evaluate_model_dictionary(
        baseline_models_d2, scaled_train_x2, train_y2.values, scaled_unseen_x2, unseen_y2.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED, k_folds=app_config.STRATIFIED_K_FOLDS
    )

    # Step 6: Train & Evaluate Novel Custom Models on Unseen Data
    print("\n[Phase 4B] Training & Evaluating Novel Non-Tree Classifiers on 100% Unseen Test Data...")
    novel_models_d1 = get_novel_non_tree_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)
    novel_models_d2 = get_novel_non_tree_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)

    novel_d1 = ModelBenchmarkingService.evaluate_model_dictionary(
        novel_models_d1, scaled_train_x1, train_y1.values, scaled_unseen_x1, unseen_y1.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED, k_folds=app_config.STRATIFIED_K_FOLDS
    )
    novel_d2 = ModelBenchmarkingService.evaluate_model_dictionary(
        novel_models_d2, scaled_train_x2, train_y2.values, scaled_unseen_x2, unseen_y2.values,
        random_seed=app_config.DATA_SPLIT_RANDOM_SEED, k_folds=app_config.STRATIFIED_K_FOLDS
    )

    # Attach sample counts for reporting
    for k, v in novel_d1.items():
        v['total_count'] = len(unseen_y1)
        v['correct_count'] = int(np.sum(v['test_predictions'] == unseen_y1.values))

    for k, v in novel_d2.items():
        v['total_count'] = len(unseen_y2)
        v['correct_count'] = int(np.sum(v['test_predictions'] == unseen_y2.values))

    # Display Unseen Performance Tables
    print_unseen_performance_table(novel_d1, "Dataset 1 - Stress Level (Unseen Data)")
    print_unseen_performance_table(novel_d2, "Dataset 2 - Stress Type (Unseen Data)")

    # Step 7: Identify Unseen Data Champions & Detailed Classification Reports
    print("\n[Phase 5] Unseen Data Champion Models & Metrics Breakdown...")
    champion_name_1, champion_data_1 = AssessmentDiagnostics.pick_champion_model(novel_d1)
    champion_name_2, champion_data_2 = AssessmentDiagnostics.pick_champion_model(novel_d2)

    print("=" * 108)
    print(f"[CHAMPION ON UNSEEN DATA] DATASET 1 (STRESS LEVEL): {champion_name_1}")
    print(f"   - Unseen Test Accuracy: {champion_data_1['holdout_accuracy']*100:.2f}% ({champion_data_1['correct_count']}/{champion_data_1['total_count']} Correct)")
    print(f"   - Validation Accuracy:  {champion_data_1['validation_accuracy']*100:.2f}% (+/- {champion_data_1['validation_std']*200:.2f}%)")
    print(f"   - Weighted F1 Score:    {champion_data_1['holdout_f1']:.4f}")
    print("\nDetailed Classification Breakdown on Unseen Data:")
    print(AssessmentDiagnostics.build_classification_text_report(
        unseen_y1.values, champion_data_1['test_predictions'], category_names=['Level 0 (Low)', 'Level 1 (Medium)', 'Level 2 (High)']
    ))

    print("=" * 108)
    print(f"[CHAMPION ON UNSEEN DATA] DATASET 2 (STRESS TYPE): {champion_name_2}")
    print(f"   - Unseen Test Accuracy: {champion_data_2['holdout_accuracy']*100:.2f}% ({champion_data_2['correct_count']}/{champion_data_2['total_count']} Correct)")
    print(f"   - Validation Accuracy:  {champion_data_2['validation_accuracy']*100:.2f}% (+/- {champion_data_2['validation_std']*200:.2f}%)")
    print(f"   - Weighted F1 Score:    {champion_data_2['holdout_f1']:.4f}")
    print("\nDetailed Classification Breakdown on Unseen Data:")
    print(AssessmentDiagnostics.build_classification_text_report(
        unseen_y2.values, champion_data_2['test_predictions'], category_names=list(encoder_d2.classes_)
    ))

    # Confusion Matrices on Unseen Data
    unseen_cm_d1 = AssessmentDiagnostics.calculate_normalized_confusion_matrix(
        unseen_y1.values, champion_data_1['test_predictions']
    )
    unseen_cm_d2 = AssessmentDiagnostics.calculate_normalized_confusion_matrix(
        unseen_y2.values, champion_data_2['test_predictions']
    )

    # Feature Importance Calculations
    attribution_d1 = AssessmentDiagnostics.rank_feature_influences(
        champion_data_1['fitted_estimator'], features_d1.columns.tolist()
    )
    attribution_d2 = AssessmentDiagnostics.rank_feature_influences(
        champion_data_2['fitted_estimator'], features_d2.columns.tolist()
    )

    # Step 8: Export Visualizations
    print("\n[Phase 6] Exporting Visualizations & Persisting Models...")
    
    # 1. Unseen Data Performance Dashboard
    fig_unseen = plot_generator.render_unseen_performance_dashboard(
        novel_d1, novel_d2,
        unseen_cm_d1, ['Level 0 (Low)', 'Level 1 (Medium)', 'Level 2 (High)'],
        unseen_cm_d2, list(encoder_d2.classes_)
    )

    # 2. Baseline vs Novel Benchmark & Confusion
    fig_bench = plot_generator.render_benchmark_and_confusion(
        baseline_d1, novel_d1,
        baseline_d2, novel_d2,
        unseen_cm_d1, ['Level 0 (Low)', 'Level 1 (Medium)', 'Level 2 (High)'],
        unseen_cm_d2, list(encoder_d2.classes_)
    )

    # 3. Feature Importance Ranking
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
    print(f" 1. {fig_unseen.name} (Unseen Data Performance Dashboard)")
    print(f" 2. {fig_bench.name}")
    print(f" 3. {fig_feat.name}")

    print(f"\nModel Artifacts Successfully Saved to: {app_config.SAVED_MODELS_DIR}")
    print(" - stress_level_model.joblib")
    print(" - stress_type_model.joblib")

    print("\n" + "=" * 108)
    print("PIPELINE EXECUTION COMPLETE & UNSEEN ACCURACY RECORDED!")
    print("=" * 108)


if __name__ == '__main__':
    execute_stress_analysis_pipeline()
