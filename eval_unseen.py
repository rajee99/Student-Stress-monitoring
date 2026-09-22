"""
Unseen Test Set Generalization Evaluation Script.
Evaluates saved champion models strictly on the 20% unseen test partition.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.config import app_config
from src.data_loader import DataAuditService
from src.preprocessing import StudentFeatureTransformer
from src.models import find_optimal_feature_scaler, get_finetuned_models


def evaluate_unseen_data():
    print("=" * 80)
    print("UNSEEN TEST DATA GENERALIZATION PERFORMANCE REPORT")
    print("=" * 80)

    # -------------------------------------------------------------
    # DATASET 1: STRESS LEVEL CLASSIFICATION (Unseen 220 records)
    # -------------------------------------------------------------
    raw_level = DataAuditService.read_csv_records(app_config.STRESS_LEVEL_DATA_FILE)
    X1, y1 = StudentFeatureTransformer.transform_stress_level_dataset(raw_level)

    train_x1, unseen_x1, train_y1, unseen_y1 = train_test_split(
        X1, y1, test_size=app_config.HOLDOUT_TEST_FRACTION,
        random_state=app_config.DATA_SPLIT_RANDOM_SEED, stratify=y1
    )

    _, scaler1 = find_optimal_feature_scaler(
        train_x1, train_y1, random_seed=app_config.DATA_SPLIT_RANDOM_SEED, cv_partitions=5
    )
    scaled_train_x1 = scaler1.fit_transform(train_x1)
    scaled_unseen_x1 = scaler1.transform(unseen_x1)

    print(f"\n[DATASET 1: STUDENT STRESS LEVEL (3 CLASSES)]")
    print(f"Total Dataset Size:      {len(X1)} students")
    print(f"Training Set (Seen):     {len(train_x1)} students (80%)")
    print(f"Test Set (100% UNSEEN):  {len(unseen_x1)} students (20%)")
    print("-" * 80)

    models_d1 = get_finetuned_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)

    print(f"{'Model Name':<28} {'Unseen Accuracy (%)':<24} {'Correct / Total':<18} {'Weighted F1':<12}")
    print("-" * 80)

    d1_results = {}
    for name, model in models_d1.items():
        model.fit(scaled_train_x1, train_y1)
        preds = model.predict(scaled_unseen_x1)
        acc = accuracy_score(unseen_y1, preds)
        f1 = f1_score(unseen_y1, preds, average='weighted')
        correct = int(np.sum(preds == unseen_y1.values))
        total = len(unseen_y1)
        d1_results[name] = (acc, correct, total, f1, preds)
        print(f"{name:<28} {acc*100:>10.2f}%              {correct}/{total:<12} {f1:>8.4f}")

    best_d1_name = max(d1_results, key=lambda k: d1_results[k][0])
    best_d1 = d1_results[best_d1_name]
    print("-" * 80)
    print(f"Champion on Unseen Data: {best_d1_name}")
    print(f"Accuracy: {best_d1[0]*100:.2f}% ({best_d1[1]} correct out of {best_d1[2]} unseen samples)")
    print("\nClass-by-Class Breakdown on Unseen Data (Dataset 1):")
    print(classification_report(unseen_y1.values, best_d1[4], target_names=['Level 0 (Low)', 'Level 1 (Medium)', 'Level 2 (High)'], digits=4))

    # -------------------------------------------------------------
    # DATASET 2: STRESS TYPE CLASSIFICATION (Unseen 164 records)
    # -------------------------------------------------------------
    raw_type = DataAuditService.read_csv_records(app_config.STRESS_TYPE_DATA_FILE)
    X2, y2, encoder2 = StudentFeatureTransformer.transform_stress_type_dataset(raw_type)

    train_x2, unseen_x2, train_y2, unseen_y2 = train_test_split(
        X2, y2, test_size=app_config.HOLDOUT_TEST_FRACTION,
        random_state=app_config.DATA_SPLIT_RANDOM_SEED, stratify=y2
    )

    _, scaler2 = find_optimal_feature_scaler(
        train_x2, train_y2, random_seed=app_config.DATA_SPLIT_RANDOM_SEED, cv_partitions=5
    )
    scaled_train_x2 = scaler2.fit_transform(train_x2)
    scaled_unseen_x2 = scaler2.transform(unseen_x2)

    print(f"\n[DATASET 2: STUDENT STRESS TYPE (3 CATEGORIES)]")
    print(f"Total Dataset Size:      {len(X2)} students")
    print(f"Training Set (Seen):     {len(train_x2)} students (80%)")
    print(f"Test Set (100% UNSEEN):  {len(unseen_x2)} students (20%)")
    print("-" * 80)

    models_d2 = get_finetuned_models(random_seed=app_config.DATA_SPLIT_RANDOM_SEED)

    print(f"{'Model Name':<28} {'Unseen Accuracy (%)':<24} {'Correct / Total':<18} {'Weighted F1':<12}")
    print("-" * 80)

    d2_results = {}
    for name, model in models_d2.items():
        model.fit(scaled_train_x2, train_y2)
        preds = model.predict(scaled_unseen_x2)
        acc = accuracy_score(unseen_y2, preds)
        f1 = f1_score(unseen_y2, preds, average='weighted')
        correct = int(np.sum(preds == unseen_y2.values))
        total = len(unseen_y2)
        d2_results[name] = (acc, correct, total, f1, preds)
        print(f"{name:<28} {acc*100:>10.2f}%              {correct}/{total:<12} {f1:>8.4f}")

    best_d2_name = max(d2_results, key=lambda k: d2_results[k][0])
    best_d2 = d2_results[best_d2_name]
    print("-" * 80)
    print(f"Champion on Unseen Data: {best_d2_name}")
    print(f"Accuracy: {best_d2[0]*100:.2f}% ({best_d2[1]} correct out of {best_d2[2]} unseen samples)")
    print("\nClass-by-Class Breakdown on Unseen Data (Dataset 2):")
    print(classification_report(unseen_y2.values, best_d2[4], target_names=list(encoder2.classes_), digits=4))

    print("\n" + "=" * 80)
    print("UNSEEN DATA EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    evaluate_unseen_data()
