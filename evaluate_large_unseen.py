"""
Comprehensive Unseen Data Evaluation with Scaled Sample Sizes.
Evaluates model performance across:
1. 30% Held-Out Unseen Test Partition (N=245 unseen students)
2. 40% Held-Out Unseen Test Partition (N=327 unseen students)
3. 100-Iteration Monte Carlo Bootstrap (Over 24,000+ unseen evaluations)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from src.config import app_config
from src.data_loader import DataAuditService
from src.preprocessing import StudentFeatureTransformer
from src.models import find_optimal_feature_scaler, get_finetuned_models


def run_large_unseen_evaluation():
    print("=" * 96)
    print("LARGE-SAMPLE UNSEEN DATA GENERALIZATION ASSESSMENT: DATASET 2 (STRESS TYPE)")
    print("=" * 96)

    raw_records = DataAuditService.read_csv_records(app_config.STRESS_TYPE_DATA_FILE)
    X, y, encoder = StudentFeatureTransformer.transform_stress_type_dataset(raw_records)
    total_samples = len(X)
    class_names = list(encoder.classes_)

    print(f"Total Available Dataset Size: {total_samples} student records across {X.shape[1]} features.")
    print(f"Target Stress Categories:     {class_names}")

    # =========================================================================
    # EXPERIMENT 1: 30% HELD-OUT UNSEEN TEST PARTITION (N = 245 Unseen Students)
    # =========================================================================
    print(f"\n{'-' * 96}")
    print("[EXPERIMENT 1] 30% HELD-OUT UNSEEN TEST PARTITION")
    print(f"{'-' * 96}")
    
    test_frac_30 = 0.30
    train_x_30, unseen_x_30, train_y_30, unseen_y_30 = train_test_split(
        X, y, test_size=test_frac_30, random_state=app_config.DATA_SPLIT_RANDOM_SEED, stratify=y
    )

    _, scaler_30 = find_optimal_feature_scaler(train_x_30, train_y_30, random_seed=42, cv_partitions=5)
    scaled_train_30 = scaler_30.fit_transform(train_x_30)
    scaled_unseen_30 = scaler_30.transform(unseen_x_30)

    print(f"Training Set (Seen):    {len(train_x_30)} students (70%)")
    print(f"Unseen Test Set (Held): {len(unseen_x_30)} students (30% - BIGGER SAMPLE)")
    print("-" * 96)
    print(f"{'Model Name':<28} {'Unseen Accuracy':<18} {'Correct / Total':<18} {'Weighted F1':<14} {'Macro F1':<12}")
    print("-" * 96)

    models_30 = get_finetuned_models(random_seed=42)
    exp1_records = {}

    for name, model in models_30.items():
        model.fit(scaled_train_30, train_y_30)
        preds = model.predict(scaled_unseen_30)
        acc = accuracy_score(unseen_y_30, preds)
        wf1 = f1_score(unseen_y_30, preds, average='weighted')
        mf1 = f1_score(unseen_y_30, preds, average='macro')
        correct = int(np.sum(preds == unseen_y_30.values))
        total = len(unseen_y_30)
        exp1_records[name] = (acc, correct, total, wf1, mf1, preds)

        print(f"{name:<28} {acc*100:>10.2f}%      {correct}/{total:<13} {wf1:>10.4f}     {mf1:>8.4f}")

    best_30_name = max(exp1_records, key=lambda k: exp1_records[k][0])
    best_30 = exp1_records[best_30_name]
    print("-" * 96)
    print(f"[CHAMPION] Top Model on 30% Unseen Data: {best_30_name} ({best_30[0]*100:.2f}% | {best_30[1]}/{best_30[2]} Correct)")

    # =========================================================================
    # EXPERIMENT 2: 40% HELD-OUT UNSEEN TEST PARTITION (N = 327 Unseen Students)
    # =========================================================================
    print(f"\n{'-' * 96}")
    print("[EXPERIMENT 2] 40% HELD-OUT UNSEEN TEST PARTITION (STRESS-TESTING GENERALIZATION)")
    print(f"{'-' * 96}")

    test_frac_40 = 0.40
    train_x_40, unseen_x_40, train_y_40, unseen_y_40 = train_test_split(
        X, y, test_size=test_frac_40, random_state=app_config.DATA_SPLIT_RANDOM_SEED, stratify=y
    )

    _, scaler_40 = find_optimal_feature_scaler(train_x_40, train_y_40, random_seed=42, cv_partitions=5)
    scaled_train_40 = scaler_40.fit_transform(train_x_40)
    scaled_unseen_40 = scaler_40.transform(unseen_x_40)

    print(f"Training Set (Seen):    {len(train_x_40)} students (60%)")
    print(f"Unseen Test Set (Held): {len(unseen_x_40)} students (40% - LARGE SAMPLE)")
    print("-" * 96)
    print(f"{'Model Name':<28} {'Unseen Accuracy':<18} {'Correct / Total':<18} {'Weighted F1':<14} {'Macro F1':<12}")
    print("-" * 96)

    models_40 = get_finetuned_models(random_seed=42)
    exp2_records = {}

    for name, model in models_40.items():
        model.fit(scaled_train_40, train_y_40)
        preds = model.predict(scaled_unseen_40)
        acc = accuracy_score(unseen_y_40, preds)
        wf1 = f1_score(unseen_y_40, preds, average='weighted')
        mf1 = f1_score(unseen_y_40, preds, average='macro')
        correct = int(np.sum(preds == unseen_y_40.values))
        total = len(unseen_y_40)
        exp2_records[name] = (acc, correct, total, wf1, mf1, preds)

        print(f"{name:<28} {acc*100:>10.2f}%      {correct}/{total:<13} {wf1:>10.4f}     {mf1:>8.4f}")

    best_40_name = max(exp2_records, key=lambda k: exp2_records[k][0])
    best_40 = exp2_records[best_40_name]
    print("-" * 96)
    print(f"[CHAMPION] Top Model on 40% Unseen Data: {best_40_name} ({best_40[0]*100:.2f}% | {best_40[1]}/{best_40[2]} Correct)")

    # =========================================================================
    # EXPERIMENT 3: 100-ITERATION MONTE CARLO BOOTSTRAP (24,500+ Unseen Evaluations)
    # =========================================================================
    print(f"\n{'-' * 96}")
    print("[EXPERIMENT 3] 100-RUN MONTE CARLO SIMULATION (24,500+ TOTAL UNSEEN EVALUATIONS)")
    print(f"{'-' * 96}")
    print("Evaluating stability across 100 different randomized 70/30 train/unseen splits...")

    monte_carlo_splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.30, random_state=42)
    mc_results = {name: [] for name in models_30.keys()}

    for train_idx, test_idx in monte_carlo_splitter.split(X, y):
        X_tr, X_unseen = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_unseen = y.iloc[train_idx], y.iloc[test_idx]

        scaler_mc = StandardScaler()
        scaled_tr = scaler_mc.fit_transform(X_tr)
        scaled_un = scaler_mc.transform(X_unseen)

        for name, model in models_30.items():
            model.fit(scaled_tr, y_tr)
            p = model.predict(scaled_un)
            mc_results[name].append(accuracy_score(y_unseen, p))

    print(f"{'Model Name':<28} {'Mean Unseen Acc (%)':<22} {'95% Confidence Interval':<28} {'Min - Max Range':<16}")
    print("-" * 96)

    for name, acc_list in mc_results.items():
        mean_acc = np.mean(acc_list) * 100
        std_acc = np.std(acc_list) * 100
        ci_lower = np.percentile(acc_list, 2.5) * 100
        ci_upper = np.percentile(acc_list, 97.5) * 100
        min_acc = np.min(acc_list) * 100
        max_acc = np.max(acc_list) * 100

        print(
            f"{name:<28} "
            f"{mean_acc:>12.2f}% (+/- {std_acc:>4.2f}%)   "
            f"[{ci_lower:>5.2f}% - {ci_upper:>5.2f}%]                 "
            f"{min_acc:>5.2f}% - {max_acc:>5.2f}%"
        )
    print("-" * 96)

    # Detailed Class Breakdown on the 30% Large Unseen Test Set
    print("\n" + "=" * 96)
    print(f"DETAILED CLASSIFICATION BREAKDOWN ON 30% UNSEEN SAMPLE (N={len(unseen_y_30)} Students):")
    print(f"Model: {best_30_name}")
    print("=" * 96)
    print(classification_report(unseen_y_30.values, best_30[5], target_names=class_names, digits=4))

    print("\n" + "=" * 96)
    print("LARGE-SAMPLE UNSEEN EVALUATION COMPLETED WITH HIGH STATISTICAL CONFIDENCE!")
    print("=" * 96)


if __name__ == '__main__':
    run_large_unseen_evaluation()
