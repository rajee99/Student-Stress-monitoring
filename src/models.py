"""
Classification Models Engine (Zero Trees, Zero Regressions).

Features:
1. Standard Non-Tree, Non-Regression Baselines:
   - Support Vector Machine (RBF Kernel)
   - Multilayer Perceptron (MLP Neural Net)
2. Novel Custom Developed Classifiers:
   - KernelManifoldAttentionClassifier (KMAC - Metric Learning & Prototype Attention)
   - ResidualGatedFeatureClassifier (RGFN - Tabular Squeeze-and-Excitation & Hyperspherical Cosine Head)

Strict Constraints Adhered To:
- ZERO Tree-based algorithms (No Random Forest, Decision Trees, Gradient Boosting, XGBoost, etc.)
- ZERO Regression algorithms (No Logistic Regression, Linear Regression, Ridge, Lasso, etc.)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.novel_models import (
    KernelManifoldAttentionClassifier,
    ResidualGatedFeatureClassifier
)


def get_non_tree_baseline_models(random_seed: int = 42) -> Dict[str, Any]:
    """
    Standard non-tree, non-regression baseline classification models.
    """
    return {
        'SVM RBF (Baseline)': SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=random_seed
        ),
        'MLP Neural Net (Baseline)': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            alpha=0.001,
            max_iter=500,
            random_state=random_seed
        )
    }


def get_novel_non_tree_models(random_seed: int = 42) -> Dict[str, Any]:
    """
    Novel custom-developed Machine Learning classification models
    (Strictly Zero Trees, Zero Regressions).
    """
    return {
        'SVM RBF (Kernel Benchmark)': SVC(
            C=1.5,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=random_seed
        ),
        'MLP Neural Net (Deep MLP)': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            alpha=0.001,
            learning_rate_init=0.002,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=1000,
            random_state=random_seed
        ),
        'Kernel Manifold Attention (Novel Custom 1)': KernelManifoldAttentionClassifier(
            prototypes_per_class=4,
            temperature=0.45,
            kernel_hybrid_alpha=0.65,
            learning_rate=0.04,
            max_iter=250,
            l2_regularization=1e-4,
            random_state=random_seed
        ),
        'Residual Gated FeatureNet (Novel Custom 2)': ResidualGatedFeatureClassifier(
            hidden_dim=128,
            num_blocks=3,
            dropout=0.15,
            cos_scale=18.0,
            learning_rate=0.003,
            weight_decay=1e-4,
            batch_size=32,
            epochs=140,
            random_state=random_seed
        )
    }


def find_optimal_feature_scaler(
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    random_seed: int = 42,
    cv_partitions: int = 5
) -> Tuple[str, Any]:
    """
    Select the feature normalization technique yielding top baseline cross-validation score
    using a non-tree probe (RBF Kernel Support Vector Classifier).
    """
    candidate_scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }

    cv_splitter = StratifiedKFold(n_splits=cv_partitions, shuffle=True, random_state=random_seed)
    scaler_performance_map = {}

    for scaler_name, scaler_instance in candidate_scalers.items():
        scaled_matrix = scaler_instance.fit_transform(training_features)
        benchmark_probe = SVC(C=1.0, kernel='rbf', random_state=random_seed)
        cv_accuracy = cross_val_score(
            benchmark_probe, scaled_matrix, training_labels, cv=cv_splitter, scoring='accuracy'
        ).mean()
        scaler_performance_map[scaler_name] = cv_accuracy

    optimal_scaler_name = max(scaler_performance_map, key=scaler_performance_map.get)
    return optimal_scaler_name, candidate_scalers[optimal_scaler_name]


class ModelBenchmarkingService:
    """Executes cross-validation and out-of-sample evaluations across candidate classifiers."""

    @staticmethod
    def evaluate_model_dictionary(
        model_dict: Dict[str, Any],
        scaled_train_x: np.ndarray,
        train_y: np.ndarray,
        scaled_test_x: np.ndarray,
        test_y: np.ndarray,
        random_seed: int = 42,
        k_folds: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate a dictionary of classification models and return detailed metric records."""
        cv_strategy = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        benchmark_summary = {}

        for model_name, estimator in model_dict.items():
            try:
                # 1. Validation Accuracy via K-Fold Cross Validation
                cv_scores = cross_val_score(
                    estimator, scaled_train_x, train_y, cv=cv_strategy, scoring='accuracy'
                )
                validation_accuracy = float(cv_scores.mean())
                validation_std = float(cv_scores.std())

                # 2. Fit model and compute Training Accuracy
                estimator.fit(scaled_train_x, train_y)
                train_predictions = estimator.predict(scaled_train_x)
                train_accuracy = float(accuracy_score(train_y, train_predictions))

                # 3. Predict on Held-Out Test Data
                test_predictions = estimator.predict(scaled_test_x)
                test_accuracy = float(accuracy_score(test_y, test_predictions))
                test_f1 = float(f1_score(test_y, test_predictions, average='weighted'))

                predicted_probabilities = (
                    estimator.predict_proba(scaled_test_x)
                    if hasattr(estimator, 'predict_proba') else None
                )

                benchmark_summary[model_name] = {
                    'fitted_estimator': estimator,
                    'train_accuracy': train_accuracy,
                    'validation_accuracy': validation_accuracy,
                    'validation_std': validation_std,
                    'cross_val_mean': validation_accuracy,
                    'cross_val_std': validation_std,
                    'holdout_accuracy': test_accuracy,
                    'holdout_f1': test_f1,
                    'test_predictions': test_predictions,
                    'test_probabilities': predicted_probabilities
                }
            except Exception as failure_reason:
                print(f"[NOTE] Model {model_name} bypassed: {failure_reason}")

        return benchmark_summary
