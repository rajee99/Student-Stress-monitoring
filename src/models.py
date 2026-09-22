"""
Classification Models Engine (Zero Regression Algorithms).
Evaluates the core non-linear classification models discussed in the Research Paper:
1. Random Forest Classifier
2. Gradient Boosting Classifier
3. Support Vector Machine Classifier (RBF Kernel)
4. Multilayer Perceptron (MLP Neural Network)

Provides:
- Paper Baseline Configurations (default parameters)
- Fine-Tuned Configurations (optimized for peak accuracy & anti-overfitting)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


def get_paper_baseline_models(random_seed: int = 42) -> Dict[str, Any]:
    """
    Instantiate the core classification models using the default hyperparameters
    specified in Table 3 of the published research paper.
    """
    return {
        'Random Forest (Paper Baseline)': RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_features='sqrt',
            random_state=random_seed
        ),
        'Gradient Boosting (Paper Baseline)': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_seed
        ),
        'SVM RBF (Paper Baseline)': SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=random_seed
        ),
        'MLP Neural Net (Paper Baseline)': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            alpha=0.001,
            max_iter=500,
            random_state=random_seed
        )
    }


def get_finetuned_models(random_seed: int = 42) -> Dict[str, Any]:
    """
    Instantiate fine-tuned classification models with optimized depth, leaf constraints,
    and anti-overfitting regularization.
    """
    return {
        'Random Forest (Fine-Tuned)': RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=random_seed
        ),
        'Gradient Boosting (Fine-Tuned)': GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.06,
            max_depth=3,
            min_samples_split=8,
            min_samples_leaf=4,
            subsample=0.85,
            random_state=random_seed
        ),
        'SVM RBF (Fine-Tuned)': SVC(
            C=1.5,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=random_seed
        ),
        'MLP Neural Net (Fine-Tuned)': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            alpha=0.001,
            learning_rate_init=0.002,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=1000,
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
    Select the feature normalization technique yielding top baseline cross-validation score.
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
        benchmark_probe = RandomForestClassifier(
            n_estimators=80, max_depth=6, min_samples_leaf=4, random_state=random_seed
        )
        cv_accuracy = cross_val_score(
            benchmark_probe, scaled_matrix, training_labels, cv=cv_splitter, scoring='accuracy'
        ).mean()
        scaler_performance_map[scaler_name] = cv_accuracy

    optimal_scaler_name = max(scaler_performance_map, key=scaler_performance_map.get)
    return optimal_scaler_name, candidate_scalers[optimal_scaler_name]


class ModelBenchmarkingService:
    """Executes cross-validation and out-of-sample evaluations across the candidate classifiers."""

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
