"""
Classification Model Suite & Benchmarking Engine.
Includes the Paper's exact classification algorithms, fine-tuned ensemble models,
and Deep Learning architectures (LSTM & Squeeze-and-Excitation SE-Blocks).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.dl_models import SETabularClassifier, TabularLSTMClassifier

# Optional gradient boosted tree libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def build_candidate_classifier_suite(random_seed: int = 42) -> Dict[str, Any]:
    """
    Instantiate classification algorithms including:
    1. Paper's exact classification models (fine-tuned & regularized)
    2. Deep Learning models (LSTM & SE-Tabular Network)
    3. High-performance tree & voting ensembles
    """
    classifier_portfolio = {
        # --- The Paper's Exact Classification Models (Fine-Tuned) ---
        'Random Forest (Fine-Tuned)': RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=random_seed
        ),
        'Gradient Boosting (Fine-Tuned)': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=6,
            subsample=0.80,
            random_state=random_seed
        ),
        'SVM Classifier (RBF Kernel)': SVC(
            C=1.2,
            gamma='scale',
            kernel='rbf',
            probability=True,
            random_state=random_seed
        ),
        'MLP Neural Network (Fine-Tuned)': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            alpha=0.02,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_seed,
            max_iter=800
        ),

        # --- Deep Learning Architectures (LSTM & Squeeze-and-Excitation) ---
        'SE-TabularNet (Attention Block)': SETabularClassifier(
            hidden_dim=128,
            epochs=100,
            batch_size=32,
            lr=0.002,
            weight_decay=1e-4,
            dropout=0.3,
            random_state=random_seed
        ),
        'Tabular LSTM (Bi-directional)': TabularLSTMClassifier(
            embedding_dim=32,
            hidden_dim=64,
            epochs=100,
            batch_size=32,
            lr=0.003,
            weight_decay=1e-4,
            dropout=0.25,
            random_state=random_seed
        ),

        # --- Advanced Complementary Ensembles ---
        'Extra Trees Classifier': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=random_seed
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            l2_regularization=3.0,
            random_state=random_seed
        ),
        'Decision Tree (Pruned)': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=random_seed
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=9, weights='distance'),
        'Gaussian Naive Bayes': GaussianNB(var_smoothing=1e-8)
    }

    if HAS_XGB:
        classifier_portfolio['XGBoost Classifier'] = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=4,
            min_child_weight=3,
            reg_lambda=2.0,
            subsample=0.85,
            random_state=random_seed,
            eval_metric='mlogloss',
            verbosity=0
        )
    if HAS_LGB:
        classifier_portfolio['LightGBM Classifier'] = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=5,
            num_leaves=16,
            min_child_samples=8,
            reg_lambda=2.0,
            subsample=0.85,
            random_state=random_seed,
            verbose=-1
        )

    # Soft-Voting Meta-Ensemble combining diverse high-performing classifiers
    voting_estimators = [
        ('rf', classifier_portfolio['Random Forest (Fine-Tuned)']),
        ('gb', classifier_portfolio['Gradient Boosting (Fine-Tuned)']),
        ('svm', classifier_portfolio['SVM Classifier (RBF Kernel)']),
        ('mlp', classifier_portfolio['MLP Neural Network (Fine-Tuned)'])
    ]
    if HAS_LGB:
        voting_estimators.append(('lgb', classifier_portfolio['LightGBM Classifier']))

    classifier_portfolio['Soft-Voting Meta-Ensemble'] = VotingClassifier(
        estimators=voting_estimators,
        voting='soft'
    )

    return classifier_portfolio


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
    """Executes cross-validation and out-of-sample evaluations across the candidate classifier suite."""

    @staticmethod
    def run_portfolio_benchmark(
        scaled_train_x: np.ndarray,
        train_y: np.ndarray,
        scaled_test_x: np.ndarray,
        test_y: np.ndarray,
        random_seed: int = 42,
        k_folds: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit each candidate model, evaluate k-fold cross validation on training data,
        and assess training accuracy, validation accuracy, and holdout test accuracy.
        """
        portfolio = build_candidate_classifier_suite(random_seed=random_seed)
        cv_strategy = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        benchmark_summary = {}

        for model_name, estimator in portfolio.items():
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
