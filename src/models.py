"""
Model Registry, Adaptive Preconditioning, and Cross-Validation Engine.
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
    BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# Optional high-performance tree libraries
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
    Instantiate a diverse ensemble of linear, kernel, tree-based, and boosting classifiers.
    """
    classifier_portfolio = {
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=random_seed),
        'Extra Trees': ExtraTreesClassifier(n_estimators=300, random_state=random_seed),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=random_seed),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=random_seed, max_iter=200),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=random_seed),
        'Bagging Ensemble': BaggingClassifier(n_estimators=100, random_state=random_seed),
        'SVM (RBF Kernel)': SVC(kernel='rbf', random_state=random_seed, probability=True),
        'SVM (Poly Kernel)': SVC(kernel='poly', random_state=random_seed, probability=True),
        'Linear SVM': LinearSVC(random_state=random_seed, max_iter=2000, dual='auto'),
        'Logistic Regression': LogisticRegression(random_state=random_seed, max_iter=2000),
        'Ridge Classifier': RidgeClassifier(random_state=random_seed),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Distance-Weighted KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'Gaussian Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=random_seed, max_depth=10),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'MLP Deep Neural Net': MLPClassifier(
            hidden_layer_sizes=(200, 100, 50), random_state=random_seed, max_iter=1000
        )
    }

    if HAS_XGB:
        classifier_portfolio['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, random_state=random_seed, eval_metric='mlogloss', verbosity=0
        )
    if HAS_LGB:
        classifier_portfolio['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200, random_state=random_seed, verbose=-1
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
        benchmark_probe = RandomForestClassifier(n_estimators=100, random_state=random_seed)
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
        and assess generalization accuracy on the held-out test split.
        """
        portfolio = build_candidate_classifier_suite(random_seed=random_seed)
        cv_strategy = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        benchmark_summary = {}

        for model_name, estimator in portfolio.items():
            try:
                cv_scores = cross_val_score(
                    estimator, scaled_train_x, train_y, cv=cv_strategy, scoring='accuracy'
                )
                estimator.fit(scaled_train_x, train_y)
                predicted_labels = estimator.predict(scaled_test_x)

                predicted_probabilities = (
                    estimator.predict_proba(scaled_test_x)
                    if hasattr(estimator, 'predict_proba') else None
                )

                test_accuracy = accuracy_score(test_y, predicted_labels)
                test_f1 = f1_score(test_y, predicted_labels, average='weighted')

                benchmark_summary[model_name] = {
                    'fitted_estimator': estimator,
                    'cross_val_mean': float(cv_scores.mean()),
                    'cross_val_std': float(cv_scores.std()),
                    'holdout_accuracy': float(test_accuracy),
                    'holdout_f1': float(test_f1),
                    'test_predictions': predicted_labels,
                    'test_probabilities': predicted_probabilities
                }
            except Exception as failure_reason:
                print(f"[NOTE] Model {model_name} bypassed: {failure_reason}")

        return benchmark_summary
