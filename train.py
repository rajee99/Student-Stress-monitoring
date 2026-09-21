import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

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


def preprocess_dataset1(data_path="StressLevelDataset.csv"):
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()

    # Feature engineering
    df['mental_health_score'] = (df['anxiety_level'] + df['depression']) / 2.0
    df['physical_health_score'] = (df['headache'] + df['blood_pressure'] + df['breathing_problem']) / 3.0
    df['academic_pressure_score'] = (df['study_load'] + df['academic_performance'] + df['future_career_concerns']) / 3.0
    df['social_pressure_score'] = (df['peer_pressure'] + df['bullying']) / 2.0
    df['anxiety_depression_interaction'] = df['anxiety_level'] * df['depression']
    df['self_esteem_anxiety_ratio'] = df['self_esteem'] / (df['anxiety_level'] + 1.0)
    df['anxiety_squared'] = df['anxiety_level'] ** 2
    df['self_esteem_squared'] = df['self_esteem'] ** 2

    X = df.select_dtypes(include=[np.number]).drop(['stress_level'], axis=1)
    y = df['stress_level']
    X = X.fillna(X.mean())

    return X, y


def preprocess_dataset2(data_path="Stress_Dataset.csv"):
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    df.columns = df.columns.str.replace('.1', '', regex=False)
    df = df.loc[:, ~df.columns.duplicated()]

    target_col = 'Which type of stress do you primarily experience?'
    numeric_cols = [c for c in df.columns if c not in ['Gender', 'Age', target_col]]

    if len(numeric_cols) >= 5:
        df['avg_response_score'] = df[numeric_cols].mean(axis=1)
        df['response_variability'] = df[numeric_cols].std(axis=1)
        df['response_range'] = df[numeric_cols].max(axis=1) - df[numeric_cols].min(axis=1)

    X = df.select_dtypes(include=[np.number])
    X = X.fillna(X.mean())

    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))

    return X, y, le


def get_models():
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=300, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
        'SVM RBF': SVC(kernel='rbf', random_state=42, probability=True),
        'SVM Poly': SVC(kernel='poly', random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
        'Ridge Classifier': RidgeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'KNN Weighted': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'LDA': LinearDiscriminantAnalysis(),
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(200, 100, 50), random_state=42, max_iter=1000)
    }

    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss', verbosity=0)
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)

    return models


def train_and_evaluate(X, y, dataset_name, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print(f"TRAINING ON: {dataset_name}")
    print(f"Dataset Shape: X={X.shape}, y={y.shape} | Classes: {np.unique(y)}")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }

    scaler_scores = {}
    for s_name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X_train)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        score = cross_val_score(rf, X_scaled, y_train, cv=5, scoring='accuracy').mean()
        scaler_scores[s_name] = score

    best_scaler_name = max(scaler_scores, key=scaler_scores.get)
    print(f"Optimal Scaler Selected: {best_scaler_name} (RF CV Score: {scaler_scores[best_scaler_name]:.4f})")

    scaler = scalers[best_scaler_name]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = {}

    print(f"\n{'Model':<25} {'CV Accuracy':<20} {'Test Accuracy':<15} {'F1 (Weighted)':<15}")
    print("-" * 75)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_acc': acc,
                'test_f1': f1,
                'model': model
            }

            print(f"{name:<25} {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})   {acc:<15.4f} {f1:<15.4f}")
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")

    # Best model selection
    best_name = max(results, key=lambda k: (results[k]['test_acc'], results[k]['cv_mean']))
    best_info = results[best_name]

    print("\n" + "-" * 75)
    print(f"[BEST MODEL] {best_name}")
    print(f"   CV Score:      {best_info['cv_mean']:.4f}")
    print(f"   Test Accuracy: {best_info['test_acc']:.4f}")
    print(f"   Test F1 Score: {best_info['test_f1']:.4f}")

    y_pred_best = best_info['model'].predict(X_test_scaled)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_best))

    # Feature Importance if available
    if hasattr(best_info['model'], 'feature_importances_'):
        imp = pd.Series(best_info['model'].feature_importances_, index=X.columns).sort_values(ascending=False)
        print("Top 10 Most Important Features:")
        for rank, (feat, val) in enumerate(imp.head(10).items(), 1):
            print(f"  {rank:2d}. {feat:<32} {val:.4f}")

    # Save artifacts
    prefix = dataset_name.lower().replace(" ", "_")
    model_path = os.path.join(output_dir, f"{prefix}_best_model.joblib")
    scaler_path = os.path.join(output_dir, f"{prefix}_scaler.joblib")
    features_path = os.path.join(output_dir, f"{prefix}_features.joblib")

    joblib.dump(best_info['model'], model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X.columns.tolist(), features_path)

    print(f"\nArtifacts successfully saved:")
    print(f" - Model:    {model_path}")
    print(f" - Scaler:   {scaler_path}")
    print(f" - Features: {features_path}")

    return results, best_name, best_info


def main():
    print("Starting ML Model Training Pipeline...\n")

    # Dataset 1
    d1_path = "StressLevelDataset.csv"
    if os.path.exists(d1_path):
        X1, y1 = preprocess_dataset1(d1_path)
        res1, best1, info1 = train_and_evaluate(X1, y1, "Dataset 1 - Stress Level")
    else:
        print(f"Warning: {d1_path} not found!")

    print("\n\n")

    # Dataset 2
    d2_path = "Stress_Dataset.csv"
    if os.path.exists(d2_path):
        X2, y2, le2 = preprocess_dataset2(d2_path)
        res2, best2, info2 = train_and_evaluate(X2, y2, "Dataset 2 - Stress Type")
        joblib.dump(le2, os.path.join("models", "dataset_2_-_stress_type_label_encoder.joblib"))
    else:
        print(f"Warning: {d2_path} not found!")

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
