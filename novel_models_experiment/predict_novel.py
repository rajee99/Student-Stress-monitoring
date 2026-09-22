"""
Interactive & Batch Inference Utility for Novel Non-Tree Models with XAI & Prescriptive Recommendations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from architectures import (
    KernelManifoldAttentionClassifier,
    ResidualGatedFeatureClassifier
)
from xai_engine import StudentXAIEngine

CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / "models"


def predict_and_explain_student(student_data: dict, student_id: str = "Student-01") -> dict:
    """
    Predict Stress Level and generate XAI Dimension Decomposition with Prescriptive Interventions.
    """
    model = joblib.load(MODELS_DIR / "dataset1_novel_champion.joblib")
    scaler = joblib.load(MODELS_DIR / "dataset1_scaler.joblib")
    features = joblib.load(MODELS_DIR / "dataset1_features.joblib")
    
    df = pd.DataFrame([student_data])
    
    # Feature engineering
    if 'academic_pressure_ratio' not in df.columns:
        df['academic_pressure_ratio'] = (
            (df.get('academic_performance', 0) + df.get('study_load', 0) + df.get('teacher_student_relationship', 0)) /
            (df.get('sleep_quality', 1) + 1.0)
        )
    if 'mental_strain_composite' not in df.columns:
        df['mental_strain_composite'] = (
            df.get('anxiety_level', 0) + df.get('depression', 0) - df.get('self_esteem', 0)
        )
    if 'physiological_load_index' not in df.columns:
        df['physiological_load_index'] = (
            df.get('headache', 0) + df.get('blood_pressure', 0) + df.get('breathing_problem', 0)
        )
        
    df_aligned = df.reindex(columns=features, fill_value=0)
    scaled = scaler.transform(df_aligned)
    
    pred_class = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]
    
    level_names = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
    
    # Extract model metric weights for XAI
    model_weights = getattr(model, 'feature_importances_', np.ones(len(features)) / len(features))
    
    # Run XAI Engine
    explanation = StudentXAIEngine.explain_student_prediction(
        df_aligned.iloc[0],
        pred_class,
        model_weights,
        features
    )
    
    formatted_report = StudentXAIEngine.format_student_xai_report(student_id, explanation)
    
    return {
        'student_id': student_id,
        'predicted_level': int(pred_class),
        'label': level_names.get(int(pred_class), f"Level {pred_class}"),
        'probabilities': {level_names.get(i, f"Level {i}"): float(prob) for i, prob in enumerate(probs)},
        'explanation': explanation,
        'formatted_report': formatted_report
    }


if __name__ == '__main__':
    # Sample Test Student: Academic Stress Case
    academic_stress_student = {
        'anxiety_level': 14, 'self_esteem': 12, 'mental_health_history': 0, 'depression': 10,
        'headache': 2, 'blood_pressure': 1, 'sleep_quality': 2, 'breathing_problem': 1,
        'noise_level': 2, 'living_conditions': 3, 'safety': 3, 'basic_needs': 3,
        'academic_performance': 1, 'study_load': 5, 'teacher_student_relationship': 1,
        'future_career_concerns': 5, 'social_support': 3, 'peer_pressure': 3,
        'extracurricular_activities': 4, 'bullying': 0
    }
    
    res = predict_and_explain_student(academic_stress_student, student_id="Student-Academic-Test")
    print(res['formatted_report'])
