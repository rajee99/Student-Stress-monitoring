"""
Interactive & Batch Inference Utility for Novel Non-Tree Models.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from architectures import (
    KernelManifoldAttentionClassifier,
    ResidualGatedFeatureClassifier
)

CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / "models"


def predict_stress_level(student_data: dict) -> dict:
    """Predict Stress Level using the trained Kernel Manifold Attention Classifier."""
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
    return {
        'predicted_level': int(pred_class),
        'label': level_names.get(int(pred_class), f"Level {pred_class}"),
        'probabilities': {level_names.get(i, f"Level {i}"): float(prob) for i, prob in enumerate(probs)}
    }


if __name__ == '__main__':
    # Sample Test
    sample_student = {
        'anxiety_level': 18, 'self_esteem': 10, 'mental_health_history': 1, 'depression': 15,
        'headache': 3, 'blood_pressure': 2, 'sleep_quality': 1, 'breathing_problem': 3,
        'noise_level': 4, 'living_conditions': 2, 'safety': 2, 'basic_needs': 2,
        'academic_performance': 2, 'study_load': 4, 'teacher_student_relationship': 2,
        'future_career_concerns': 4, 'social_support': 1, 'peer_pressure': 4,
        'extracurricular_activities': 4, 'bullying': 3
    }
    res = predict_stress_level(sample_student)
    print("Inference Test Result:", res)
