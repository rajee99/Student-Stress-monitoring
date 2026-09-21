"""
Standalone Inference & Serving Service for Student Stress Assessment.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib

from src.config import app_config


def evaluate_student_stress_level(survey_attributes_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict overall stress severity category (Level 0: Low, Level 1: Moderate, Level 2: High).
    """
    model_artifact_path = app_config.SAVED_MODELS_DIR / "stress_level_model.joblib"
    scaler_artifact_path = app_config.SAVED_MODELS_DIR / "stress_level_scaler.joblib"
    features_metadata_path = app_config.SAVED_MODELS_DIR / "stress_level_features.joblib"

    if not model_artifact_path.exists():
        raise FileNotFoundError(
            f"Fitted model artifact not found at {model_artifact_path}. Execute run_pipeline.py first."
        )

    classifier_model = joblib.load(model_artifact_path)
    feature_scaler = joblib.load(scaler_artifact_path)
    expected_feature_names = joblib.load(features_metadata_path)

    input_df = pd.DataFrame([survey_attributes_dict])

    # Compute domain transformations
    input_df['mental_health_score'] = (input_df['anxiety_level'] + input_df['depression']) / 2.0
    input_df['physical_health_score'] = (
        input_df['headache'] + input_df['blood_pressure'] + input_df['breathing_problem']
    ) / 3.0
    input_df['academic_pressure_score'] = (
        input_df['study_load'] + input_df['academic_performance'] + input_df['future_career_concerns']
    ) / 3.0
    input_df['social_pressure_score'] = (input_df['peer_pressure'] + input_df['bullying']) / 2.0
    input_df['anxiety_depression_interaction'] = input_df['anxiety_level'] * input_df['depression']
    input_df['self_esteem_anxiety_ratio'] = input_df['self_esteem'] / (input_df['anxiety_level'] + 1.0)
    input_df['anxiety_squared'] = input_df['anxiety_level'] ** 2
    input_df['self_esteem_squared'] = input_df['self_esteem'] ** 2

    # Column alignment & missing imputation
    for feature in expected_feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0.0

    aligned_features = input_df[expected_feature_names]
    preprocessed_array = feature_scaler.transform(aligned_features)

    predicted_class = int(classifier_model.predict(preprocessed_array)[0])
    posterior_probabilities = (
        classifier_model.predict_proba(preprocessed_array)[0].tolist()
        if hasattr(classifier_model, 'predict_proba') else None
    )

    class_description_map = {
        0: "Low Stress Level",
        1: "Moderate Stress Level",
        2: "High Stress Level"
    }

    return {
        "predicted_class_index": predicted_class,
        "stress_level_label": class_description_map.get(predicted_class, f"Level {predicted_class}"),
        "confidence_probabilities": posterior_probabilities
    }


def evaluate_student_stress_type(survey_responses_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict the qualitative stress taxonomy (Eustress vs Distress).
    """
    model_artifact_path = app_config.SAVED_MODELS_DIR / "stress_type_model.joblib"
    scaler_artifact_path = app_config.SAVED_MODELS_DIR / "stress_type_scaler.joblib"
    features_metadata_path = app_config.SAVED_MODELS_DIR / "stress_type_features.joblib"
    encoder_metadata_path = app_config.SAVED_MODELS_DIR / "stress_type_label_encoder.joblib"

    if not model_artifact_path.exists():
        raise FileNotFoundError(
            f"Fitted model artifact not found at {model_artifact_path}. Execute run_pipeline.py first."
        )

    classifier_model = joblib.load(model_artifact_path)
    feature_scaler = joblib.load(scaler_artifact_path)
    expected_feature_names = joblib.load(features_metadata_path)
    label_encoder = joblib.load(encoder_metadata_path)

    input_df = pd.DataFrame([survey_responses_dict])

    # Compute response metrics
    numeric_columns = [col for col in input_df.columns if col not in ['Gender', 'Age']]
    if len(numeric_columns) >= 5:
        input_df['avg_response_score'] = input_df[numeric_columns].mean(axis=1)
        input_df['response_variability'] = input_df[numeric_columns].std(axis=1)
        input_df['response_range'] = input_df[numeric_columns].max(axis=1) - input_df[numeric_columns].min(axis=1)

    for feature in expected_feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0.0

    aligned_features = input_df[expected_feature_names]
    preprocessed_array = feature_scaler.transform(aligned_features)

    predicted_index = int(classifier_model.predict(preprocessed_array)[0])
    decoded_stress_category = label_encoder.inverse_transform([predicted_index])[0]
    posterior_probabilities = (
        classifier_model.predict_proba(preprocessed_array)[0].tolist()
        if hasattr(classifier_model, 'predict_proba') else None
    )

    return {
        "predicted_category_index": predicted_index,
        "stress_type_name": decoded_stress_category,
        "confidence_probabilities": posterior_probabilities
    }


def main():
    cli_parser = argparse.ArgumentParser(description="Inference CLI for Student Stress Assessment.")
    cli_parser.add_argument("--dataset", type=int, choices=[1, 2], default=1, help="Target Dataset (1: Level, 2: Type)")
    cli_args = cli_parser.parse_args()

    if cli_args.dataset == 1:
        sample_assessment = {
            'anxiety_level': 14, 'self_esteem': 20, 'mental_health_history': 0,
            'depression': 11, 'headache': 2, 'blood_pressure': 1, 'sleep_quality': 2,
            'breathing_problem': 4, 'noise_level': 2, 'living_conditions': 3,
            'safety': 3, 'basic_needs': 2, 'academic_performance': 3, 'study_load': 2,
            'teacher_student_relationship': 3, 'future_career_concerns': 3,
            'social_support': 2, 'peer_pressure': 3, 'extracurricular_activities': 3,
            'bullying': 2
        }
        inference_output = evaluate_student_stress_level(sample_assessment)
        print("Inference Result (Stress Level):", inference_output)
    else:
        sample_survey = {
            'Gender': 0, 'Age': 20,
            'Have you recently experienced stress in your life?': 3,
            'Have you noticed a rapid heartbeat or palpitations?': 4,
            'Have you been dealing with anxiety or tension recently?': 2,
            'Do you face any sleep problems or difficulties falling asleep?': 5,
            'Have you been getting headaches more often than usual?': 2,
            'Do you get irritated easily?': 1,
            'Do you have trouble concentrating on your academic tasks?': 2,
            'Have you been feeling sadness or low mood?': 2,
            'Have you been experiencing any illness or health issues?': 3,
            'Do you often feel lonely or isolated?': 1,
            'Do you feel overwhelmed with your academic workload?': 5,
            'Are you in competition with your peers, and does it affect you?': 1,
            'Do you find that your relationship often causes you stress?': 2,
            'Are you facing any difficulties with your professors or instructors?': 3,
            'Is your working environment unpleasant or stressful?': 1,
            'Do you struggle to find time for relaxation and leisure activities?': 4,
            'Is your hostel or home environment causing you difficulties?': 1,
            'Do you lack confidence in your academic performance?': 2,
            'Do you lack confidence in your choice of academic subjects?': 1,
            'Academic and extracurricular activities conflicting for you?': 3,
            'Do you attend classes regularly?': 1,
            'Have you gained/lost weight?': 2
        }
        inference_output = evaluate_student_stress_type(sample_survey)
        print("Inference Result (Stress Type):", inference_output)


if __name__ == '__main__':
    main()
