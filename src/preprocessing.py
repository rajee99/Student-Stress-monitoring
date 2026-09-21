"""
Feature Engineering & Transformation Pipeline.
Synthesizes domain psychological indices, interaction signals, and behavioral statistics.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class StudentFeatureTransformer:
    """Transforms raw survey responses into engineered feature representations."""

    @classmethod
    def transform_stress_level_dataset(
        cls, student_survey_dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Derives physiological, psychological, and social strain indices for Stress Level modeling.
        """
        cleaned_records = student_survey_dataframe.copy().drop_duplicates()

        # Psychological Distress Composite
        cleaned_records['mental_health_score'] = (
            cleaned_records['anxiety_level'] + cleaned_records['depression']
        ) / 2.0

        # Somatic / Physiological Strain Composite
        cleaned_records['physical_health_score'] = (
            cleaned_records['headache'] + 
            cleaned_records['blood_pressure'] + 
            cleaned_records['breathing_problem']
        ) / 3.0

        # Academic Load & Pressure Composite
        cleaned_records['academic_pressure_score'] = (
            cleaned_records['study_load'] + 
            cleaned_records['academic_performance'] + 
            cleaned_records['future_career_concerns']
        ) / 3.0

        # Interpersonal / Social Adversity Composite
        cleaned_records['social_pressure_score'] = (
            cleaned_records['peer_pressure'] + cleaned_records['bullying']
        ) / 2.0

        # Non-linear Interaction & Buffering Ratios
        cleaned_records['anxiety_depression_interaction'] = (
            cleaned_records['anxiety_level'] * cleaned_records['depression']
        )
        cleaned_records['self_esteem_anxiety_ratio'] = (
            cleaned_records['self_esteem'] / (cleaned_records['anxiety_level'] + 1.0)
        )

        # Higher-Order Polynomial Dynamics
        cleaned_records['anxiety_squared'] = cleaned_records['anxiety_level'] ** 2
        cleaned_records['self_esteem_squared'] = cleaned_records['self_esteem'] ** 2

        # Extract features and target
        target_series = cleaned_records['stress_level']
        feature_matrix = cleaned_records.select_dtypes(include=[np.number]).drop(
            columns=['stress_level'], errors='ignore'
        )

        # Fill any missing values with column means
        feature_matrix = feature_matrix.fillna(feature_matrix.mean())

        return feature_matrix, target_series

    @classmethod
    def transform_stress_type_dataset(
        cls, survey_records_dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
        """
        Sanitizes survey header anomalies and computes response dispersion metrics for Stress Type modeling.
        """
        deduplicated_dataframe = survey_records_dataframe.copy().drop_duplicates()

        # Sanitize column headers (remove duplicate suffixes like '.1')
        deduplicated_dataframe.columns = deduplicated_dataframe.columns.str.replace(r'\.1$', '', regex=True)
        deduplicated_dataframe = deduplicated_dataframe.loc[:, ~deduplicated_dataframe.columns.duplicated()]

        target_column_name = 'Which type of stress do you primarily experience?'
        non_question_columns = ['Gender', 'Age', target_column_name]
        survey_question_columns = [
            col for col in deduplicated_dataframe.columns if col not in non_question_columns
        ]

        # Compute response dynamics across survey items
        if len(survey_question_columns) >= 5:
            deduplicated_dataframe['avg_response_score'] = deduplicated_dataframe[survey_question_columns].mean(axis=1)
            deduplicated_dataframe['response_variability'] = deduplicated_dataframe[survey_question_columns].std(axis=1)
            deduplicated_dataframe['response_range'] = (
                deduplicated_dataframe[survey_question_columns].max(axis=1) -
                deduplicated_dataframe[survey_question_columns].min(axis=1)
            )

        # Extract numeric feature matrix
        feature_matrix = deduplicated_dataframe.select_dtypes(include=[np.number])
        feature_matrix = feature_matrix.fillna(feature_matrix.mean())

        # Encode target categories
        stress_category_encoder = LabelEncoder()
        target_encoded_series = stress_category_encoder.fit_transform(
            deduplicated_dataframe[target_column_name].astype(str)
        )
        target_series = pd.Series(target_encoded_series, name=target_column_name)

        return feature_matrix, target_series, stress_category_encoder
