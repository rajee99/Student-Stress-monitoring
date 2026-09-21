"""
Dataset Ingestion, Verification, and Schema Integrity Auditing.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd


def verify_file_existence(target_path: Path) -> None:
    """Validate that the designated dataset file is present on the filesystem."""
    if not target_path.exists():
        raise FileNotFoundError(f"Missing required dataset: '{target_path.resolve()}'")


class DataAuditService:
    """Services for reading raw records and computing data health summaries."""

    @staticmethod
    def audit_tabular_health(raw_dataframe: pd.DataFrame, dataset_identifier: str) -> Dict[str, Any]:
        """
        Extract tabular diagnostics including missingness, duplicates, and schema dimensions.
        """
        total_missing_cells = int(raw_dataframe.isnull().sum().sum())
        total_duplicate_rows = int(raw_dataframe.duplicated().sum())
        total_records, total_attributes = raw_dataframe.shape
        data_type_breakdown = raw_dataframe.dtypes.value_counts().to_dict()

        return {
            "dataset_identifier": dataset_identifier,
            "sample_count": total_records,
            "feature_count": total_attributes,
            "missing_cells": total_missing_cells,
            "duplicate_rows": total_duplicate_rows,
            "type_distribution": data_type_breakdown
        }

    @staticmethod
    def read_csv_records(dataset_file_path: Path) -> pd.DataFrame:
        """Read CSV records into a pandas DataFrame after verifying existence."""
        verify_file_existence(dataset_file_path)
        survey_dataframe = pd.read_csv(dataset_file_path)
        return survey_dataframe
