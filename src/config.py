"""
Global Configuration & Environment Settings for Stress Assessment Pipeline.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class AppConfig:
    """Project-level directories, input datasets, and modeling constants."""
    
    # Root directory resolution
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # Raw Dataset Files
    STRESS_LEVEL_DATA_FILE: Path = PROJECT_ROOT / "StressLevelDataset.csv"
    STRESS_TYPE_DATA_FILE: Path = PROJECT_ROOT / "Stress_Dataset.csv"

    # Destination Directories
    SAVED_MODELS_DIR: Path = PROJECT_ROOT / "models"
    OUTPUT_METRICS_DIR: Path = PROJECT_ROOT / "outputs"
    CHARTS_EXPORT_DIR: Path = PROJECT_ROOT / "outputs" / "figures"

    # ML Experiment Constants
    DATA_SPLIT_RANDOM_SEED: int = 42
    HOLDOUT_TEST_FRACTION: float = 0.20
    STRATIFIED_K_FOLDS: int = 5

    # Column Identifiers
    LEVEL_TARGET_LABEL: str = "stress_level"
    TYPE_TARGET_LABEL: str = "Which type of stress do you primarily experience?"
    METRIC_EXCLUSIONS: List[str] = field(default_factory=lambda: [
        "Gender", "Age", "Which type of stress do you primarily experience?"
    ])


# Global Singleton Configuration
app_config = AppConfig()


def initialize_output_directories() -> None:
    """Ensure all required artifact and chart directories exist."""
    app_config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    app_config.OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    app_config.CHARTS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
