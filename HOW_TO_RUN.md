# How to Run: Student Stress Prediction ML Framework

This guide walks you step-by-step through setting up a Python virtual environment (`.venv`), installing dependencies, fixing common PowerShell execution policy issues, training the pipeline, and running predictions.

---

## 1. Prerequisites

- **Python 3.10 or 3.11** installed on your system.
- Check your Python installation:
  ```powershell
  python --version
  # OR if using Python launcher:
  py -3.11 --version
  ```

---

## 2. Setting Up Virtual Environment (`.venv`)

Open PowerShell in the project directory (`D:\ml`):

```powershell
# 1. Create a virtual environment named .venv
python -m venv .venv

# (If using py launcher with specific Python 3.11 version):
py -3.11 -m venv .venv
```

---

## 3. Activating the Virtual Environment (PowerShell Fix)

### If PowerShell blocks activation:
When trying to run `.\.venv\Scripts\Activate.ps1`, PowerShell might show an error:
> *"File ... cannot be loaded because running scripts is disabled on this system."*

### Solution A: Bypass Execution Policy for Current Session (Recommended)
Run this command in your PowerShell window before activating:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Solution B: Set ExecutionPolicy for Current User (Permanent)
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Solution C: Run Directly Without Activating
You can always invoke the virtual environment's Python executable directly:
```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

---

## 4. Install Dependencies

With the virtual environment active (or pointing to `.venv\Scripts\pip.exe`):

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt
```

---

## 5. Running the Full Training & Evaluation Pipeline

Execute the master pipeline to train models across both datasets, compute cross-validation, and generate diagnostic figures:

```powershell
python run_pipeline.py
```

### What happens when you run this:
1. Ingests and audits both datasets (`StressLevelDataset.csv` and `Stress_Dataset.csv`).
2. Applies domain feature engineering (composite scores, interaction terms, response variability).
3. Evaluates 16+ classifiers with 5-fold stratified cross-validation and adaptive scalers.
4. Generates and saves 2 publication-quality visualization figures to `outputs/figures/`.
5. Persists the best models, scalers, and encoders to `models/`.

---

## 6. Running Inference / Predictions

Use [`predict.py`](file:///D:/ml/predict.py) to run predictions on new sample data.

### Predict Student Stress Level (Dataset 1):
```powershell
python predict.py --dataset 1
```
**Sample Output:**
```json
{
  "predicted_class": 2,
  "stress_level": "High Stress",
  "class_probabilities": [0.216, 0.220, 0.564]
}
```

### Predict Student Stress Type (Dataset 2):
```powershell
python predict.py --dataset 2
```
**Sample Output:**
```json
{
  "predicted_class_id": 1,
  "stress_type": "Eustress (Positive Stress) - Stress that motivates and enhances performance.",
  "class_probabilities": [0.000, 1.000, 0.000]
}
```

---

## 7. Project Structure & Output Artifacts

```
D:\ml\
├── requirements.txt           # Dependency specifications
├── HOW_TO_RUN.md              # Setup and execution guide
├── run_pipeline.py            # Master training & visualization pipeline
├── predict.py                 # Standalone prediction script
├── src\
│   ├── config.py              # Configuration paths and hyperparameters
│   ├── data_loader.py         # Data loading and health checks
│   ├── preprocessing.py       # Domain feature engineering logic
│   ├── models.py              # Model suite (16+ algorithms) & scaling
│   ├── evaluate.py            # Metrics, classification reports & importance
│   └── visualize.py           # Clean chart generator
├── models\                    # Saved .joblib model artifacts
│   ├── stress_level_model.joblib
│   ├── stress_level_scaler.joblib
│   ├── stress_type_model.joblib
│   └── stress_type_scaler.joblib
└── outputs\
    └── figures\               # Generated diagnostic charts (.png)
        ├── benchmark_and_confusion.png
        └── feature_importance_ranking.png
```

---

## 8. Quick Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `PSSecurityException` | PowerShell script policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `ModuleNotFoundError` | Package not installed in current Python | Verify `.venv` is active: `where python` should show `.venv\Scripts\python.exe` |
| `FileNotFoundError` for CSVs | Script run from wrong directory | Make sure you run commands from `D:\ml` root |
