# Student Stress Level & Stress Type Prediction System

An end-to-end, modular machine learning framework for predicting student stress levels (Low, Moderate, High) and identifying specific stress types (Eustress, Distress).

For quick setup and instructions, refer to **[`HOW_TO_RUN.md`](HOW_TO_RUN.md)**.

---

## Quick Start

```powershell
# 1. Create and activate virtual environment
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models and generate figures
python run_pipeline.py

# 4. Run sample predictions
python predict.py --dataset 1
python predict.py --dataset 2
```

---

## Results Summary

| Task | Top Classifier | Scaler | Test Accuracy | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset 1 (Stress Level)** | Random Forest | `StandardScaler` | **90.00%** | **0.9000** |
| **Dataset 2 (Stress Type)** | LightGBM | `MinMaxScaler` | **99.39%** | **0.9937** |
