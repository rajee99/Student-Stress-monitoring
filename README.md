# Student Stress Monitoring: Advanced Machine Learning & Explainable AI Framework

An end-to-end, modular machine learning and deep learning framework for student mental health monitoring. Evaluates both **Stress Severity** (Low, Medium, High) and **Stress Typology** (Eustress vs Distress vs No Stress), featuring custom first-principles algorithms, Explainable AI (XAI) factor decomposition, and prescriptive intervention mapping.

---

## 🌟 Key Highlights & Core Novelties

1. **Zero Trees & Zero Regressions Paradigm (`novel-non-tree-classifiers` branch):**
   - **`KernelManifoldAttentionClassifier` (KMAC):** Multi-prototype Riemannian manifold clustering + learnable Mahalanobis precision weights ($\mathbf{w} = \exp(\boldsymbol{\theta})$) + temperature-scaled hybrid kernel attention.
   - **`ResidualGatedFeatureClassifier` (RGFN):** Layer-Normalized Feature-Gated Linear Units (GLU) + Squeeze-and-Excitation (SE) channel recalibration + Hyperspherical Cosine Similarity classification head on $\mathbb{S}^{d-1}$.
2. **Superior Generalization on 100% Unseen Data:**
   - **Dataset 1 (Stress Level):** **`90.45%` Unseen Accuracy** (199/220 correct) vs. paper baseline of $89.09\%$.
   - **Dataset 2 (Stress Type):** **`96.34%` to `98.17%` Unseen Accuracy** vs. paper baseline of $93.59\%$.
3. **Integrated Explainable AI (XAI) & Prescriptive Recommendation Engine:**
   - 5-Dimensional Stress Factor Decomposition (*Academic, Psychological, Physical, Environmental, Social*).
   - Prescriptive intervention triggers (e.g. *Academic Stress $\to$ Study load reduction plan, Time management coaching, Academic advisor meeting*).

---

## 📂 Repository Structure

```text
├── StressLevelDataset.csv                  # Dataset 1: 1,100 records, 20 survey features
├── Stress_Dataset.csv                       # Dataset 2: 816 records, 26 survey features
├── run_pipeline.py                          # Master baseline & fine-tuning runner
├── evaluate_large_unseen.py                 # Large-sample unseen test & Monte Carlo bootstrap
├── predict.py                               # CLI inference for standard models
│
├── novel_models_experiment/                 # Standalone Novel Non-Tree Architecture Suite
│   ├── architectures.py                     # KMAC and RGFN custom ML classifiers
│   ├── xai_engine.py                        # 5-Dimension XAI decomposition & prescriptive rules
│   ├── train_and_evaluate.py                # Retraining, 5-fold CV, & report generation
│   ├── predict_novel.py                     # Single-student inference with XAI recommendations
│   ├── NOVEL_MODELS_RESEARCH_REPORT.md      # Detailed mathematical formulations & empirical report
│   ├── models/                              # Serialized .joblib champion models
│   └── outputs/figures/                     # Publication-quality figures & heatmaps
│
├── src/                                     # Core modular package
│   ├── config.py                            # Path configs & hyperparameters
│   ├── data_loader.py                       # Ingestion & health audit
│   ├── preprocessing.py                     # Feature engineering & scaling
│   ├── models.py                            # Estimator suites & non-tree benchmarks
│   ├── novel_models.py                      # KMAC & RGFN scikit-learn compatible classes
│   ├── evaluate.py                          # Metrics & contingency diagnostics
│   └── visualize.py                         # Chart rendering engine
│
├── FINAL_RESEARCH_REPORT.md                 # Complete research report
└── HOW_TO_RUN.md                            # Detailed step-by-step execution guide
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```powershell
# Create and activate Python 3.11 virtual environment
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Novel Non-Tree Machine Learning Experiment
```powershell
# Switch to the novel classifiers branch
git checkout novel-non-tree-classifiers

# Train, evaluate, and generate full research report with XAI figures
py -3.11 novel_models_experiment/train_and_evaluate.py

# Run single-student inference with XAI prescriptive recommendations
py -3.11 novel_models_experiment/predict_novel.py
```

### 3. Run Large Unseen Generalization Benchmarks
```powershell
# Monte Carlo 100-run simulation & 30%/40% unseen sample tests
py -3.11 evaluate_large_unseen.py
```

---

## 📊 Empirical Results Summary

### Dataset 1: Student Stress Level ($N = 220$ Unseen Students)
| Classifier Name | Paradigm | 5-Fold CV Acc (%) | Unseen Test Acc (%) | Weighted F1 | Correct / Total |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Kernel Manifold Attention** | **Novel Custom 1** | **87.05%** | **90.45%** | **0.9043** | **199 / 220** |
| Residual Gated FeatureNet | Novel Custom 2 | 87.84% | 88.18% | 0.8804 | 194 / 220 |
| MLP Neural Net | Baseline Deep MLP | 87.05% | 88.18% | 0.8819 | 194 / 220 |
| SVM RBF | Baseline Kernel | 86.70% | 87.27% | 0.8723 | 192 / 220 |

### Dataset 2: Student Stress Type ($N = 164$ Unseen Students)
| Classifier Name | Paradigm | 5-Fold CV Acc (%) | Unseen Test Acc (%) | Weighted F1 | Correct / Total |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Residual Gated FeatureNet** | **Novel Custom 2** | **94.02%** | **96.34%** | **0.9611** | **158 / 164** |
| SVM RBF | Baseline Kernel | 95.71% | 96.34% | 0.9585 | 158 / 164 |
| Kernel Manifold Attention | Novel Custom 1 | 93.41% | 93.90% | 0.9463 | 154 / 164 |
| MLP Neural Net | Baseline Deep MLP | 93.71% | 91.46% | 0.8739 | 150 / 164 |

---

## 🌿 Git Branches

- **[`main`](https://github.com/rajee99/Student-Stress-monitoring/tree/main):** Core baseline reproduction and fine-tuned tree/non-tree models.
- **[`novel-non-tree-classifiers`](https://github.com/rajee99/Student-Stress-monitoring/tree/novel-non-tree-classifiers):** Strictly Zero-Tree, Zero-Regression custom metric & hyperspherical architectures + XAI prescriptive recommendations.
- **[`deep-learning-lstm-se`](https://github.com/rajee99/Student-Stress-monitoring/tree/deep-learning-lstm-se):** PyTorch Bi-LSTM sequence embeddings and Squeeze-and-Excitation networks.
