# first-payment-default-model-evaluation
End-to-end ML model evaluation and selection project for first-payment default prediction, featuring train/validate/test benchmarking across classification models, rigorous ROC-AUC evaluation with experiment tracking, GridSearchCV tuning, class weights/SMOTE, and SHAP interpretability.

# Model Evaluation & Benchmarking — First Payment Default Prediction (Kaggle)

This project focuses on **model evaluation**: building a reproducible pipeline to **train, compare, and validate multiple classification models** on an imbalanced credit-risk dataset. The goal is not just to build a model, but to **measure performance rigorously**, test improvement strategies (tuning + imbalance methods), and communicate results clearly.

## Why this project
In real-world analytics, selecting the “best” model requires more than accuracy. This project demonstrates:
- **Model benchmarking** across algorithms
- **Cross-validated evaluation** and test-set validation
- **Hyperparameter tuning** (GridSearchCV)
- **Imbalance handling** (class weights, SMOTE)
- **Interpretability** (SHAP)

## Dataset
Kaggle competition: *Optimizing Default Model by First Payment Default*  
Target is highly imbalanced (defaults are rare).

> Dataset is not included in this repo. Download from Kaggle and place as `data/kaggle_dataset.csv`.

## Pipeline Overview
### 1) Data Preparation
- Train/validation/test split (stratified)
- Dropped high-missingness features (>35%)
- Median imputation for remaining missing values
- Feature selection: variance threshold, ANOVA, and model-based importance

### 2) Models Evaluated
- Logistic Regression  
- Random Forest  
- XGBoost  
- SVM (RBF)

### 3) Evaluation Metric
Primary metric: **ROC-AUC** (chosen due to class imbalance)

Baseline validation ROC-AUC:
- Logistic: 0.644
- Random Forest: 0.600
- XGBoost: 0.650
- SVM (RBF): 0.554

Baseline test ROC-AUC (XGBoost): 0.681

### 4) Experiments to Improve Performance
- Hyperparameter tuning (GridSearchCV)
- Class weights (`scale_pos_weight` for XGBoost)
- SMOTE oversampling

Key experiment outcomes:
- Tuned XGBoost validation ROC-AUC: ~0.662
- Class-weighted XGBoost test ROC-AUC: ~0.677
- SMOTE test ROC-AUC: ~0.590

### 5) Model Explainability
Used **SHAP** to identify global feature importance and explain an individual high-risk prediction.

## How to Run
1) Install dependencies:
```bash
pip install -r requirements.txt
