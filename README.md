# Credit Risk Classification with Machine Learning

Predicting whether a Lending Club loan will be **fully repaid** or **charged off** (default), and comparing seven machine learning models on unseen data.

## Objective

The goal was to **build and compare classification models that could be used to predict the credit risk of future loan applicants**, identifying the model that best distinguishes higher-risk from lower-risk loans on data it has never seen.

## Key results

Evaluated on a held-out test set of 10,000 loans (default rate: 19.6%):

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.692** | **0.345** | 0.652 | 0.303 |
| Support Vector Classifier | 0.685 | 0.326 | 0.658 | 0.298 |
| XGBoost | 0.676 | 0.321 | 0.089 | 0.399 |
| Random Forest | 0.672 | 0.309 | 0.144 | 0.363 |
| Neural Network (MLP) | 0.660 | 0.311 | 0.527 | 0.293 |
| KNN | 0.603 | 0.248 | 0.532 | 0.250 |
| Decision Tree | 0.538 | 0.211 | 0.304 | 0.245 |

- **Logistic regression performs best**: it ranks risky loans above safe ones better than more complex models (ROC-AUC 0.69, where 0.5 is random) and is about 75% better than a random classifier at finding defaults (PR-AUC 0.35 vs. 0.20).
- **Accuracy is misleading here**: XGBoost and Random Forest reach ~80% accuracy, but mostly by predicting "no default" and miss over 85% of actual defaults.

## Method

1. **Data and features** — Lending Club loan-level data (random sample of 50,000 loans). Engineered financial indicators such as loan-to-income ratio, credit usage ratio, credit age and average FICO score.
2. **Preprocessing** — ordinal encoding for loan grade, one-hot encoding for home ownership, outlier removal and min-max scaling.
3. **Exploration** — K-Means clustering and PCA to look for natural borrower segments (exploratory only, not used as predictors).
4. **Class imbalance** — SMOTE oversampling applied **inside the cross-validation pipeline**, only on training folds. Validation folds and the test set keep the real class distribution, so there is no data leakage.
5. **Model comparison** — stratified 5-fold cross-validation and a final evaluation on a held-out test set (80/20 split), using ROC-AUC, PR-AUC, precision, recall and F1 rather than accuracy alone.
6. **SAFE AI** — explainability (RGE) and robustness (RGR) of each model.

## SAFE AI findings (logistic regression)

- **Explainability (RGE):** loan amount, instalment and interest rate are the variables that most affect the model's predictions.
- **Robustness (RGR):** predictions are very stable under perturbation for most variables (RGR close to 1); the model is somewhat more sensitive to loan amount, interest rate and instalment, i.e. its most important inputs.

## Limitations

- Hyperparameters are not tuned: tree-based models in particular could improve.
- The classification threshold is fixed at 0.5; in practice it should reflect the relative cost of a missed default versus a rejected good borrower.
- This is a modelling exercise, not a production credit scoring system.

## Repository

- `credit_risk_classification.ipynb` — full Python workflow
- `credit_risk_classification_presentation.pdf` — project presentation

The Lending Club dataset (~1.5 GB) is not included.

**Tools:** Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · Matplotlib · seaborn
