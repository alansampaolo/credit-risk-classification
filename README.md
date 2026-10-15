# Credit Risk Classification with Machine Learning

Predicting whether a Lending Club loan will be **fully repaid** or **charged off** (default), and comparing seven machine learning models on unseen data.

## Objective

The goal was to **build and compare classification models that could be used to predict the credit risk of future loan applicants**, identifying the model that best distinguishes higher-risk from lower-risk loans on data it has never seen.

## Key results

Evaluated on a held-out test set of 10,000 loans (default rate: 19.9%):

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.697** | **0.373** | 0.647 | 0.308 |
| XGBoost | 0.669 | 0.329 | 0.121 | 0.425 |
| Random Forest | 0.666 | 0.310 | 0.155 | 0.371 |
| Neural Network (MLP) | 0.657 | 0.327 | 0.494 | 0.309 |
| Support Vector Classifier* | 0.657 | 0.308 | 0.609 | 0.291 |
| KNN | 0.599 | 0.248 | 0.534 | 0.253 |
| Decision Tree | 0.546 | 0.217 | 0.320 | 0.258 |

\* Trained on a stratified subsample of 10,000 training loans for computational reasons.

- **Logistic regression performs best**: it ranks risky loans above safe ones better than more complex models (ROC-AUC 0.70, where 0.5 is random) and is almost twice as good as a random classifier at finding defaults (PR-AUC 0.37 vs. 0.20).
- **Accuracy is misleading here**: XGBoost and Random Forest reach about 78–79% accuracy, but mostly by predicting "no default" and miss over 80% of actual defaults.

## Method

1. **Data and features** — Lending Club loan-level data (random sample of 50,000 loans), using only information available at origination. Engineered financial indicators such as loan-to-income ratio, credit usage ratio, average FICO score and credit age at the loan issue date.
2. **Train/test split first** — stratified 80/20 split performed before any statistical preprocessing, so the test set keeps the real distribution of borrowers and default rate.
3. **Preprocessing without leakage** — ordinal encoding for loan grade and one-hot encoding for home ownership; outlier removal estimated and applied on the training set only (37,120 loans retained); min-max scaling fitted inside the model pipeline.
4. **Class imbalance** — SMOTE oversampling applied inside the same pipeline, so scaling and synthetic defaults are fitted only on training folds and never touch validation folds or the test set.
5. **Model comparison** — stratified 5-fold cross-validation on the training set and a final evaluation on the held-out test set, using ROC-AUC, PR-AUC, precision, recall and F1 rather than accuracy alone.
6. **Exploration** — K-Means clustering and PCA on the training set to look for natural borrower segments (exploratory only, not used as predictors).
7. **SAFE AI** — explainability (RGE) and robustness (RGR) of each model.

## SAFE AI findings (logistic regression)

- **Explainability (RGE):** loan amount and instalment are the variables that most affect the model's predictions, followed by renting status and interest rate.
- **Robustness (RGR):** predictions are very stable under perturbation for most variables (RGR close to 1); the model is more sensitive to loan amount, instalment and interest rate, i.e. its most important inputs.

## Limitations

- Hyperparameters are not tuned: tree-based models in particular could improve.
- The classification threshold is fixed at 0.5; in practice it should reflect the relative cost of a missed default versus a rejected good borrower.
- The Support Vector Classifier is trained on a smaller subsample, so its ranking is not fully comparable with the other models.

## Repository

- `credit_risk_classification.ipynb` — full Python workflow
- `credit_risk_classification_presentation.pdf` — project presentation

The Lending Club dataset (~1.5 GB) is not included.

**Tools:** Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · Matplotlib · seaborn
