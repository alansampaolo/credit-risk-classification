# Credit Risk Classification with Machine Learning

Machine learning project for predicting whether Lending Club loans will be fully repaid or charged off.

The project develops an end-to-end credit risk classification pipeline including data preprocessing, feature engineering, unsupervised exploration, supervised model comparison and SAFE AI robustness and explainability analysis.

## Project Overview

The objective is to classify loans into two outcomes:

- `Fully Paid`
- `Charged Off`

The project investigates how borrower characteristics, loan features and engineered financial indicators can be used to predict credit risk.

The workflow includes:

1. data preprocessing and feature engineering;
2. categorical encoding and scaling;
3. exploratory data analysis;
4. unsupervised learning with clustering and PCA;
5. supervised classification;
6. model comparison using multiple evaluation metrics;
7. SAFE AI analysis using RGE and RGR.

## Dataset

The project is based on Lending Club loan data.

Each observation represents a loan and includes information on:

- loan characteristics;
- borrower income;
- credit history;
- FICO score;
- debt-to-income ratio;
- home ownership;
- account activity;
- loan status.

The original dataset is not included in this repository because of its large file size.

The notebook expects the dataset to be available locally under the following filename:

`dataset_1_lendingclub.csv`

## Feature Engineering

Several financially meaningful variables are created from the original data.

Examples include:

- `credit_age` — length of the borrower's credit history;
- `credit_usage_ratio` — revolving credit usage relative to available revolving credit;
- `loan_to_income_ratio` — loan amount relative to annual income;
- `fico_avg` — average of the reported FICO score range.

These engineered variables are designed to capture borrower leverage, credit history and overall credit quality.

## Feature Selection

The modeling dataset includes variables from three main groups.

### Loan characteristics

- Loan amount
- Interest rate
- Installment
- Grade
- Debt-to-Income ratio

### Credit profile

- FICO score
- Credit age
- Open accounts
- Total accounts
- Public records

### Engineered features

- Credit usage ratio
- Public record indicator
- Recent activity score
- Loan-to-income ratio
- Average FICO score

## Categorical Encoding

Different encoding strategies are used depending on the type of categorical variable.

### Ordinal variables

Loan grade has a natural ranking:

`A > B > C > D > E > F > G`

The categories are mapped to numerical values while preserving this ordering.

### Nominal variables

Home ownership has no natural ranking and is therefore transformed using one-hot encoding.

The original categories include:

- Mortgage
- Rent
- Own
- Any
- Other
- None

A separate binary dummy variable is initially created for each category.

Some low-frequency categories are later removed before the supervised modeling stage.

## Missing Values

Missing observations are inspected before modeling.

Most variables contain very limited missingness, while some engineered variables such as credit usage ratio and recent activity score contain approximately 5% missing values.

The dataset is cleaned before the final modeling stage.

## Exploratory Data Analysis

Exploratory analysis is used to understand:

- variable distributions;
- outliers;
- class frequencies;
- correlations;
- relationships between predictors;
- target behavior.

The analysis includes histograms, boxplots, frequency plots and correlation heatmaps.

## Feature Scaling

Numerical variables are standardized using:

`z = (x - mean) / standard deviation`

This transformation centers variables around zero and rescales them to unit standard deviation.

Standardization is preferred to min-max normalization for the modeling pipeline.

## Class Imbalance

The target variable is imbalanced.

Fully Paid loans are approximately five times more frequent than Charged Off loans.

For this reason, model evaluation is not based only on accuracy.

The project also considers:

- Precision;
- Recall;
- F1-score;
- AUC.

This is particularly important in credit risk, where failing to identify a risky borrower can be more costly than incorrectly flagging a safe borrower.

## Unsupervised Learning

Before supervised classification, the project performs an unsupervised exploration of the data.

### K-Means Clustering

K-Means clustering is used to identify groups of loans with similar borrower and loan characteristics without using the target variable.

The objective is exploratory: to determine whether natural borrower segments emerge from the feature space.

### Principal Component Analysis

PCA is used to reduce the multidimensional feature space to two principal components.

This allows the clusters to be visualized in two dimensions.

The cluster labels are not used as predictors in the final supervised models.

## Supervised Classification

The following classification models are compared:

- Logistic Regression
- Neural Network
- K-Nearest Neighbours
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Classifier

The positive class corresponds to:

`Charged Off = 1`

while:

`Fully Paid = 0`

## Confusion Matrix

For each model, a confusion matrix is used to evaluate classification errors.

For the Random Forest model:

```text
                Predicted 0   Predicted 1
True 0              147           19
True 1               18          136

This corresponds to:

- True Negatives: `147`
- False Positives: `19`
- False Negatives: `18`
- True Positives: `136`

In a credit risk context, False Negatives are particularly relevant because they represent loans predicted as safe that are actually charged off.

## Evaluation Metrics

### Accuracy

Measures the proportion of correctly classified observations.

`Accuracy = (TP + TN) / Total`

### Precision

Measures how many loans predicted as Charged Off are actually Charged Off.

`Precision = TP / (TP + FP)`

### Recall

Measures how many actual Charged Off loans are correctly identified.

`Recall = TP / (TP + FN)`

### F1-score

Provides a balance between Precision and Recall.

`F1 = 2 * (Precision * Recall) / (Precision + Recall)`

### AUC

The Area Under the ROC Curve measures the ability of the model to rank risky borrowers above safer borrowers across different classification thresholds.

An AUC close to `1` indicates strong discrimination, while an AUC around `0.5` corresponds approximately to random ranking.

## Model Results

The main results are:

| Model | Accuracy | Precision | Recall | F1-score | AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.884 | 0.877 | 0.883 | 0.880 | 0.94 |
| XGBoost | 0.850 | 0.844 | 0.844 | 0.844 | 0.93 |
| KNN | 0.769 | 0.698 | 0.916 | 0.792 | 0.88 |
| Decision Tree | 0.781 | 0.759 | 0.799 | 0.778 | 0.78 |
| Neural Network | 0.738 | 0.703 | 0.786 | 0.742 | 0.80 |
| SVC | 0.691 | 0.663 | 0.727 | 0.693 | 0.76 |
| Logistic Regression | 0.669 | 0.638 | 0.721 | 0.677 | 0.73 |

Random Forest provides the strongest overall performance.

KNN achieves the highest Recall, identifying the largest proportion of actual Charged Off loans.

## SAFE AI Analysis

The best-performing Random Forest model is also evaluated using SAFE AI metrics.

The analysis focuses on:

- Explainability through RGE;
- Robustness through RGR.

## Rank Graduation Explainability

RGE compares:

- predictions from the full model;
- predictions from a reduced model excluding one feature.

The objective is to assess how much the model behavior changes when a particular feature is removed.

A larger contribution indicates that the feature plays a more important role in explaining the model's predictions.

The highest-ranked variables include:

1. Grade
2. Inquiries in the last 6 months
3. Loan-to-income ratio
4. Interest rate
5. Open accounts
6. Annual income
7. FICO range high
8. Debt-to-Income ratio
9. Credit age
10. Average FICO

The ranking shows that credit quality, recent credit activity and borrower leverage are important drivers of the Random Forest predictions.

## Rank Graduation Robustness

RGR evaluates the stability of model predictions under perturbations of the input data.

It compares:

- predictions generated from the original data;
- predictions generated after perturbing the input features.

The metric lies between `0` and `1`.

Values close to `1` indicate that the model remains highly stable after perturbation.

In this project, most RGR values are very close to `1`, indicating strong robustness of the Random Forest model under the perturbations considered.

Some examples include:

- `home_ownership_ANY ≈ 1.000`
- `credit_age ≈ 0.997`
- `interest_rate ≈ 0.990`
- `grade ≈ 0.984`

The relatively lower values for variables such as Grade and Interest Rate indicate that the model is somewhat more sensitive to perturbations in these important predictors.

## Main Findings

- Feature engineering improves the financial interpretation of the input data.
- The dataset exhibits substantial class imbalance, making Recall, F1-score and AUC important evaluation metrics.
- Unsupervised clustering provides an exploratory view of borrower segmentation.
- PCA allows the multidimensional borrower profiles to be visualized in two dimensions.
- Random Forest achieves the strongest overall classification performance.
- KNN achieves the highest Recall and identifies the largest proportion of charged-off loans.
- Grade, recent credit inquiries, loan-to-income ratio and interest rate are among the most relevant explanatory variables.
- RGR values close to one indicate that the Random Forest predictions are generally robust to the perturbations considered.
- SAFE AI metrics complement traditional predictive metrics by adding explainability and robustness analysis.

## Repository Structure

```text
credit-risk-classification/
│
├── README.md
├── credit_risk_classification.ipynb
└── credit_risk_classification_presentation.pdf
```

## Main Files

- `credit_risk_classification.ipynb` — complete Python workflow including preprocessing, feature engineering, exploratory analysis, clustering, PCA, supervised classification and SAFE AI analysis
- `credit_risk_classification_presentation.pdf` — project presentation containing methodology, model comparison and final results

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- Machine Learning
- Credit Risk
- Classification
- Unsupervised Learning
- SAFE AI

## Limitations

The original Lending Club dataset is not included in this repository because of its large file size.

The notebook therefore requires the dataset to be downloaded separately and placed in the working directory under the filename:

`dataset_1_lendingclub.csv`

The project is intended as a machine learning and credit risk modeling exercise rather than as a production-ready credit scoring system.
