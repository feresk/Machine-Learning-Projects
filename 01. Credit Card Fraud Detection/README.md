# Credit Card Fraud Detection

This project detects fraudulent credit card transactions using a sophisticated pipeline that combines outlier-aware data cleaning, hybrid resampling, feature selection, and ensemble/stacked Logistic Regression models. The notebook covers EDA, class imbalance handling, dual-model training (one for clean transactions, one for outlier transactions), and several ensemble strategies.

## Table of Contents
* About the Dataset
* Project Overview
* Model Results
* Summary

---

## About the Dataset

The [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contains credit card transactions made by European cardholders in September 2013. It includes:

* **Time:** Seconds elapsed since the first transaction in the dataset.
* **V1 – V28:** Numerical features from a PCA transformation (original features withheld for confidentiality).
* **Amount:** Transaction amount.
* **Class:** Target variable — `1` = fraudulent, `0` = legitimate.

The dataset contains **284,807 transactions**, of which only **492 (0.17%)** are fraudulent, making standard accuracy a misleading metric. Precision, recall, F1-score, F-beta score, and AUC-ROC are used instead.

---

## Project Overview

### A. EDA

* **Class Imbalance Analysis:** Confirmed a severe 99.83% / 0.17% class split.
* **Feature Distribution:** Visualized all features using boxplots and histograms, both overall and split by class, to identify distributional differences.
* **Correlation Analysis:** Identified the most discriminative PCA features by comparing per-class means. Features with normalized mean distances below a threshold (0.3) were flagged as low-information and later removed: `V8, V13, V15, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount`.

### B. Data Preprocessing

* **Outlier Removal (Step 1 of hybrid sampling):** IQR-based outlier detection was applied to the majority class. Rows with 3 or more outlier features were removed, reducing the dataset from 284,807 to **254,768 rows** while retaining all 492 fraud cases. Removed majority-class outlier rows were preserved separately for a secondary model.
* **Downsampling (Step 2):** The cleaned majority class was downsampled from a 99.8/0.2 ratio to approximately **90/10**, yielding a training subset of ~4,180 rows.
* **Oversampling with SMOTE / ADASYN (Step 3):** The minority class was oversampled to a **50/50 ratio** using SMOTE (`k_neighbors=7`) and ADASYN (`n_neighbors=7`), then compared.
* **Feature Scaling:** `StandardScaler` was applied to `Time` and `Amount` (PCA features are already standardized).
* **Train/Test Split:** 85/15 stratified split (`random_state=10`).

### C. Modeling

All models use **Logistic Regression** with L2 regularization and class-weighted training. Two separate classifiers are trained:

**Main classifier** (trained on clean transactions):
* LR + SMOTE without standardization
* LR + SMOTE with standardization
* LR + SMOTE with standardization + reduced feature set
* LR + ADASYN without standardization
* LR + ADASYN with standardization

**Outlier classifier** (trained on majority-class outlier rows vs. fraud):
* LR + SMOTE with standardization (full features)
* LR + SMOTE with standardization + reduced features (based on normal data)
* LR + SMOTE with standardization + reduced features (based on outlier data)

**Ensemble / Stacking strategies:**
* **Linear Ensemble:** Combines predictions of the main and outlier classifiers via a weighted alpha parameter.
* **Method 1 – Decision Function Stacking:** Passes the decision function outputs of both classifiers as features to a meta Logistic Regression.
* **Method 2a – Predict Proba Stacking:** Uses predicted probabilities as meta-features.
* **Method 2b – Log Proba Stacking:** Uses log-probabilities as meta-features (with ±30 clipping for infinities).
* **Method 3 – Outlier Detection Model:** Trains a dedicated outlier detector, then routes transactions to the appropriate classifier.

Statistical significance of features in the final model was verified using `statsmodels.Logit` — all retained features showed p-values of 0.0000.

### D. Model Performance

#### Main Classifier — SMOTE variants

| Model | Precision | Recall | AUC-ROC | Avg Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| LR + SMOTE (non-standardized) | 0.8214 | 0.9324 | 0.9660 | 0.7661 | 0.8734 |
| LR + SMOTE (standardized) | 0.9324 | 0.9324 | 0.9662 | 0.8696 | 0.9324 |
| LR + SMOTE (standardized + reduced features) | **0.9718** | 0.9324 | 0.9662 | 0.9063 | **0.9517** |

#### Main Classifier — ADASYN variants

| Model | Precision | Recall | AUC-ROC | Avg Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| LR + ADASYN (non-standardized) | 0.6731 | **0.9459** | **0.9725** | 0.6368 | 0.7865 |
| LR + ADASYN (standardized) | 0.6481 | **0.9459** | 0.9725 | 0.6132 | 0.7692 |

#### Outlier Classifier

| Model | Precision | Recall | AUC-ROC | Avg Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| LR + SMOTE (standardized) | 0.7922 | 0.8243 | 0.9104 | 0.6559 | 0.8079 |
| LR + SMOTE (stand. + normal-data reduced) | 0.8451 | 0.8108 | 0.9042 | 0.6882 | 0.8276 |
| LR + SMOTE (stand. + outlier-data reduced) | 0.8243 | 0.8243 | 0.9107 | 0.6823 | 0.8243 |

#### Ensemble & Stacking Models

| Method | Precision | Recall | AUC-ROC | Avg Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| Linear Ensemble | 0.5037 | 0.9189 | 0.9579 | 0.4631 | 0.6507 |
| Method 1 – Decision Function | 0.9971 | 0.8195 | 0.9089 | 0.8910 | 0.8996 |
| Method 2a – Predict Proba | 0.9971 | 0.8240 | 0.9112 | 0.8936 | 0.9023 |
| Method 2b – Log Proba | 0.9970 | 0.7956 | 0.8970 | 0.8768 | 0.8850 |
| Method 3 – Outlier Detector (normal split) | 0.8492 | 0.6913 | 0.8384 | 0.6196 | 0.7622 |
| Method 3 – Outlier Detector (alt. split) | 0.8342 | 0.7011 | 0.8423 | 0.6163 | 0.7618 |

---

## Summary

This project presents an end-to-end fraud detection pipeline on a real-world severely imbalanced dataset:

* **Key Challenge:** The 0.17% fraud rate requires moving beyond accuracy as a metric and applying careful multi-step imbalance handling.
* **Preprocessing Strategy:** A three-stage hybrid sampling pipeline — outlier removal, majority class downsampling, then minority class oversampling via SMOTE — was found to be more effective than naive resampling alone.
* **Dual-Model Design:** Separating "normal" transactions from majority-class outliers and training a dedicated classifier on each improved recall and precision for edge cases.
* **Best Single Classifier:** LR + SMOTE (standardized, reduced features) achieved the best balance of precision (0.972) and F1-score (0.952) on the test set.
* **Best Stacked Model:** Method 2a (predict-proba stacking) achieved the highest combined stacking F1 (0.902) with near-perfect training precision (0.997), though with slightly lower recall (0.824) compared to ADASYN models.
* **SMOTE vs. ADASYN:** SMOTE consistently outperformed ADASYN on precision and F1-score, while ADASYN yielded marginally higher recall and AUC-ROC.
* **Feature Insights:** 13 PCA-transformed features (`V8, V13, V15, V20–V28, Amount`) showed low discriminative power between classes and were safely removed without hurting performance.
