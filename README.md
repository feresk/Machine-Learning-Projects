# 🚀 Machine Learning Projects

End-to-end machine learning projects tackling real-world classification problems, with a focus on **imbalanced learning, feature engineering, and ensemble modeling**. Each project goes beyond baseline accuracy to address the practical challenges of noisy, skewed, and high-dimensional data.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/statsmodels-32a852.svg?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-blue?style=for-the-badge&logo=python&logoColor=white)

---

## 📂 Projects Overview

| Project | Problem Type | Dataset Size | Key Techniques | Best Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Credit Card Fraud Detection** | Binary Classification (Imbalanced) | 284,807 transactions | Hybrid Sampling, Dual-Model Architecture, Stacking | Precision: **0.9718** · F1: **0.9517** |
| **Date Fruit Classification** | Multi-class Classification | 897 samples · 34 features | SMOTE, P-value Feature Selection, Horizontal Split Ensemble | Accuracy: **0.9389** · Macro F1: **0.9254** |

---

## 💳 Credit Card Fraud Detection

Detecting fraudulent transactions on a severely imbalanced dataset (0.17% fraud rate) using a multi-stage sampling pipeline, dual-model architecture, and multiple ensemble/stacking strategies.

**Dataset:** 284,807 credit card transactions by European cardholders — PCA-transformed features V1–V28, `Amount`, and binary `Class` label.

### Highlights

- **Hybrid 3-stage sampling pipeline:** IQR-based outlier removal → majority class downsampling (99/1 → 90/10) → SMOTE or ADASYN oversampling (50/50)
- **Dual-model architecture:** separate classifiers trained on clean transactions and on majority-class outlier rows, each with independent feature selection
- **Multiple stacking strategies:** linear ensemble, decision function stacking, predict-proba stacking, log-proba stacking — evaluated against each other
- **SMOTE vs. ADASYN:** SMOTE consistently outperformed on precision and F1; ADASYN yielded marginally higher recall and AUC-ROC
- **Statistical feature selection:** `statsmodels.Logit` p-value analysis used to identify and remove 13 low-discriminative PCA features

### Key Result

| Metric | Score |
| :--- | :---: |
| Precision | **0.9718** |
| Recall | 0.9324 |
| F1-score | **0.9517** |
| AUC-ROC | 0.9662 |

---

## 🌴 Date Fruit Classification

Multi-class classification of 7 date fruit varieties from morphological, shape, and color features — with multiple feature reduction and ensemble strategies compared.

**Dataset:** 897 samples (1 outlier removed) across 7 varieties: BERHI, DEGLET, DOKOL, IRAQI, ROTANA, SAFAVI, SOGAY — 34 features covering area, perimeter, axis lengths, eccentricity, shape factors, and color channel statistics.

### Highlights

- **P-value feature reduction:** `statsmodels.Logit` identified 9 statistically insignificant features — removing them improved accuracy from 92.2% → **93.9%**
- **Custom feature grouping:** color channel triplets (RR/RG/RB) averaged into single descriptors for domain-informed dimensionality reduction
- **Horizontal split ensemble:** KNN-based class grouping splits the 7 classes into two subsets (D1: 4 classes, D2: 3 classes); a binary router model directs each instance to the appropriate specialist — sub-models achieved **96.2%** and **96.0%** accuracy on their respective groups
- **Feature group ablation:** morphological-only (77.2%), shape-only, and color-only models trained separately to quantify each group's individual contribution

### Key Result

| Metric | Score |
| :--- | :---: |
| Accuracy | **0.9389** |
| Macro F1 | **0.9254** |
| Macro Precision | 0.9345 |
| Macro Recall | 0.9217 |

---

## ⚙️ Core Techniques

- Exploratory Data Analysis — boxplots, histograms, class distribution analysis
- Feature Engineering & Selection — p-value testing, variance-based filtering, domain-informed grouping
- Handling Class Imbalance — SMOTE, ADASYN, hybrid downsampling + oversampling pipelines
- Ensemble Learning & Model Stacking — linear ensemble, decision function / predict-proba / log-proba stacking
- Statistical Validation — `statsmodels` Logit for coefficient significance testing
- Evaluation — precision, recall, F1, F-beta, AUC-ROC, average precision; ROC / PR / calibration curves; learning curves

---
