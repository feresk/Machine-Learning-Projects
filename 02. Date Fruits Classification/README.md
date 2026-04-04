# Date Fruit Classification

This project classifies seven varieties of date fruits using morphological, shape, and color features extracted from images. The notebook covers EDA, preprocessing, and multiple modeling strategies using Logistic Regression with SMOTE oversampling — including p-value-based feature reduction, custom feature grouping, and a horizontal-split ensemble approach.

## Table of Contents
* About the Dataset
* Project Overview
* Model Results
* Summary

---

## About the Dataset

The [dataset](https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets) is loaded from `Date_Fruit_Datasets.xlsx` and contains **897 samples** (1 outlier removed) across **7 date fruit varieties**:

`BERHI`, `DEGLET`, `DOKOL`, `IRAQI`, `ROTANA`, `SAFAVI`, `SOGAY`

Features include 34 morphological, shape, and color descriptor columns (area, perimeter, axis lengths, eccentricity, shape factors, and color channel statistics). No missing values are present. The dataset has a moderate class imbalance addressed via SMOTE oversampling during training.

---

## Project Overview

### A. EDA

* **Class distribution** is visualized with a bar plot showing per-class sample counts.
* **Feature distributions** across classes are explored using boxplots for all 34 features, grouped by class, to identify discriminative features and detect outliers.
* **Outlier removal:** One sample with an extreme `ASPECT_RATIO > 500` was identified and removed.

### B. Data Preprocessing

* Categorical class labels are encoded numerically.
* **Train/test split:** 80/20 stratified split (`test_size=0.2`, `random_state=10`), yielding a test set of **180 samples**.
* **SMOTE** oversampling is applied to the training set to balance class representation.
* **StandardScaler** is applied to all features before training.

### C. Modeling

Four modeling strategies are explored, all using **Logistic Regression (L2, 500 iterations)**:

**Base Model:** Full feature set + SMOTE + standardization.

**Attempt I — P-value Feature Reduction:** A `statsmodels.Logit` model is fitted to identify statistically insignificant features (p-value > 0.8). Nine features are removed: `ASPECT_RATIO`, `MeanRR`, `MeanRG`, `MeanRB`, `StdDevRR`, `SkewRR`, `ALLdaub4RR`, `ALLdaub4RG`, `ALLdaub4RB`.

**Attempt II — Custom Feature Grouping:** Color channel features (`RR`, `RG`, `RB`) are averaged into single channel-group means, reducing the feature vector dimensionality through domain-informed aggregation.

**Attempt III — Horizontal Split Ensemble:**
* Classes are grouped into two subsets using KNN on per-class feature means:
  * **Group D1** (4 classes): `BERHI`, `IRAQI`, `ROTANA`, `SOGAY`
  * **Group D2** (3 classes): `DEGLET`, `DOKOL`, `SAFAVI`
* Three models are trained:
  * **M1** — Logistic Regression on D1 samples (p-value reduced features)
  * **M2** — Logistic Regression on D2 samples
  * **N (router)** — Binary classifier predicting which group an instance belongs to
* Final prediction: `N * M1_pred + (1 - N) * M2_pred`

**Extra Analysis — Feature Group Ablation:** Models are trained separately on morphological features only (columns 0–11), shape features only (columns 12–15), and color features only (columns 16+) to quantify each group's individual contribution.

### D. Evaluation

All models are evaluated with classification reports (precision, recall, F1 per class) and ROC/Precision-Recall/calibration curve comparisons using `eval_summary`.

---

## Model Results

#### Base and Feature-Reduction Models (test set, 180 samples)

| Model | Accuracy | Macro F1 |
|:---|:---:|:---:|
| LR + SMOTE (full features) | 0.9222 | 0.9021 |
| LR + SMOTE (p-value reduced, −9 features) | **0.9389** | **0.9254** |
| LR + SMOTE (custom grouped features) | 0.8778 | 0.8349 |

#### Per-class Results — Base Model

| Class | Precision | Recall | F1 | Support |
|:---|:---:|:---:|:---:|:---:|
| BERHI | 0.9167 | 0.8462 | 0.8800 | 13 |
| DEGLET | 0.7895 | 0.7500 | 0.7692 | 20 |
| DOKOL | 0.9500 | 0.9268 | 0.9383 | 41 |
| IRAQI | 0.9286 | 0.9286 | 0.9286 | 14 |
| ROTANA | 0.9167 | 1.0000 | 0.9565 | 33 |
| SAFAVI | 1.0000 | 1.0000 | 1.0000 | 40 |
| SOGAY | 0.8421 | 0.8421 | 0.8421 | 19 |

#### Per-class Results — P-value Reduced Model

| Class | Precision | Recall | F1 | Support |
|:---|:---:|:---:|:---:|:---:|
| BERHI | 1.0000 | 0.8462 | 0.9167 | 13 |
| DEGLET | 0.8824 | 0.7500 | 0.8108 | 20 |
| DOKOL | 0.9500 | 0.9268 | 0.9383 | 41 |
| IRAQI | 0.9286 | 0.9286 | 0.9286 | 14 |
| ROTANA | 0.9167 | 1.0000 | 0.9565 | 33 |
| SAFAVI | 1.0000 | 1.0000 | 1.0000 | 40 |
| SOGAY | 0.8636 | 1.0000 | 0.9268 | 19 |

#### Horizontal Split Ensemble — Sub-model Results

| Model | Classes | Accuracy | Macro F1 |
|:---|:---|:---:|:---:|
| M1 (D1 group) | BERHI, IRAQI, ROTANA, SOGAY | 0.9620 | 0.9540 |
| M2 (D2 group) | DEGLET, DOKOL, SAFAVI | 0.9604 | 0.9516 |

#### Feature Group Ablation (Morphological Features Only)

| Class | Precision | Recall | F1 |
|:---|:---:|:---:|:---:|
| BERHI | 0.5000 | 0.5833 | 0.5385 |
| DOKOL | 0.9623 | 0.9623 | 0.9623 |
| SAFAVI | 0.9000 | 0.7297 | 0.8060 |
| **Overall accuracy** | | | **0.7722** |

---

## Summary

* **Best single model:** LR + SMOTE with p-value feature reduction achieved 93.9% accuracy, improving over the full-feature base model by removing 9 statistically insignificant features.
* **Ensemble approach:** The horizontal split ensemble trained specialist sub-models per class group, achieving 96.2% (M1) and 96.0% (M2) accuracy on their respective subsets.
* **Feature insights:** Color channel features (RR, RG, RB columns) showed the most redundancy, while morphological features alone achieved only 77.2% accuracy, highlighting the importance of shape and color descriptors for fine-grained variety discrimination. DOKOL and SAFAVI are the most consistently well-classified varieties across all model variants.
