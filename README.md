# Predicting Student Depression Using Decision Trees

**Author:** Andile Brian Sithole 
---

## 📌 Overview

An end-to-end machine learning project that predicts **depression in
university students** from demographic, academic, and mental-health features
using an **interpretable Decision Tree classifier**. The pipeline includes
exploratory data analysis, leakage-safe preprocessing with a
`ColumnTransformer`, hyper-parameter tuning via `GridSearchCV`, and
**cost-complexity pruning (`ccp_alpha`)** to balance accuracy and
interpretability.

The model achieves **80.8% test accuracy** with **100% recall on the
no-depression class** — a strong baseline for a high-stakes mental-health
screening context.

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **80.8 %** |
| Mean CV Accuracy (5-fold) | 78.7 % |
| Recall (No Depression) | 1.00 |
| Recall (Depression) | 0.44 |
| Precision (Depression) | 1.00 |

Confusion matrix (26 test samples):

|                     | Predicted No | Predicted Yes |
|---------------------|--------------|---------------|
| **Actual No (17)**  | 17           | 0             |
| **Actual Yes (9)**  | 5            | 4             |

> **Why this matters:** In mental-health screening, high recall on
> non-depressed students means healthy students are rarely mis-labelled —
> reducing unnecessary interventions. The trade-off (moderate recall on
> the depressed class) is discussed in the report.

---

## 📊 Dataset

- **Source:** `data/student_mental_health.csv` — 101 survey responses
- **Target:** `Do you have Depression?` (Yes / No)
- **Class balance:** ~65 % No · ~35 % Yes

| Feature | Type | Description |
|---|---|---|
| `Choose your gender` | Categorical | Female / Male |
| `Age` | Continuous | Student's age (18–24) |
| `What is your course?` | Categorical | 48 unique courses |
| `Your current year of Study` | Categorical | year 1 – year 4 |
| `What is your CGPA?` | Categorical (ordinal) | CGPA bracket |
| `Marital status` | Binary | Yes / No |
| `Do you have Depression?` | Binary | **Target variable** |
| `Do you have Anxiety?` | Binary | Self-reported |
| `Do you have Panic attack?` | Binary | Self-reported |
| `Did you seek any specialist for a treatment?` | Binary | Self-reported |

---

## 🔬 Methodology

### 1. Data Pre-processing
- **Dropped the `Timestamp`** column — not predictive
- **Missing values:** median for `Age`, most-frequent for categoricals
- **One-hot encoding** for all categorical features
- **Stratified 75/25 train-test split** — preserves class balance
- All preprocessing wrapped in a **`ColumnTransformer` + `Pipeline`** to
  prevent data leakage

### 2. Exploratory Data Analysis
- Class balance check
- Correlation between anxiety, panic attacks, and depression
- Distribution of depression by gender and academic features

### 3. Model Training
- **Algorithm:** `DecisionTreeClassifier(criterion='entropy')`
- **Hyper-parameter tuning:** `GridSearchCV` with **5-fold cross-validation**
  over:
  - `max_depth` ∈ {3, 4, 5, 6, 8, None}
  - `min_samples_split` ∈ {2, 5, 10, 20}
  - `min_samples_leaf` ∈ {1, 2, 5, 10}
  - `ccp_alpha` ∈ {0, 0.001, 0.005, 0.01} (cost-complexity pruning)

### 4. Evaluation
- Accuracy, classification report (precision / recall / F1)
- Confusion matrix
- **Feature importance** ranking
- **Depth-3 tree visualisation** for interpretability

---

## 📈 Feature Importance

Top predictors of student depression:

| Rank | Feature | Insight |
|---|---|---|
| 1 | **Anxiety status** | Strongest predictor (~38% importance) — aligns with clinical co-morbidity literature |
| 2 | **Panic attacks** | Closely coupled with anxiety |
| 3 | **CGPA** | Academic performance correlates with mental health |
| 4 | Course / year of study | Weak but non-zero signal |
| 5 | Gender | **Zero importance** — depression affects all genders similarly |
