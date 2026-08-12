# 📊 Financial Distress Early-Warning System

### Machine Learning for Predicting Financial Distress of Vietnamese Listed Companies

> **Early-Warning System using Machine Learning with a Temporal `t-1 → t` Design**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()
[![Domain](https://img.shields.io/badge/Domain-FinTech%20%7C%20Financial%20Risk-6f42c1)]()

---

## 🔎 Overview

This project develops a **Machine Learning-based Early-Warning System** for identifying Vietnamese listed companies that may experience **financial distress in the following year**.

The key methodological idea is to avoid the common mistake of predicting a company's financial distress using financial information from the **same period** in which the distress label is defined.

Instead, the project adopts a temporal prediction framework:

```text
Financial indicators at t-1
          ↓
   Machine Learning
          ↓
Financial distress at t
```

This design better reflects how an actual financial risk monitoring system would operate, where only information available before the prediction period can be used.

The study uses panel data from **533 Vietnamese listed companies on HOSE and HNX during 2020–2025**, resulting in **2,538 observations after preprocessing**.

---

## 🎯 Research Objectives

The project aims to:

* Build a financial distress prediction dataset for Vietnamese listed companies.
* Design an **early-warning prediction framework using `t-1 → t` temporal features**.
* Detect and eliminate potential **Data Leakage** before model training.
* Compare multiple Machine Learning algorithms.
* Evaluate models using metrics suitable for **imbalanced classification**.
* Identify the financial variables that contribute most to distress prediction.
* Explore how the model could support financial risk management.

The research evaluates **seven Machine Learning algorithms** and combines predictive performance with model interpretation.

---

# 🧠 Key Methodological Contribution

## 1. Temporal Prediction: `t-1 → t`

Rather than:

```text
Financial data at t → Distress at t
```

the project uses:

```text
Financial data at t-1 → Distress at t
```

This is critical because the target variable is constructed from financial distress signals observed in year `t`.

The target is defined as:

> `label_distress = 1` when a company has at least **2 of 4 distress signals** in year `t`.

The four signals are:

1. Negative net income
2. Negative operating cash flow
3. Working capital deficit
4. High financial leverage

All predictive features are shifted by one period to ensure that the model only uses information available before the target period.

---

# 🚨 2. Data Leakage Audit

One of the most important parts of the project is the explicit **Data Leakage Audit**.

The experiment compares three scenarios:

| Prediction Design                       |       AUC-ROC |
| --------------------------------------- | ------------: |
| Same-year + all variables               |    **1.0000** |
| Same-year + remove `signal_*` variables |    **0.9910** |
| Temporal `t-1 → t` design               | **0.79–0.87** |

The near-perfect performance under same-year prediction was identified as a warning sign rather than a success.

Even after removing the direct `signal_*` variables, the remaining same-year financial indicators could still reproduce the rule used to construct the target.

Therefore, the final models were trained using the **lagged `t-1 → t` dataset**.

### Why this matters

A model with an AUC close to 1.0 is not necessarily a better financial prediction model.

In this project:

```text
AUC ≈ 1.00
      ↓
Potential information leakage
      ↓
Unrealistically optimistic evaluation
```

while:

```text
AUC ≈ 0.85
      ↓
Harder prediction problem
      ↓
More realistic early-warning setting
```

This distinction is particularly important for financial risk applications.

---

# 📁 Dataset

## Data Sources

The dataset was manually collected and cross-checked from:

* **CafeF**
* **Vietstock Finance**

The dataset covers companies listed on:

* HOSE
* HNX

for the period:

```text
2020 – 2025
```

The research dataset integrates several groups of financial information, including:

* Firm Life Cycle indicators
* Fundamental financial indicators
* CAMEL-related ratios
* Beneish M-Score components

The final clean panel contains:

```text
533 companies
2,538 observations
41 columns
2020–2025
~14.7% positive distress observations
```

The class distribution is therefore highly imbalanced, making metrics such as Recall, F1, ROC-AUC and PR-AUC more informative than Accuracy alone.

> **Data note:** The raw Excel dataset is not included in this repository by default. Place `Data - ML.xlsx` in the expected data directory before running the notebook.

---

# 🛠️ Tech Stack

### Programming & Environment

* Python 3.12
* Jupyter Notebook
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Data Visualization

* Matplotlib
* Seaborn

### Statistical & Model Analysis

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-score
* Confusion Matrix
* Permutation Importance
* GroupKFold Cross-Validation
* McNemar Test
* Threshold Tuning

The notebook implements the complete workflow from data loading and EDA to model comparison, threshold tuning, feature importance and dashboard generation.

---

# 🤖 Machine Learning Models

The project compares seven classification algorithms:

| Category       | Model                  |
| -------------- | ---------------------- |
| Linear         | Logistic Regression    |
| Tree           | Decision Tree          |
| Ensemble       | Random Forest          |
| Boosting       | XGBoost                |
| Distance-based | K-Nearest Neighbors    |
| Kernel-based   | Support Vector Machine |
| Probabilistic  | Gaussian Naive Bayes   |

All models are trained using the same temporal Train/Test framework:

```text
Train: 2021–2023
Test : 2024–2025
```

with:

```text
Train = 1,504 observations
Test  = 1,034 observations
```

This time-based split avoids using future observations to predict the past.

---

# ⚙️ Data Preprocessing Pipeline

The preprocessing pipeline includes:

```text
Raw Financial Data
        │
        ▼
Data Cleaning
        │
        ▼
Data Leakage Audit
        │
        ▼
Remove Leakage / Redundant Features
        │
        ▼
Lag Feature Engineering
        │
        ▼
EDA
        │
        ▼
Time-based Train/Test Split
        │
        ├───────────────┐
        ▼               ▼
Scaled Branch       Tree Branch
        │               │
        ▼               ▼
Logistic / KNN / SVM   DT / RF / XGBoost
        │               │
        └───────┬───────┘
                ▼
        Model Evaluation
                │
                ▼
       Model Interpretation
```

### Main preprocessing techniques

* Lag-1 feature engineering
* Missing-value imputation using training-set median
* One-Hot Encoding
* Signed `log1p` transformation
* StandardScaler
* Class weighting
* `scale_pos_weight` for XGBoost
* Time-based train/test split
* GroupKFold validation by company ticker

For example, the signed log transformation is defined as:

```text
x' = sign(x) × log(1 + |x|)
```

and is applied to models sensitive to feature scale.

---

# 📊 Model Performance

Performance was evaluated on the independent Test set.

| Model                |    ROC-AUC |     PR-AUC | Accuracy | Precision |     Recall |         F1 |
| -------------------- | ---------: | ---------: | -------: | --------: | ---------: | ---------: |
| **Random Forest**    | **0.8690** |     0.4309 |   0.8617 |    0.4699 |     0.5865 |     0.5217 |
| Logistic Regression  |     0.8660 | **0.4450** |   0.7544 |    0.3256 | **0.8496** |     0.4708 |
| SVM (RBF)            |     0.8619 |     0.4158 |   0.7485 |    0.3116 |     0.7895 |     0.4468 |
| XGBoost              |     0.8530 |     0.4078 |   0.8569 |    0.4599 |     0.6466 | **0.5375** |
| Decision Tree        |     0.8434 |     0.3972 |   0.7863 |    0.3523 |     0.7895 |          — |
| KNN                  |     0.8267 |     0.3678 |   0.8704 |    0.3333 |     0.0075 |     0.0147 |
| Gaussian Naive Bayes |     0.7982 |     0.3555 |   0.7756 |    0.3146 |     0.6316 |     0.4200 |

The results show that **Random Forest achieved the highest ROC-AUC (0.8690)**, while **Logistic Regression achieved the highest PR-AUC (0.4450)** and **XGBoost achieved the highest F1-score (0.5375)** among the reported model results.

### Key observation

Accuracy alone would be misleading in this project.

For example, KNN achieved:

```text
Accuracy = 87.04%
Recall   = 0.75%
```

which means it correctly classified many non-distressed companies but almost completely failed to identify the minority distress class.

This demonstrates why **Recall, PR-AUC and F1-score** are particularly important for an early-warning system.

---

# 🔬 Model Stability

The project additionally applies **5-fold GroupKFold Cross-Validation** using company ticker as the grouping variable.

```text
Mean AUC-ROC = 0.8363
Std. Dev.    = 0.0288
```

Using ticker-level grouping prevents observations from the same company from appearing simultaneously in training and validation folds.

This provides an additional check that the model is not simply memorizing company-specific characteristics.

---

# 🔍 Model Interpretation

The project does not stop at model performance.

Several interpretability techniques are implemented.

## Permutation Importance

Permutation Importance is used to identify the financial variables that contribute most to the Random Forest's predictive performance.

The analysis indicates that **financial leverage**, particularly `DebtToAssets_lag1`, plays an important role in the prediction of financial distress.

This is economically intuitive because higher leverage increases the financial burden and vulnerability of firms under adverse operating conditions.

---

## 🎚️ Threshold Tuning

The default classification threshold:

```text
0.50
```

is not necessarily optimal for an early-warning system.

The project therefore evaluates the Precision–Recall trade-off and searches for an F1-optimizing threshold.

For XGBoost:

```text
Default threshold = 0.500
Optimized threshold ≈ 0.507
```

The optimized threshold slightly improves Precision while maintaining Recall at approximately the same level.

---

# 🧩 Error Analysis

The project also examines the **Confusion Matrix** of the XGBoost model.

At the selected threshold, the estimated confusion matrix is approximately:

|                         | Predicted Non-Distress | Predicted Distress |
| ----------------------- | ---------------------: | -----------------: |
| **Actual Non-Distress** |                    802 |                 99 |
| **Actual Distress**     |                     47 |                 86 |

The analysis highlights an important limitation:

> Some financially distressed companies may not yet show sufficiently strong leverage signals one year before distress occurs.

This motivates future feature engineering based on **multi-year financial trends**, rather than relying only on a single `t-1` snapshot.

---

# 📈 Visualizations

The project generates several analytical visualizations, including:

* Target-class distribution
* Financial feature distributions
* Correlation analysis
* Model performance comparison
* ROC Curves
* Precision–Recall Curves
* Permutation Importance
* Precision–Recall vs. classification threshold
* Confusion Matrix
* Model comparison dashboard

The notebook also generates an **HTML dashboard** summarizing model performance and error analysis.

---

# 📂 Project Structure

A recommended GitHub repository structure is:

```text
financial-distress-early-warning/
│
├── README.md
│
├── notebooks/
│   └── financial_distress_prediction.ipynb
│
├── data/
│   └── Data - ML.xlsx
│
├── outputs/
│   ├── confusion_matrix_xgboost.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── dashboard.html
│
├── report/
│   └── financial_distress_report.pdf
│
├── requirements.txt
│
└── .gitignore
```

> **Privacy / data consideration:** If the raw dataset cannot be redistributed, keep `Data - ML.xlsx` outside the public repository and provide only the data schema, sample data or instructions for reproducing the dataset.

---

# 🚀 How to Run

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/financial-distress-early-warning.git
cd financial-distress-early-warning
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Add the dataset

Place:

```text
Data - ML.xlsx
```

in the directory expected by the notebook.

## 5. Launch Jupyter

```bash
jupyter notebook
```

Open the notebook and run the cells sequentially.

---

# 📦 Requirements

A minimal `requirements.txt` can include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
openpyxl
jupyter
```

---

# 💡 Practical Applications

The proposed system is intended as a **decision-support tool**, not as a standalone financial decision maker.

### 🏦 Banks & Credit Institutions

Potential applications include:

* Corporate credit screening
* Early-warning monitoring
* Post-loan risk surveillance
* Customer risk segmentation

### 📈 Investors

The model can provide an additional quantitative signal for:

* Corporate risk screening
* Portfolio monitoring
* Identifying companies with deteriorating financial health

### 🏢 Corporations

Companies can potentially use the framework as an internal financial health monitoring tool to identify warning signs before severe financial distress occurs.

The report specifically highlights leverage indicators such as `DebtToAssets` as important signals for financial risk monitoring.

---

# ⚠️ Limitations

Several limitations should be considered before interpreting the results as a deployable production system:

* The dataset covers a limited period of **2020–2025**.
* The study focuses on listed companies on **HOSE and HNX**.
* The target variable is constructed from four financial distress signals and may not capture every form of financial distress.
* The model primarily uses financial information from `t-1`, which may miss longer-term deterioration patterns.
* The class distribution is imbalanced.
* Some important macroeconomic and industry-level variables are not incorporated.
* Model performance on historical data does not guarantee equivalent performance in future market conditions.

The research therefore positions the model as an **early-warning support system**, rather than a definitive bankruptcy prediction mechanism.

---

# 🔮 Future Improvements

Potential directions include:

### 1. Multi-year temporal features

Instead of using only:

```text
X(t-1)
```

incorporate:

```text
X(t-1)
X(t-2)
X(t-3)
ΔX
Growth Rate
Trend
Volatility
```

This could help detect companies whose financial health deteriorates gradually before distress becomes visible through leverage alone.

### 2. Macroeconomic variables

Future versions could incorporate:

* GDP growth
* Interest rates
* Inflation
* Exchange rates
* Market volatility
* Industry-level indicators

### 3. More advanced explainability

Potential extensions:

* SHAP
* Partial Dependence
* Individual Conditional Expectation
* Local explanations

### 4. Production deployment

The research prototype could be extended into:

```text
Financial Data
      ↓
Data Pipeline
      ↓
Feature Engineering
      ↓
ML Model
      ↓
Risk Probability
      ↓
Early-Warning Dashboard
      ↓
Risk Monitoring
```

---

# 📚 References

Key methodological references include:

* Altman, E. I. (1968). *Financial ratios, discriminant analysis and the prediction of corporate bankruptcy.*
* Ohlson, J. A. (1980). *Financial ratios and the probabilistic prediction of bankruptcy.*
* Breiman, L. (2001). *Random Forests.*
* Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.*
* Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.*
* Dickinson, V. (2011). *Cash flow patterns as a proxy for firm life cycle.*

The report also documents CafeF and Vietstock Finance as the primary financial data sources.

---

# 👨‍💻 Author

**Trần Nguyễn Ánh Tú**

Faculty of Finance & Banking
University of Economics and Law (UEL)

**Course:** Machine Learning and Artificial Intelligence in Finance
**Instructor:** ThS. Phan Huy Tâm

---

## ⭐ Key Takeaways

```text
✓ 533 Vietnamese listed companies
✓ 2,538 clean panel observations
✓ 2020–2025 financial data
✓ Temporal prediction: t-1 → t
✓ Explicit Data Leakage Audit
✓ 7 Machine Learning algorithms
✓ Time-based Train/Test split
✓ GroupKFold validation by company
✓ ROC-AUC / PR-AUC / Precision / Recall / F1
✓ Permutation Importance
✓ Threshold Tuning
✓ Confusion Matrix & Error Analysis
✓ HTML analytical dashboard
```

> **The core lesson of this project:**
> A high-performing financial distress model is not necessarily a useful early-warning model.
> **Preventing information leakage and respecting the temporal structure of financial data are just as important as maximizing predictive accuracy.**

---

<p align="center">
  <b>Financial Risk × Machine Learning × Early Warning</b>
  <br>
  Built with Python
</p>
