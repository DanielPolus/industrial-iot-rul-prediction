# Industrial IoT RUL Prediction  
## Regression models for Remaining Useful Life of industrial machines

This project is a small course work for the **Methods of Machine Learning** class.  
It predicts the **Remaining Useful Life (RUL)** of industrial machines based on a synthetic **Industrial IoT** dataset from Kaggle.

---

## Problem statement

Given telemetry, maintenance history and operating conditions of machines,  
we want to predict:

> **`Remaining_Useful_Life_days`** – how many days the machine can operate before failure.

This is a **regression** task.

---

## Dataset

- Source: Kaggle — *Synthetic Industrial IoT Dataset*  
- Loaded programmatically via `kagglehub` (dataset is **not** stored in the repo).
- Size: 500,000 rows, 22 columns.
- Features include:
  - `Operational_Hours`, `Temperature_C`, `Vibration_mms`, `Sound_dB`
  - `Oil_Level_pct`, `Coolant_Level_pct`, `Power_Consumption_kW`
  - Maintenance and failure history, error codes, AI supervision flags, etc.
- Target: `Remaining_Useful_Life_days`

---

## Models

Baseline and three ML models were trained and compared:

- `DummyRegressor` (baseline, predicts mean RUL)
- `Ridge` (linear regression with L2-regularization)
- `RandomForestRegressor`
- `GradientBoostingRegressor`  ← best model in this project

Models are trained in a unified `sklearn` pipeline with preprocessing.

---

## Preprocessing

- Drop pure identifier: `Machine_ID`
- Convert boolean features → `0/1`:
  - `AI_Supervision`, `Failure_Within_7_Days`
- Numerical features:
  - `SimpleImputer(strategy="median")`
  - `StandardScaler`
- Categorical features:
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`
- Train/test split: **80% / 20%**

---

## Evaluation

Main metrics:

- **RMSE** (Root Mean Squared Error) — primary metric for model selection  
- **MAE** (Mean Absolute Error)  
- **R²** (coefficient of determination)

On the test set (~100k samples), the best model  
(**GradientBoostingRegressor**) achieves:

- RMSE ≈ **47.5 days**
- MAE ≈ **36.9 days**
- R² ≈ **0.973**

The baseline `DummyRegressor` (mean RUL) has RMSE ≈ 288.7 days,  
so ML models significantly improve prediction quality.

---

## Project structure

- `main.py` — full workflow:
  - dataset loading via `kagglehub`
  - exploratory data analysis (EDA)
  - preprocessing pipelines
  - model training + hyperparameter tuning with `GridSearchCV`
  - evaluation and comparison of models
  - simple visualizations (metrics comparison, predicted vs. true RUL, feature importance)

---

## How to run

```bash
# (optional) create and activate virtual env
# python -m venv venv
# source venv/bin/activate         # Linux / macOS
# venv\Scripts\activate            # Windows

pip install -r requirements.txt

python main.py
```

The script will:

Download the dataset from Kaggle via kagglehub

Run preprocessing and train all models

Print metrics to the console

Show basic plots (if a graphical backend is available)
