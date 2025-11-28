import kagglehub
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
CV_SUBSAMPLE_SIZE = 50000

path = kagglehub.dataset_download("canozensoy/industrial-iot-dataset-synthetic")
data_dir = Path(path)

csv_files = list(data_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("У директорії з датасетом не знайдено жодного CSV-файлу.")
csv_path = csv_files[0]

df = pd.read_csv(csv_path)

print("=== Базова інформація про датасет ===")
print("Перші рядки датасету:")
print(df.head())
print("\nРозмір датасету (рядки, стовпчики):", df.shape)
print("\nТипи даних:")
print(df.dtypes)
print("\nОписова статистика для числових ознак:")
print(df.describe().T)
print("\nКількість пропущених значень по стовпцях:")
print(df.isna().sum())

print("\n=== Побудова гістограм числових ознак ===")
numeric_cols_all = df.select_dtypes(include=[np.number]).columns
df[numeric_cols_all].hist(figsize=(16, 10), bins=30)
plt.tight_layout()
plt.show()

print("\n=== Кореляційна матриця числових ознак ===")
cols = ["Remaining_Useful_Life_days",
        "Operational_Hours",
        "Last_Maintenance_Days_Ago",
        "Maintenance_History_Count",
        "Failure_History_Count"]

plt.figure(figsize=(6, 4))
sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Кореляція RUL з основними ознаками")
plt.tight_layout()
plt.show()

TARGET_COL = "Remaining_Useful_Life_days"

if TARGET_COL not in df.columns:
    raise KeyError(
        f"Стовпця '{TARGET_COL}' не знайдено в датасеті. "
        f"Перевірте назву цільової ознаки та змініть змінну TARGET_COL."
    )

df = df.drop(columns=["Machine_ID"])

df["AI_Supervision"] = df["AI_Supervision"].astype(int)
df["Failure_Within_7_Days"] = df["Failure_Within_7_Days"].astype(int)

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

print("\n=== Формування матриці ознак та цілі ===")
print("Форма X:", X.shape)
print("Форма y:", y.shape)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nЧислові ознаки:", numeric_features)
print("Категоріальні ознаки:", categorical_features)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("\n=== Поділ на train/test ===")
print("Розмір train:", X_train.shape, "Розмір test:", X_test.shape)

print("\n=== Базова модель: DummyRegressor (прогнозує середнє значення RUL) ===")

dummy_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DummyRegressor(strategy="mean")),
    ]
)

dummy_pipe.fit(X_train, y_train)
y_pred_dummy = dummy_pipe.predict(X_test)

dummy_mae = mean_absolute_error(y_test, y_pred_dummy)
dummy_mse = mean_squared_error(y_test, y_pred_dummy)
dummy_rmse = np.sqrt(dummy_mse)
dummy_r2 = r2_score(y_test, y_pred_dummy)

print(f"Dummy Test MAE:  {dummy_mae:.4f}")
print(f"Dummy Test RMSE: {dummy_rmse:.4f}")
print(f"Dummy Test R^2:  {dummy_r2:.4f}")

models_and_params = {
    "Ridge": (
        Ridge(),
        {
            "model__alpha": [0.1, 1.0, 10.0],
        },
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        {
            "model__n_estimators": [100],
            "model__max_depth": [None, 20],
            "model__min_samples_split": [2, 5],
        },
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3],
        },
    ),
}

results = []

for name, (model, param_grid) in models_and_params.items():
    print(f"\n=== Модель: {name} ===")

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    if name in ["RandomForest", "GradientBoosting"]:
        train_size = min(CV_SUBSAMPLE_SIZE, X_train.shape[0])
        X_for_cv, _, y_for_cv, _ = train_test_split(
            X_train,
            y_train,
            train_size=train_size,
            random_state=RANDOM_STATE,
        )
        print(f"GridSearch для {name} виконується на підвибірці з {train_size} об'єктів.")
    else:
        X_for_cv, y_for_cv = X_train, y_train
        print(f"GridSearch для {name} виконується на повному train.")

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        verbose=1,
    )

    grid.fit(X_for_cv, y_for_cv)

    print("Найкращі гіперпараметри:", grid.best_params_)
    print("Найкращий CV RMSE:", -grid.best_score_)

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R^2:  {r2:.4f}")

    results.append(
        {
            "model": name,
            "best_params": grid.best_params_,
            "cv_rmse": -grid.best_score_,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
            "best_estimator": best_model,
        }
    )

print("\n=== Порівняння моделей (включаючи базову) ===")

summary_rows = [
    {
        "model": "Dummy (mean)",
        "cv_rmse": np.nan,
        "test_mae": dummy_mae,
        "test_rmse": dummy_rmse,
        "test_r2": dummy_r2,
    }
] + [
    {
        "model": r["model"],
        "cv_rmse": r["cv_rmse"],
        "test_mae": r["test_mae"],
        "test_rmse": r["test_rmse"],
        "test_r2": r["test_r2"],
    }
    for r in results
]

results_df = pd.DataFrame(summary_rows)
print(results_df)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.barplot(data=results_df, x="model", y="test_rmse")
plt.title("Порівняння моделей за RMSE")
plt.xticks(rotation=15)
plt.tight_layout()

plt.subplot(1, 2, 2)
sns.barplot(data=results_df, x="model", y="test_r2")
plt.title("Порівняння моделей за R^2")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

best_by_rmse = min(results, key=lambda r: r["test_rmse"])
best_model_name = best_by_rmse["model"]
best_estimator = best_by_rmse["best_estimator"]

print(f"\nНайкраща модель за RMSE (без урахування Dummy): {best_model_name}")

y_pred_best = best_estimator.predict(X_test)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_best, alpha=0.3)
max_val = max(y_test.max(), y_pred_best.max())
min_val = min(y_test.min(), y_pred_best.min())
plt.plot([min_val, max_val], [min_val, max_val], "r--")
plt.xlabel("Справжнє значення RUL")
plt.ylabel("Прогнозоване значення RUL")
plt.title(f"Фактичні vs. прогнозовані значення для моделі {best_model_name}")
plt.tight_layout()
plt.show()

if hasattr(best_estimator.named_steps["model"], "feature_importances_"):
    importances = best_estimator.named_steps["model"].feature_importances_
    feature_names = best_estimator.named_steps["preprocessor"].get_feature_names_out()

    fi = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("\nТоп-10 найважливіших ознак для найкращої моделі:")
    print(fi.head(10))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=fi.head(10), x="importance", y="feature")
    plt.title(f"Топ-10 важливих ознак для моделі {best_model_name}")
    plt.tight_layout()
    plt.show()
else:
    print(
        f"\nМодель {best_model_name} не має атрибуту feature_importances_. "
        "Аналіз важливості ознак не виконується."
    )
