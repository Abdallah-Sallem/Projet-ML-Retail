from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_data, save_data

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_TRAIN_TEST_DIR = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _infer_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "Churn",
        "churn",
        "CustomerChurn",
        "customer_churn",
        "IsChurn",
        "is_churn",
        "Exited",
        "Target",
        "target",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError("Target column could not be inferred from processed dataset.")


def load_processed_train_test() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """Load train/test matrices produced by preprocessing, with fallback split logic."""
    x_train_file = DATA_TRAIN_TEST_DIR / "X_train_processed.csv"
    x_test_file = DATA_TRAIN_TEST_DIR / "X_test_processed.csv"
    y_train_file = DATA_TRAIN_TEST_DIR / "y_train.csv"
    y_test_file = DATA_TRAIN_TEST_DIR / "y_test.csv"

    if all(path.exists() for path in [x_train_file, x_test_file, y_train_file, y_test_file]):
        x_train = load_data(x_train_file)
        x_test = load_data(x_test_file)
        y_train_df = load_data(y_train_file)
        y_test_df = load_data(y_test_file)
        target_col = y_train_df.columns[0]
        y_train = y_train_df.iloc[:, 0]
        y_test = y_test_df.iloc[:, 0]
        return x_train, x_test, y_train, y_test, target_col

    processed_file = DATA_PROCESSED_DIR / "retail_processed.csv"
    if not processed_file.exists():
        raise FileNotFoundError(
            "Processed files not found. Run src/preprocessing.py before training models."
        )

    processed_df = load_data(processed_file)
    target_col = _infer_target_column(processed_df)

    if "split" in processed_df.columns:
        train_df = processed_df[processed_df["split"].str.lower() == "train"].copy()
        test_df = processed_df[processed_df["split"].str.lower() == "test"].copy()

        x_train = train_df.drop(columns=[target_col, "split"], errors="ignore")
        x_test = test_df.drop(columns=[target_col, "split"], errors="ignore")
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        return x_train, x_test, y_train, y_test, target_col

    x = processed_df.drop(columns=[target_col], errors="ignore")
    y = processed_df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test, target_col


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    return numeric_df.fillna(0.0)


def _get_probabilities(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))

    predictions = model.predict(x)
    return predictions.astype(float)


def evaluate_classifier(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(x_test)
    y_prob = _get_probabilities(model, x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    metrics["classification_report"] = classification_report(y_test, y_pred, zero_division=0)
    return metrics


def get_model_search_spaces() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {
                "pca__n_components": [0.9, 0.95, 0.99],
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
            {
                "pca__n_components": [0.9, 0.95],
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "pca__n_components": [0.9, 0.95, 0.99],
                "model__n_neighbors": [5, 11, 21],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "pca__n_components": [0.9, 0.95],
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def train_and_select_best_model() -> Dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, target_col = load_processed_train_test()
    x_train = _ensure_numeric(x_train)
    x_test = _ensure_numeric(x_test)
    y_train = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)
    y_test = pd.to_numeric(y_test, errors="coerce").fillna(0).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search_spaces = get_model_search_spaces()

    all_results: list[Dict[str, Any]] = []
    fitted_searches: Dict[str, GridSearchCV] = {}

    for model_name, (estimator, param_grid) in search_spaces.items():
        pipeline = Pipeline(
            steps=[
                ("pca", PCA(random_state=42)),
                ("model", estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(x_train, y_train)
        fitted_searches[model_name] = search

        metrics = evaluate_classifier(search.best_estimator_, x_test, y_test)
        all_results.append(
            {
                "model_name": model_name,
                "cv_best_score": float(search.best_score_),
                "best_params": search.best_params_,
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1_score"],
                "test_roc_auc": metrics["roc_auc"],
                "classification_report": metrics["classification_report"],
            }
        )

    results_df = pd.DataFrame(all_results).sort_values(
        by=["test_roc_auc", "cv_best_score"], ascending=False
    )
    save_data(results_df, REPORTS_DIR / "model_comparison.csv", index=False)

    best_row = results_df.iloc[0]
    best_model_name = str(best_row["model_name"])
    best_search = fitted_searches[best_model_name]
    best_estimator = best_search.best_estimator_

    joblib.dump(best_estimator, MODELS_DIR / "best_model.joblib")

    with (REPORTS_DIR / "best_model_classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(str(best_row["classification_report"]))

    training_summary = {
        "target_column": target_col,
        "x_train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "x_test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "best_model_name": best_model_name,
        "best_cv_roc_auc": float(best_row["cv_best_score"]),
        "best_test_metrics": {
            "accuracy": float(best_row["test_accuracy"]),
            "precision": float(best_row["test_precision"]),
            "recall": float(best_row["test_recall"]),
            "f1": float(best_row["test_f1"]),
            "roc_auc": float(best_row["test_roc_auc"]),
        },
        "best_params": best_search.best_params_,
    }

    with (REPORTS_DIR / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    return training_summary


def main() -> None:
    summary = train_and_select_best_model()
    print("Training completed successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
