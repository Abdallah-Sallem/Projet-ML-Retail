from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import detect_missing_values, load_data, outlier_detection, save_data

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_TRAIN_TEST_DIR = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_RAW_FILE = DATA_RAW_DIR / "retail_customers_COMPLETE_CATEGORICAL.csv"


def infer_target_column(df: pd.DataFrame) -> str:
    """Infer churn target column from common naming conventions."""
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
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Target column not found automatically. Pass --target-col with the churn label column name."
    )


def _robust_parse_datetime(series: pd.Series) -> pd.Series:
    """Try multiple date parsing strategies for inconsistent date formats."""
    parsed_default = pd.to_datetime(series, errors="coerce", dayfirst=False, format="mixed")
    parsed_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    return parsed_default.fillna(parsed_dayfirst)


def _extract_ip_tuple(value: Any) -> Tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan

    raw_value = str(value).strip()
    if not raw_value:
        return np.nan, np.nan

    try:
        parsed_ip = ipaddress.ip_address(raw_value)
        first_octet = float(raw_value.split(".")[0]) if parsed_ip.version == 4 else np.nan
        is_private = float(int(parsed_ip.is_private))
        return first_octet, is_private
    except ValueError:
        return np.nan, np.nan


def _extract_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic numeric features from IP-like columns."""
    transformed = df.copy()
    ip_columns: List[str] = []
    for col in transformed.columns:
        normalized = col.lower().replace("_", "")
        if normalized.endswith("ip") or "ipaddress" in normalized or normalized == "ip":
            ip_columns.append(col)

    for col in ip_columns:
        parsed = transformed[col].apply(_extract_ip_tuple)
        transformed[f"{col}_first_octet"] = parsed.apply(lambda item: item[0])
        transformed[f"{col}_is_private"] = parsed.apply(lambda item: item[1])
        transformed.drop(columns=[col], inplace=True)

    return transformed


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning and feature engineering on raw input data."""
    cleaned = df.copy()

    # Drop explicitly irrelevant or leakage-prone columns when present.
    drop_candidates = [
        "Newsletter",
        "newsletter",
        "ChurnRiskCategory",
        "churn_risk_category",
    ]
    to_drop = [col for col in drop_candidates if col in cleaned.columns]
    if to_drop:
        cleaned.drop(columns=to_drop, inplace=True)

    if "RegistrationDate" in cleaned.columns:
        registration = _robust_parse_datetime(cleaned["RegistrationDate"])
        cleaned["RegistrationYear"] = registration.dt.year
        cleaned["RegistrationMonth"] = registration.dt.month
        cleaned["RegistrationDayOfWeek"] = registration.dt.dayofweek

        snapshot = registration.max()
        if pd.notna(snapshot):
            cleaned["TenureDays"] = (snapshot - registration).dt.days

        cleaned.drop(columns=["RegistrationDate"], inplace=True)

    object_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].astype("string").str.strip().replace({"": pd.NA})

    cleaned = _extract_ip_features(cleaned)
    return cleaned


def _encode_target(y: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """Encode target into binary labels whenever possible."""
    if y.dtype == bool:
        return y.astype(int), {"False": 0, "True": 1}

    if pd.api.types.is_numeric_dtype(y):
        y_no_na = y.fillna(y.median())
        unique_vals = sorted(pd.unique(y_no_na))
        if len(unique_vals) == 2:
            mapping = {str(unique_vals[0]): 0, str(unique_vals[1]): 1}
            encoded = y_no_na.map({unique_vals[0]: 0, unique_vals[1]: 1}).astype(int)
            return encoded, mapping
        return y_no_na.astype(int), {str(val): int(val) for val in unique_vals}

    normalized = y.astype("string").str.strip().str.lower()
    explicit_mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "churn": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
        "not churn": 0,
    }

    mapped = normalized.map(explicit_mapping)
    if mapped.notna().mean() >= 0.9 and mapped.dropna().nunique() <= 2:
        return mapped.fillna(0).astype(int), {"0": 0, "1": 1}

    filled = y.astype("string").fillna("missing")
    encoded, uniques = pd.factorize(filled)
    mapping = {str(label): int(idx) for idx, label in enumerate(uniques)}
    encoded_series = pd.Series(encoded, index=y.index, name=y.name)
    return encoded_series, mapping


def build_preprocessing_pipeline(
    x_train: pd.DataFrame,
    ordinal_columns: Optional[Sequence[str]] = None,
    knn_missing_threshold: float = 0.3,
) -> Tuple[Pipeline, Dict[str, List[str]]]:
    """Build train-only preprocessing steps for imputation, encoding, and scaling."""
    numeric_cols = x_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in x_train.columns if col not in numeric_cols]

    ordinal_columns = [col for col in (ordinal_columns or []) if col in categorical_cols]
    nominal_columns = [col for col in categorical_cols if col not in ordinal_columns]

    numeric_missing = x_train[numeric_cols].isna().mean() if numeric_cols else pd.Series(dtype=float)
    numeric_skew = x_train[numeric_cols].apply(lambda s: s.dropna().skew()).abs() if numeric_cols else pd.Series(dtype=float)

    knn_cols = [col for col in numeric_cols if numeric_missing.get(col, 0.0) > knn_missing_threshold]
    remaining_numeric = [col for col in numeric_cols if col not in knn_cols]
    median_cols = [col for col in remaining_numeric if numeric_skew.get(col, 0.0) > 1.0]
    mean_cols = [col for col in remaining_numeric if col not in median_cols]

    transformers = []

    if mean_cols:
        transformers.append(
            (
                "num_mean",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))]),
                mean_cols,
            )
        )

    if median_cols:
        transformers.append(
            (
                "num_median",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                median_cols,
            )
        )

    if knn_cols:
        transformers.append(
            (
                "num_knn",
                Pipeline(steps=[("imputer", KNNImputer(n_neighbors=5))]),
                knn_cols,
            )
        )

    if nominal_columns:
        transformers.append(
            (
                "cat_nominal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                nominal_columns,
            )
        )

    if ordinal_columns:
        categories = [
            sorted(x_train[col].dropna().astype(str).unique().tolist())
            if x_train[col].notna().any()
            else ["missing"]
            for col in ordinal_columns
        ]
        transformers.append(
            (
                "cat_ordinal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                categories=categories,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ordinal_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
        ]
    )

    summary = {
        "mean_imputed_numeric": mean_cols,
        "median_imputed_numeric": median_cols,
        "knn_imputed_numeric": knn_cols,
        "nominal_encoded": nominal_columns,
        "ordinal_encoded": ordinal_columns,
    }

    return pipeline, summary


def preprocess_and_save(
    raw_file_path: Path,
    target_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    ordinal_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run full preprocessing and persist artifacts to disk."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_data(raw_file_path)

    missing_report = detect_missing_values(raw_df)
    save_data(missing_report, REPORTS_DIR / "missing_values_report.csv", index=False)

    cleaned_df = clean_raw_data(raw_df)

    outlier_report, _ = outlier_detection(cleaned_df)
    save_data(outlier_report, REPORTS_DIR / "outlier_report.csv", index=False)

    if target_col is None:
        target_col = infer_target_column(cleaned_df)

    if target_col not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_col}' not found after cleaning.")

    y_raw = cleaned_df[target_col]
    x_raw = cleaned_df.drop(columns=[target_col])

    y_encoded, target_mapping = _encode_target(y_raw)

    stratify_labels = y_encoded if y_encoded.nunique() <= 10 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_raw,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    preprocessing_pipeline, preprocessing_summary = build_preprocessing_pipeline(
        x_train,
        ordinal_columns=ordinal_columns,
    )

    x_train_processed = preprocessing_pipeline.fit_transform(x_train)
    x_test_processed = preprocessing_pipeline.transform(x_test)

    feature_names = preprocessing_pipeline.named_steps["preprocessor"].get_feature_names_out()

    x_train_df = pd.DataFrame(x_train_processed, columns=feature_names, index=x_train.index)
    x_test_df = pd.DataFrame(x_test_processed, columns=feature_names, index=x_test.index)

    save_data(x_train_df, DATA_TRAIN_TEST_DIR / "X_train_processed.csv", index=False)
    save_data(x_test_df, DATA_TRAIN_TEST_DIR / "X_test_processed.csv", index=False)
    save_data(pd.DataFrame({target_col: y_train}), DATA_TRAIN_TEST_DIR / "y_train.csv", index=False)
    save_data(pd.DataFrame({target_col: y_test}), DATA_TRAIN_TEST_DIR / "y_test.csv", index=False)

    train_bundle = x_train_df.copy()
    train_bundle[target_col] = y_train.values
    train_bundle["split"] = "train"

    test_bundle = x_test_df.copy()
    test_bundle[target_col] = y_test.values
    test_bundle["split"] = "test"

    processed_df = pd.concat([train_bundle, test_bundle], axis=0).reset_index(drop=True)
    save_data(processed_df, DATA_PROCESSED_DIR / "retail_processed.csv", index=False)

    preprocessing_artifact = {
        "pipeline": preprocessing_pipeline,
        "feature_columns": x_train.columns.tolist(),
        "target_column": target_col,
        "target_mapping": target_mapping,
        "generated_features": feature_names.tolist(),
        "preprocessing_summary": preprocessing_summary,
    }
    joblib.dump(preprocessing_artifact, MODELS_DIR / "preprocessing_pipeline.joblib")

    summary = {
        "raw_file": str(raw_file_path),
        "rows_raw": int(raw_df.shape[0]),
        "columns_raw": int(raw_df.shape[1]),
        "rows_cleaned": int(cleaned_df.shape[0]),
        "columns_cleaned": int(cleaned_df.shape[1]),
        "target_column": target_col,
        "target_mapping": target_mapping,
        "x_train_shape": [int(x_train_df.shape[0]), int(x_train_df.shape[1])],
        "x_test_shape": [int(x_test_df.shape[0]), int(x_test_df.shape[1])],
        "preprocessing_summary": preprocessing_summary,
    }

    with (REPORTS_DIR / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess retail customer dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_RAW_FILE),
        help="Path to input raw CSV dataset.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Name of churn target column. Auto-inferred when omitted.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved for test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--ordinal-cols",
        type=str,
        default="",
        help="Comma-separated list of ordinal categorical columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ordinal_columns = [col.strip() for col in args.ordinal_cols.split(",") if col.strip()]

    summary = preprocess_and_save(
        raw_file_path=Path(args.input),
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
        ordinal_columns=ordinal_columns,
    )

    print("Preprocessing completed successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
