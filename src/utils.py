from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def load_data(file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a dataset from disk (CSV or Parquet)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)

    raise ValueError(f"Unsupported file format: {suffix}")


def save_data(df: pd.DataFrame, file_path: str | Path, index: bool = False, **kwargs: Any) -> None:
    """Save a DataFrame to disk (CSV or Parquet)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index, **kwargs)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=index, **kwargs)
        return

    raise ValueError(f"Unsupported file format: {suffix}")


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of missing values by column."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / max(len(df), 1)) * 100
    summary = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
            "dtype": [str(df[col].dtype) for col in missing_count.index],
        }
    )
    return summary.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def correlation_analysis(
    df: pd.DataFrame,
    method: str = "pearson",
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Compute correlation matrix and optional target correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method=method)

    if target_col and target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].sort_values(ascending=False)
        return corr_matrix, target_corr

    return corr_matrix, None


def remove_multicollinearity(
    df: pd.DataFrame,
    threshold: float = 0.9,
    ignore_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Drop highly correlated numeric features above the given threshold."""
    ignore_set = set(ignore_columns or [])
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ignore_set]

    if not numeric_cols:
        return df.copy(), [], pd.DataFrame()

    corr = df[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if (upper[column] > threshold).any()]

    reduced_df = df.drop(columns=to_drop, errors="ignore")
    return reduced_df, to_drop, corr


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from existing columns."""
    engineered = df.copy()

    if "RegistrationDate" in engineered.columns:
        registration = pd.to_datetime(engineered["RegistrationDate"], errors="coerce")
        engineered["RegistrationYear"] = registration.dt.year
        engineered["RegistrationMonth"] = registration.dt.month
        engineered["RegistrationDayOfWeek"] = registration.dt.dayofweek

        snapshot = registration.max()
        if pd.notna(snapshot):
            engineered["TenureDays"] = (snapshot - registration).dt.days

    if {"TotalSpend", "OrderCount"}.issubset(engineered.columns):
        order_count = engineered["OrderCount"].replace({0: np.nan})
        engineered["AvgBasketValue"] = engineered["TotalSpend"] / order_count

    return engineered


def outlier_detection(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: str = "iqr",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Detect outliers and return a summary plus a row-level outlier mask."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    mask = pd.Series(False, index=df.index)
    rows = []

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            rows.append({"column": col, "outlier_count": 0, "outlier_pct": 0.0, "method": method})
            continue

        if method.lower() == "zscore":
            std = valid.std(ddof=0)
            if std == 0:
                col_mask = pd.Series(False, index=df.index)
            else:
                z_scores = (series - valid.mean()) / std
                col_mask = z_scores.abs() > z_threshold
        else:
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            col_mask = (series < lower) | (series > upper)

        col_mask = col_mask.fillna(False)
        mask |= col_mask

        outlier_count = int(col_mask.sum())
        outlier_pct = (outlier_count / max(len(df), 1)) * 100
        rows.append(
            {
                "column": col,
                "outlier_count": outlier_count,
                "outlier_pct": round(outlier_pct, 2),
                "method": method,
            }
        )

    summary = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).reset_index(drop=True)
    return summary, mask


def encoding_categorical(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    nominal_cols: Optional[Sequence[str]] = None,
    ordinal_cols: Optional[Sequence[str]] = None,
    ordinal_categories: Optional[Dict[str, Sequence[Any]]] = None,
    handle_unknown: str = "ignore",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], ColumnTransformer, list[str]]:
    """Encode categorical features using OneHot (nominal) and Ordinal (ordinal)."""
    train_copy = train_df.copy()
    test_copy = test_df.copy() if test_df is not None else None

    if nominal_cols is None and ordinal_cols is None:
        detected = train_copy.select_dtypes(include=["object", "category"]).columns.tolist()
        nominal_cols = detected
        ordinal_cols = []
    else:
        nominal_cols = list(nominal_cols or [])
        ordinal_cols = list(ordinal_cols or [])

    for col in nominal_cols + ordinal_cols:
        if col in train_copy.columns:
            train_copy[col] = train_copy[col].astype("string").fillna("Missing")
        if test_copy is not None and col in test_copy.columns:
            test_copy[col] = test_copy[col].astype("string").fillna("Missing")

    transformers = []

    if nominal_cols:
        transformers.append(
            (
                "nominal",
                OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False),
                nominal_cols,
            )
        )

    if ordinal_cols:
        if ordinal_categories:
            categories = [list(ordinal_categories.get(col, sorted(train_copy[col].dropna().unique()))) for col in ordinal_cols]
        else:
            categories = [sorted(train_copy[col].dropna().unique()) for col in ordinal_cols]

        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            )
        )

    encoder = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    train_encoded_arr = encoder.fit_transform(train_copy)
    feature_names = encoder.get_feature_names_out().tolist()
    train_encoded = pd.DataFrame(train_encoded_arr, columns=feature_names, index=train_copy.index)

    test_encoded = None
    if test_copy is not None:
        test_encoded_arr = encoder.transform(test_copy)
        test_encoded = pd.DataFrame(test_encoded_arr, columns=feature_names, index=test_copy.index)

    return train_encoded, test_encoded, encoder, feature_names


def scaling_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """Scale numeric columns using StandardScaler fitted on train only."""
    train_scaled = train_df.copy()
    test_scaled = test_df.copy() if test_df is not None else None

    if columns is None:
        columns = train_scaled.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = list(columns)

    if not columns:
        return train_scaled, test_scaled, StandardScaler()

    scaler = StandardScaler()
    train_scaled[columns] = scaler.fit_transform(train_scaled[columns])

    if test_scaled is not None:
        test_scaled[columns] = scaler.transform(test_scaled[columns])

    return train_scaled, test_scaled, scaler
