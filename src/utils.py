# src/utils.py

import pandas as pd
import numpy as np


def report_shape(df):
    """Affiche la dimension du dataset"""
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


def report_missing_values(df):
    """Affiche le nombre et le pourcentage de valeurs manquantes"""
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    report = pd.DataFrame({
        "MissingCount": missing_count,
        "MissingPercent": missing_percent
    })

    report = report[report["MissingCount"] > 0]
    return report.sort_values(by="MissingPercent", ascending=False)


def drop_constant_columns(df):
    """Supprime les colonnes à variance nulle"""
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    return df, constant_cols


def correlation_matrix(df, threshold=0.8):
    """Retourne les paires de colonnes fortement corrélées"""
    corr_matrix = df.corr(numeric_only=True)
    high_corr = []

    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if col != row and abs(corr_matrix.loc[row, col]) > threshold:
                high_corr.append((row, col, corr_matrix.loc[row, col]))

    return high_corr