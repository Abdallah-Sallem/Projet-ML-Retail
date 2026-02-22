# src/preprocessing.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from utils import report_shape, report_missing_values, drop_constant_columns


def load_data(path):
    df = pd.read_csv(path)
    return df


def parse_dates(df):
    df["RegistrationDate"] = pd.to_datetime(
        df["RegistrationDate"],
        dayfirst=True,
        errors="coerce"
    )

    df["RegYear"] = df["RegistrationDate"].dt.year
    df["RegMonth"] = df["RegistrationDate"].dt.month
    df["RegDay"] = df["RegistrationDate"].dt.day
    df["RegWeekday"] = df["RegistrationDate"].dt.weekday

    df = df.drop(columns=["RegistrationDate"])
    return df


def basic_cleaning(df):
    df, dropped_cols = drop_constant_columns(df)
    print("Dropped constant columns:", dropped_cols)
    return df


def feature_engineering(df):
    if "MonetaryTotal" in df.columns and "Recency" in df.columns:
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)

    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)

    return df


def split_data(df, target="Churn"):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_numeric(X_train, X_test):
    numeric_cols = X_train.select_dtypes(include=np.number).columns

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, scaler


def main():
    df = load_data("data/raw/dataset.csv")

    report_shape(df)

    print(report_missing_values(df))

    df = parse_dates(df)
    df = basic_cleaning(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, scaler = scale_numeric(X_train, X_test)

    X_train.to_csv("data/train_test/X_train.csv", index=False)
    X_test.to_csv("data/train_test/X_test.csv", index=False)
    y_train.to_csv("data/train_test/y_train.csv", index=False)
    y_test.to_csv("data/train_test/y_test.csv", index=False)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()