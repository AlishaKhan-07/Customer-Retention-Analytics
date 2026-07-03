"""
Module 1: Data Preprocessing
Customer Retention & Churn Prediction Analytics Project

Loads the raw European Bank customer dataset, cleans it, engineers
foundational fields, and writes a processed dataset used by every
downstream module (EDA, engagement, product utilization, retention
scoring, ML modeling, and the Streamlit dashboard).
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/European_Bank.csv"
OUT_PATH = "outputs/processed_customers.csv"


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dupes_removed = before - len(df)

    # Drop duplicate CustomerId, keep first occurrence
    before = len(df)
    df = df.drop_duplicates(subset="CustomerId", keep="first")
    dupe_ids_removed = before - len(df)

    # Basic sanity checks / clipping (protects against corrupt values)
    df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]
    df = df[df["CreditScore"].between(300, 900)]
    df = df[df["NumOfProducts"].between(1, 4)]
    df = df[df["Tenure"].between(0, 20)]

    # No missing values expected, but handle defensively
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    cat_cols = df.select_dtypes(include="object").columns
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    print(f"Removed {dupes_removed} exact duplicate rows")
    print(f"Removed {dupe_ids_removed} duplicate CustomerId rows")
    print(f"Final row count: {len(df)}")

    return df.reset_index(drop=True)


def engineer_base_features(df):
    """Foundational engineered fields reused across modules."""
    df = df.copy()

    # Balance-to-salary ratio (financial engagement signal)
    df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"].replace(0, np.nan)
    df["BalanceSalaryRatio"] = df["BalanceSalaryRatio"].fillna(0)

    # Has zero balance flag (common churn indicator in bank data)
    df["HasZeroBalance"] = (df["Balance"] == 0).astype(int)

    # Age bands for cohort analysis
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[18, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"],
        right=True,
    )

    # Tenure bands
    df["TenureGroup"] = pd.cut(
        df["Tenure"],
        bins=[-1, 1, 3, 6, 10, 20],
        labels=["0-1 yrs", "2-3 yrs", "4-6 yrs", "7-10 yrs", "10+ yrs"],
    )

    # Product utilization tier
    df["ProductTier"] = pd.cut(
        df["NumOfProducts"],
        bins=[0, 1, 2, 4],
        labels=["Single Product", "Two Products", "Multi-Product (3-4)"],
    )

    # Customer value tier (based on balance + salary as proxy for value)
    df["EstimatedValue"] = df["Balance"] * 0.6 + df["EstimatedSalary"] * 0.4
    df["ValueTier"] = pd.qcut(
        df["EstimatedValue"], q=4, labels=["Low", "Medium", "High", "Premium"]
    )

    return df


def run(save=True):
    os.makedirs("outputs", exist_ok=True)
    df = load_raw()
    df = clean_data(df)
    df = engineer_base_features(df)

    if save:
        df.to_csv(OUT_PATH, index=False)
        print(f"Saved processed dataset -> {OUT_PATH} ({df.shape[0]} rows, {df.shape[1]} cols)")

    return df


if __name__ == "__main__":
    run()
