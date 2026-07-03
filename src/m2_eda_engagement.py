"""
Module 2: Exploratory Data Analysis (EDA) + Customer Engagement Analysis

Generates summary statistics, distribution/correlation views, and
engagement-specific metrics (activity status, tenure patterns,
demographic churn cuts) used by the dashboard and business report.
"""

import pandas as pd
import numpy as np
import json
import os

IN_PATH = "outputs/processed_customers.csv"


def load():
    return pd.read_csv(IN_PATH)


# ---------------------------------------------------------------- EDA -----
def run_eda(df):
    summary = {}

    summary["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    summary["overall_churn_rate"] = round(float(df["Exited"].mean()) * 100, 2)

    summary["numeric_summary"] = (
        df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]]
        .describe()
        .round(2)
        .to_dict()
    )

    summary["geography_distribution"] = df["Geography"].value_counts().to_dict()
    summary["gender_distribution"] = df["Gender"].value_counts().to_dict()

    # Churn rate cuts by key categorical dimensions
    summary["churn_by_geography"] = (
        df.groupby("Geography")["Exited"].mean().mul(100).round(2).to_dict()
    )
    summary["churn_by_gender"] = (
        df.groupby("Gender")["Exited"].mean().mul(100).round(2).to_dict()
    )
    summary["churn_by_age_group"] = (
        df.groupby("AgeGroup", observed=True)["Exited"].mean().mul(100).round(2).to_dict()
    )
    summary["churn_by_num_products"] = (
        df.groupby("NumOfProducts")["Exited"].mean().mul(100).round(2).to_dict()
    )
    summary["churn_by_active_member"] = (
        df.groupby("IsActiveMember")["Exited"].mean().mul(100).round(2).to_dict()
    )
    summary["churn_by_has_cr_card"] = (
        df.groupby("HasCrCard")["Exited"].mean().mul(100).round(2).to_dict()
    )

    # Correlation matrix (numeric features vs churn)
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited",
    ]
    corr = df[num_cols].corr(numeric_only=True)["Exited"].drop("Exited").sort_values(
        ascending=False
    )
    summary["correlation_with_churn"] = corr.round(3).to_dict()

    return summary


# --------------------------------------------------- Engagement analysis --
def engagement_analysis(df):
    eng = {}

    # Active vs inactive base rates
    eng["active_member_share"] = round(float(df["IsActiveMember"].mean()) * 100, 2)

    # Engagement score: composite of activity, credit card, tenure length, product count
    df = df.copy()
    df["EngagementScore"] = (
        df["IsActiveMember"] * 40
        + df["HasCrCard"] * 15
        + (df["Tenure"] / df["Tenure"].max()) * 20
        + (df["NumOfProducts"] / df["NumOfProducts"].max()) * 25
    ).round(2)

    eng["avg_engagement_score"] = round(float(df["EngagementScore"].mean()), 2)
    eng["engagement_score_by_churn"] = (
        df.groupby("Exited")["EngagementScore"].mean().round(2).to_dict()
    )

    # Tenure-based engagement trend
    eng["churn_by_tenure_group"] = (
        df.groupby("TenureGroup", observed=True)["Exited"].mean().mul(100).round(2).to_dict()
    )

    # Cross-cut: inactive + low tenure = highest disengagement risk
    disengaged = df[(df["IsActiveMember"] == 0) & (df["Tenure"] <= 2)]
    eng["disengaged_low_tenure_count"] = int(len(disengaged))
    eng["disengaged_low_tenure_churn_rate"] = (
        round(float(disengaged["Exited"].mean()) * 100, 2) if len(disengaged) else 0.0
    )

    df.to_csv("outputs/engagement_scored.csv", index=False)
    return eng, df


def run(save=True):
    df = load()
    eda_summary = run_eda(df)
    eng_summary, df_eng = engagement_analysis(df)

    combined = {"eda": eda_summary, "engagement": eng_summary}

    if save:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/eda_engagement_summary.json", "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print("Saved -> outputs/eda_engagement_summary.json")
        print("Saved -> outputs/engagement_scored.csv")

    return combined, df_eng


if __name__ == "__main__":
    combined, _ = run()
    print(json.dumps(combined["eda"]["churn_by_geography"], indent=2))
    print(json.dumps(combined["engagement"], indent=2))
