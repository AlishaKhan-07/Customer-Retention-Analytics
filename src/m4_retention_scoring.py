"""
Module 4: Retention Strength Scoring

Builds a composite, explainable 0-100 "Retention Strength Score" per
customer from weighted business signals (not the ML model - this is a
transparent, rule-based score business users can trust and act on
without needing to understand a model).
"""

import pandas as pd
import numpy as np
import json
import os

IN_PATH = "outputs/customers_with_flags.csv"


def load():
    return pd.read_csv(IN_PATH)


def minmax(s):
    rng = s.max() - s.min()
    if rng == 0:
        return s * 0
    return (s - s.min()) / rng


def compute_retention_score(df):
    df = df.copy()

    # Individual normalized sub-scores (0-1), each weighted by business relevance
    active_score = df["IsActiveMember"]                       # 1 = active, strong retention signal
    tenure_score = minmax(df["Tenure"])                       # longer tenure -> stickier
    product_score = df["NumOfProducts"].apply(
        lambda n: 1.0 if n == 2 else (0.5 if n == 1 else 0.0)  # 2 products = healthy sweet spot
    )
    credit_score_norm = minmax(df["CreditScore"])
    balance_engaged = (df["Balance"] > 0).astype(int)          # non-zero balance = engaged
    card_score = df["HasCrCard"]

    weights = {
        "active": 0.30,
        "tenure": 0.15,
        "product": 0.25,
        "credit": 0.10,
        "balance": 0.10,
        "card": 0.10,
    }

    df["RetentionStrengthScore"] = (
        active_score * weights["active"]
        + tenure_score * weights["tenure"]
        + product_score * weights["product"]
        + credit_score_norm * weights["credit"]
        + balance_engaged * weights["balance"]
        + card_score * weights["card"]
    ) * 100
    df["RetentionStrengthScore"] = df["RetentionStrengthScore"].round(1)

    # Segment into bands
    df["RetentionBand"] = pd.cut(
        df["RetentionStrengthScore"],
        bins=[-1, 30, 50, 70, 101],
        labels=["Critical Risk", "At Risk", "Stable", "Strong"],
    )

    return df


def summarize(df):
    out = {}
    out["avg_retention_score"] = round(float(df["RetentionStrengthScore"].mean()), 2)
    out["retention_band_distribution"] = df["RetentionBand"].value_counts().to_dict()
    out["churn_rate_by_band"] = (
        df.groupby("RetentionBand", observed=True)["Exited"].mean().mul(100).round(2).to_dict()
    )
    out["avg_score_churned_vs_retained"] = (
        df.groupby("Exited")["RetentionStrengthScore"].mean().round(2).to_dict()
    )

    # Critical risk high-value customers = top retention priority list
    critical = df[
        (df["RetentionBand"] == "Critical Risk") & (df["EstimatedValue"] >= df["EstimatedValue"].quantile(0.5))
    ].sort_values("EstimatedValue", ascending=False)
    out["critical_risk_priority_count"] = int(len(critical))

    return out, critical


def run(save=True):
    df = load()
    df_scored = compute_retention_score(df)
    summary, priority_list = summarize(df_scored)

    if save:
        os.makedirs("outputs", exist_ok=True)
        df_scored.to_csv("outputs/customers_scored_final.csv", index=False)
        priority_cols = [
            "CustomerId", "Surname", "Geography", "EstimatedValue",
            "RetentionStrengthScore", "RetentionBand", "NumOfProducts",
            "IsActiveMember", "Exited",
        ]
        priority_list[priority_cols].to_csv("outputs/retention_priority_list.csv", index=False)
        with open("outputs/retention_scoring_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("Saved -> outputs/customers_scored_final.csv")
        print("Saved -> outputs/retention_priority_list.csv")
        print("Saved -> outputs/retention_scoring_summary.json")

    return df_scored, summary, priority_list


if __name__ == "__main__":
    df_scored, summary, priority_list = run()
    print(json.dumps(summary, indent=2, default=str))
