"""
Module 3: Product Utilization Analysis + High-Value Disengaged
Customer Identification

Analyzes how many products customers hold and how that relates to churn,
then flags the highest-value customers who show disengagement signals -
the segment most worth proactive retention effort.
"""

import pandas as pd
import numpy as np
import json
import os

IN_PATH = "outputs/engagement_scored.csv"


def load():
    return pd.read_csv(IN_PATH)


# ------------------------------------------------- Product utilization ---
def product_utilization(df):
    out = {}

    out["product_distribution"] = df["NumOfProducts"].value_counts().sort_index().to_dict()
    out["churn_rate_by_products"] = (
        df.groupby("NumOfProducts")["Exited"].mean().mul(100).round(2).to_dict()
    )
    out["avg_balance_by_products"] = (
        df.groupby("NumOfProducts")["Balance"].mean().round(2).to_dict()
    )
    out["product_tier_distribution"] = df["ProductTier"].value_counts().to_dict()
    out["churn_by_product_tier"] = (
        df.groupby("ProductTier", observed=True)["Exited"].mean().mul(100).round(2).to_dict()
    )

    # Cross-sell opportunity: single-product active customers with high balance
    cross_sell_target = df[
        (df["NumOfProducts"] == 1) & (df["IsActiveMember"] == 1) & (df["Balance"] > df["Balance"].median())
    ]
    out["cross_sell_opportunity_count"] = int(len(cross_sell_target))

    # Over-indexed risk: customers with 3-4 products (frequently a distressed-customer
    # signal in this dataset - worth surfacing explicitly)
    out["churn_rate_3plus_products"] = round(
        float(df[df["NumOfProducts"] >= 3]["Exited"].mean()) * 100, 2
    )

    return out


# --------------------------------------- High-value disengaged customers -
def high_value_disengaged(df):
    df = df.copy()

    # High value = top 25% by EstimatedValue (already computed in preprocessing)
    value_threshold = df["EstimatedValue"].quantile(0.75)

    # Disengagement signals: inactive member, zero/low balance-to-salary movement,
    # single product only, low engagement score
    df["DisengagementFlag"] = (
        (df["IsActiveMember"] == 0).astype(int)
        + (df["NumOfProducts"] == 1).astype(int)
        + (df["EngagementScore"] < df["EngagementScore"].median()).astype(int)
    )

    high_value_mask = df["EstimatedValue"] >= value_threshold
    disengaged_mask = df["DisengagementFlag"] >= 2  # at least 2 of 3 risk signals

    at_risk = df[high_value_mask & disengaged_mask].copy()
    at_risk = at_risk.sort_values("EstimatedValue", ascending=False)

    summary = {
        "high_value_threshold": round(float(value_threshold), 2),
        "high_value_customer_count": int(high_value_mask.sum()),
        "high_value_disengaged_count": int(len(at_risk)),
        "high_value_disengaged_pct_of_highvalue": round(
            len(at_risk) / max(high_value_mask.sum(), 1) * 100, 2
        ),
        "high_value_disengaged_churn_rate": round(float(at_risk["Exited"].mean()) * 100, 2)
        if len(at_risk)
        else 0.0,
        "estimated_value_at_risk": round(float(at_risk["EstimatedValue"].sum()), 2),
    }

    cols = [
        "CustomerId", "Surname", "Geography", "Age", "Balance", "EstimatedSalary",
        "EstimatedValue", "NumOfProducts", "IsActiveMember", "EngagementScore",
        "DisengagementFlag", "Exited",
    ]
    at_risk_export = at_risk[cols].reset_index(drop=True)

    return summary, at_risk_export, df


def run(save=True):
    df = load()
    product_summary = product_utilization(df)
    hv_summary, hv_customers, df_flagged = high_value_disengaged(df)

    combined = {"product_utilization": product_summary, "high_value_disengaged": hv_summary}

    if save:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/product_highvalue_summary.json", "w") as f:
            json.dump(combined, f, indent=2, default=str)
        hv_customers.to_csv("outputs/high_value_disengaged_customers.csv", index=False)
        df_flagged.to_csv("outputs/customers_with_flags.csv", index=False)
        print("Saved -> outputs/product_highvalue_summary.json")
        print("Saved -> outputs/high_value_disengaged_customers.csv")
        print("Saved -> outputs/customers_with_flags.csv")

    return combined, hv_customers, df_flagged


if __name__ == "__main__":
    combined, hv_customers, _ = run()
    print(json.dumps(combined, indent=2))
    print(f"\nTop 5 high-value disengaged customers:\n{hv_customers.head()}")
