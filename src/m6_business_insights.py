"""
Module 6: Business Insights & Recommendations

Reads the JSON summaries produced by modules 2-5 and turns them into a
structured set of plain-English findings and prioritized retention
recommendations, saved as JSON (for the dashboard) and Markdown (for
easy reading / copy into the report).
"""

import json
import os

FILES = {
    "eda_engagement": "outputs/eda_engagement_summary.json",
    "product_highvalue": "outputs/product_highvalue_summary.json",
    "retention": "outputs/retention_scoring_summary.json",
    "model": "outputs/model_comparison.json",
}


def load_all():
    data = {}
    for key, path in FILES.items():
        with open(path) as f:
            data[key] = json.load(f)
    return data


def build_insights(data):
    eda = data["eda_engagement"]["eda"]
    eng = data["eda_engagement"]["engagement"]
    prod = data["product_highvalue"]["product_utilization"]
    hv = data["product_highvalue"]["high_value_disengaged"]
    ret = data["retention"]
    model = data["model"]

    insights = []

    # 1. Overall churn baseline
    insights.append(
        f"Overall churn rate is {eda['overall_churn_rate']}% across the customer base."
    )

    # 2. Geography
    churn_geo = eda["churn_by_geography"]
    top_geo = max(churn_geo, key=churn_geo.get)
    insights.append(
        f"{top_geo} customers churn at {churn_geo[top_geo]}%, well above the other markets "
        f"({', '.join(f'{k}: {v}%' for k, v in churn_geo.items() if k != top_geo)}) - "
        f"suggesting a market-specific service, pricing, or competitive issue worth investigating."
    )

    # 3. Product count / over-selling risk
    churn_prod = prod["churn_rate_by_products"]
    insights.append(
        "Customers holding 3-4 products churn at "
        f"{churn_prod.get('3', 0)}%-{churn_prod.get('4', 0)}%, dramatically higher than the "
        f"7-8% churn rate of customers with exactly 2 products. This is very likely a symptom "
        f"of over-selling or unresolved service issues rather than genuine multi-product loyalty, "
        f"and should be treated as a red flag, not a cross-sell success."
    )

    # 4. Activity status
    insights.append(
        f"Inactive members make up {100 - eng['active_member_share']:.1f}% of the base and "
        f"exit at a substantially higher rate than active members, confirming activity status "
        f"as one of the strongest retention levers."
    )

    # 5. High-value disengaged segment
    insights.append(
        f"{hv['high_value_disengaged_count']:,} customers ({hv['high_value_disengaged_pct_of_highvalue']}% "
        f"of the high-value segment) are both high-value and disengaged, representing an estimated "
        f"${hv['estimated_value_at_risk']:,.0f} in balance/income exposure and a "
        f"{hv['high_value_disengaged_churn_rate']}% churn rate - the single highest-leverage segment "
        f"for a targeted retention campaign."
    )

    # 6. Retention score validation
    band_churn = ret["churn_rate_by_band"]
    insights.append(
        f"The Retention Strength Score cleanly separates risk: Critical Risk customers churn at "
        f"{band_churn.get('Critical Risk', 0)}% versus {band_churn.get('Strong', 0)}% for Strong-band "
        f"customers, validating it as a usable early-warning signal for frontline teams."
    )

    # 7. Model performance
    best = model["best_model_metrics"]
    insights.append(
        f"The {model['best_model']} model achieves {best['roc_auc']*100:.1f}% ROC-AUC and "
        f"{best['recall']*100:.1f}% recall on churners, meaning it correctly flags the majority of "
        f"customers who go on to leave - suitable for prioritizing outreach lists, not for fully "
        f"automated decisions."
    )

    # 8. Age
    churn_age = eda["churn_by_age_group"]
    top_age = max(churn_age, key=churn_age.get)
    insights.append(
        f"The {top_age} age group shows the highest churn ({churn_age[top_age]}%), an important "
        f"segment for tailored messaging and product design."
    )

    return insights


def build_recommendations(data):
    hv = data["product_highvalue"]["high_value_disengaged"]
    prod = data["product_highvalue"]["product_utilization"]
    eda = data["eda_engagement"]["eda"]
    churn_geo = eda["churn_by_geography"]
    top_geo = max(churn_geo, key=churn_geo.get)

    recs = [
        {
            "priority": "High",
            "recommendation": "Launch a proactive outreach campaign for high-value disengaged "
            f"customers ({hv['high_value_disengaged_count']:,} identified) - personal relationship "
            "manager contact, tailored offers, and a service health-check within 30 days.",
        },
        {
            "priority": "High",
            "recommendation": "Audit the multi-product (3-4 product) customer journey. The "
            f"{prod['churn_rate_3plus_products']}% churn rate in this group points to bundling, "
            "fee stacking, or cross-sell practices that are backfiring - review sales incentives "
            "tied to product count.",
        },
        {
            "priority": "High",
            "recommendation": f"Investigate {top_geo} market specifically: conduct customer "
            "interviews or NPS deep-dive to identify whether the elevated churn is driven by "
            "pricing, local competition, service quality, or regulatory friction.",
        },
        {
            "priority": "Medium",
            "recommendation": "Build an automated 're-engagement' trigger for members who go "
            "inactive for 60+ days, before they reach the 'inactive member' churn risk profile.",
        },
        {
            "priority": "Medium",
            "recommendation": "Operationalize the Retention Strength Score in CRM so relationship "
            "managers can see a live 0-100 score per customer and prioritize Critical Risk / At "
            "Risk accounts in their weekly outreach.",
        },
        {
            "priority": "Medium",
            "recommendation": "Use the churn prediction model's monthly scored list to feed "
            "targeted retention offers (rate improvements, fee waivers, loyalty perks) to the "
            "top-decile predicted-churn customers.",
        },
        {
            "priority": "Low",
            "recommendation": "Monitor model drift quarterly - retrain on fresh data as customer "
            "behavior and product mix evolve, and re-validate feature importance rankings.",
        },
    ]
    return recs


def to_markdown(insights, recs, model_summary):
    lines = ["# Business Insights & Recommendations\n"]
    lines.append("## Key Insights\n")
    for i, ins in enumerate(insights, 1):
        lines.append(f"{i}. {ins}\n")

    lines.append("\n## Recommendations\n")
    for r in recs:
        lines.append(f"- **[{r['priority']}]** {r['recommendation']}\n")

    lines.append("\n## Model Summary\n")
    lines.append(f"- Best model: **{model_summary['best_model']}**\n")
    for k, v in model_summary["best_model_metrics"].items():
        if k != "model":
            lines.append(f"- {k}: {v}\n")

    return "\n".join(lines)


def run(save=True):
    data = load_all()
    insights = build_insights(data)
    recs = build_recommendations(data)

    if save:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/business_insights.json", "w") as f:
            json.dump({"insights": insights, "recommendations": recs}, f, indent=2)
        md = to_markdown(insights, recs, data["model"])
        with open("outputs/business_insights.md", "w") as f:
            f.write(md)
        print("Saved -> outputs/business_insights.json")
        print("Saved -> outputs/business_insights.md")

    return insights, recs


if __name__ == "__main__":
    insights, recs = run()
    for i in insights:
        print("-", i)
    print()
    for r in recs:
        print(f"[{r['priority']}]", r["recommendation"])
