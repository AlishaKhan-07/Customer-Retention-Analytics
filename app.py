"""
Customer Retention & Churn Prediction Analytics Dashboard
Streamlit application - 8 pages covering Part 1 (Data Analytics) and
Part 2 (AI/ML churn prediction as a value-add feature).
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Bank Churn Analytics & Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------- Loaders --
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    feature_list = joblib.load("models/feature_list.pkl")
    model_name = joblib.load("models/best_model_name.pkl")
    return model, scaler, encoders, feature_list, model_name


DATA_PATH = "outputs/customers_scored_final.csv"
EDA_PATH = "outputs/eda_engagement_summary.json"
PROD_PATH = "outputs/product_highvalue_summary.json"
RET_PATH = "outputs/retention_scoring_summary.json"
MODEL_PATH = "outputs/model_comparison.json"
INSIGHTS_PATH = "outputs/business_insights.json"
HV_PATH = "outputs/high_value_disengaged_customers.csv"
PRIORITY_PATH = "outputs/retention_priority_list.csv"

df = load_csv(DATA_PATH)
eda_data = load_json(EDA_PATH)
prod_data = load_json(PROD_PATH)
ret_data = load_json(RET_PATH)
model_data = load_json(MODEL_PATH)
insights_data = load_json(INSIGHTS_PATH)
hv_df = load_csv(HV_PATH)
priority_df = load_csv(PRIORITY_PATH)

COLORS = {"churn": "#E4572E", "retain": "#17A398", "neutral": "#4C5C68", "accent": "#F4B942"}

# ---------------------------------------------------------------- Sidebar -
st.sidebar.title("🏦 Bank Churn Analytics")
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Dataset Overview",
        "Customer Engagement Analytics",
        "Product Utilization Analytics",
        "High-Value Customer Detector",
        "Retention Strength Score",
        "Churn Prediction (AI/ML)",
        "Business Recommendations",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} customers")
st.sidebar.caption(f"Overall churn rate: {eda_data['eda']['overall_churn_rate']}%")
st.sidebar.caption(f"Best model: {model_data['best_model']}")

# ================================================================== HOME ==
if page == "Home":
    st.title("Customer Retention & Churn Prediction Analytics")
    st.markdown(
        "An end-to-end analytics platform for a European retail bank: understand **why** "
        "customers leave, **who** is most at risk, and **which** customers to prioritize for "
        "retention outreach - backed by an AI/ML churn prediction model."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Overall Churn Rate", f"{eda_data['eda']['overall_churn_rate']}%")
    c3.metric("High-Value Disengaged", f"{prod_data['high_value_disengaged']['high_value_disengaged_count']:,}")
    c4.metric("Best Model ROC-AUC", f"{model_data['best_model_metrics']['roc_auc']*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Geography")
        geo_churn = eda_data["eda"]["churn_by_geography"]
        fig = px.bar(
            x=list(geo_churn.keys()), y=list(geo_churn.values()),
            labels={"x": "Country", "y": "Churn Rate (%)"},
            color=list(geo_churn.values()),
            color_continuous_scale=["#17A398", "#E4572E"],
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Retention Band Distribution")
        band_dist = ret_data["retention_band_distribution"]
        order = ["Critical Risk", "At Risk", "Stable", "Strong"]
        band_dist = {k: band_dist.get(k, 0) for k in order if k in band_dist}
        fig = px.pie(
            names=list(band_dist.keys()), values=list(band_dist.values()),
            color=list(band_dist.keys()),
            color_discrete_map={
                "Critical Risk": "#E4572E", "At Risk": "#F4B942",
                "Stable": "#7FB3D5", "Strong": "#17A398",
            },
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Insights at a Glance")
    for ins in insights_data["insights"][:4]:
        st.markdown(f"- {ins}")
    st.info("See the **Business Recommendations** page for the full insight and action list.")

# ========================================================= DATASET OVERVIEW
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.caption("Part 1: Data Preprocessing & Exploratory Data Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{eda_data['eda']['shape']['rows']:,}")
    c2.metric("Columns", eda_data['eda']['shape']['columns'])
    c3.metric("Churned", f"{int(df['Exited'].sum()):,}")
    c4.metric("Retained", f"{int((df['Exited']==0).sum()):,}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Raw Data Sample", "Summary Statistics", "Correlation with Churn"])

    with tab1:
        st.dataframe(df.head(50), use_container_width=True)
        st.caption("Preprocessing: duplicate removal, range validation, and feature engineering "
                   "(age/tenure bands, value tiers, engagement score) already applied.")

    with tab2:
        num_summary = pd.DataFrame(eda_data["eda"]["numeric_summary"])
        st.dataframe(num_summary, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Geography distribution**")
            st.bar_chart(pd.Series(eda_data["eda"]["geography_distribution"]))
        with colB:
            st.markdown("**Gender distribution**")
            st.bar_chart(pd.Series(eda_data["eda"]["gender_distribution"]))

    with tab3:
        corr = eda_data["eda"]["correlation_with_churn"]
        corr_series = pd.Series(corr).sort_values()
        fig = px.bar(
            x=corr_series.values, y=corr_series.index, orientation="h",
            labels={"x": "Correlation with Churn", "y": "Feature"},
            color=corr_series.values, color_continuous_scale=["#17A398", "#E4572E"],
        )
        fig.update_layout(coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Age and Balance show the strongest positive association with churn in this "
                   "dataset; IsActiveMember shows the strongest negative (protective) association.")

# ================================================== CUSTOMER ENGAGEMENT ==
elif page == "Customer Engagement Analytics":
    st.title("🤝 Customer Engagement Analytics")
    st.caption("Part 1: Engagement scoring, activity status, and tenure patterns")

    eng = eda_data["engagement"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Members", f"{eng['active_member_share']}%")
    c2.metric("Avg Engagement Score", f"{eng['avg_engagement_score']}/100")
    c3.metric("Disengaged + Low Tenure", f"{eng['disengaged_low_tenure_count']:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Engagement Score: Churned vs Retained")
        es = eng["engagement_score_by_churn"]
        fig = px.bar(
            x=["Retained", "Churned"], y=[es.get("0", 0), es.get("1", 0)],
            color=["Retained", "Churned"],
            color_discrete_map={"Retained": COLORS["retain"], "Churned": COLORS["churn"]},
            labels={"x": "", "y": "Avg Engagement Score"},
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Tenure Group")
        tg = eng["churn_by_tenure_group"]
        fig = px.line(
            x=list(tg.keys()), y=list(tg.values()), markers=True,
            labels={"x": "Tenure Group", "y": "Churn Rate (%)"},
        )
        fig.update_traces(line_color=COLORS["churn"])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Activity Status Breakdown")
    colA, colB = st.columns(2)
    with colA:
        active_churn = eda_data["eda"]["churn_by_active_member"]
        fig = px.bar(
            x=["Inactive", "Active"], y=[active_churn.get("0", 0), active_churn.get("1", 0)],
            color=["Inactive", "Active"],
            color_discrete_map={"Inactive": COLORS["churn"], "Active": COLORS["retain"]},
            labels={"x": "", "y": "Churn Rate (%)"},
        )
        fig.update_layout(showlegend=False, height=320, title="Churn Rate: Active vs Inactive")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        cc = eda_data["eda"]["churn_by_has_cr_card"]
        fig = px.bar(
            x=["No Credit Card", "Has Credit Card"], y=[cc.get("0", 0), cc.get("1", 0)],
            color=["No Credit Card", "Has Credit Card"],
            color_discrete_map={"No Credit Card": COLORS["neutral"], "Has Credit Card": COLORS["accent"]},
            labels={"x": "", "y": "Churn Rate (%)"},
        )
        fig.update_layout(showlegend=False, height=320, title="Churn Rate: Credit Card Ownership")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"**Key finding:** {eng['disengaged_low_tenure_count']:,} customers are both inactive and "
        f"in their first 2 years of tenure - this group churns at "
        f"{eng['disengaged_low_tenure_churn_rate']}%, making early-tenure engagement critical."
    )

# =================================================== PRODUCT UTILIZATION ==
elif page == "Product Utilization Analytics":
    st.title("📦 Product Utilization Analytics")
    st.caption("Part 1: Product holding patterns and their relationship to churn")

    p = prod_data["product_utilization"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-Sell Opportunity", f"{p['cross_sell_opportunity_count']:,}")
    c2.metric("Churn Rate (3+ Products)", f"{p['churn_rate_3plus_products']}%")
    c3.metric("Two-Product Churn Rate", f"{p['churn_rate_by_products'].get('2', 0)}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Count by Number of Products")
        pd_dist = p["product_distribution"]
        fig = px.bar(
            x=list(pd_dist.keys()), y=list(pd_dist.values()),
            labels={"x": "Number of Products", "y": "Customer Count"},
            color_discrete_sequence=[COLORS["neutral"]],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Number of Products")
        cp = p["churn_rate_by_products"]
        fig = px.bar(
            x=list(cp.keys()), y=list(cp.values()),
            labels={"x": "Number of Products", "y": "Churn Rate (%)"},
            color=list(cp.values()), color_continuous_scale=["#17A398", "#E4572E"],
        )
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "⚠️ **Red flag, not a success metric:** churn rate jumps to "
        f"{cp.get('3', 0)}% at 3 products and {cp.get('4', 0)}% at 4 products. This pattern "
        "typically signals over-selling, fee stacking, or unresolved service complaints rather "
        "than genuine loyalty - it should trigger a service review, not more upsell activity."
    )

    st.markdown("---")
    st.subheader("Average Balance by Product Count")
    ab = p["avg_balance_by_products"]
    fig = px.bar(
        x=list(ab.keys()), y=list(ab.values()),
        labels={"x": "Number of Products", "y": "Average Balance"},
        color_discrete_sequence=[COLORS["accent"]],
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ================================================ HIGH-VALUE CUSTOMER DETECTOR
elif page == "High-Value Customer Detector":
    st.title("💎 High-Value Customer Detector")
    st.caption("Part 1: Identifying high-value customers who show disengagement signals")

    hv = prod_data["high_value_disengaged"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High-Value Customers", f"{hv['high_value_customer_count']:,}")
    c2.metric("High-Value + Disengaged", f"{hv['high_value_disengaged_count']:,}")
    c3.metric("Their Churn Rate", f"{hv['high_value_disengaged_churn_rate']}%")
    c4.metric("Estimated Value at Risk", f"${hv['estimated_value_at_risk']:,.0f}")

    st.markdown("---")
    st.subheader("High-Value Disengaged Customer List")
    st.caption(
        "Customers in the top 25% by estimated value (0.6×Balance + 0.4×EstimatedSalary) "
        "showing at least 2 of 3 disengagement signals: inactive membership, single-product "
        "relationship, or below-median engagement score."
    )

    colf1, colf2 = st.columns(2)
    with colf1:
        geo_filter = st.multiselect(
            "Filter by Geography", options=sorted(hv_df["Geography"].unique()),
            default=sorted(hv_df["Geography"].unique()),
        )
    with colf2:
        churned_filter = st.selectbox("Churn Status", ["All", "Churned only", "Retained only"])

    filtered = hv_df[hv_df["Geography"].isin(geo_filter)]
    if churned_filter == "Churned only":
        filtered = filtered[filtered["Exited"] == 1]
    elif churned_filter == "Retained only":
        filtered = filtered[filtered["Exited"] == 0]

    st.dataframe(
        filtered.style.format({"Balance": "${:,.0f}", "EstimatedSalary": "${:,.0f}", "EstimatedValue": "${:,.0f}"}),
        use_container_width=True, height=400,
    )
    st.caption(f"Showing {len(filtered):,} of {len(hv_df):,} high-value disengaged customers")

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered list (CSV)", csv, "high_value_disengaged.csv", "text/csv")

# ======================================================= RETENTION SCORE ==
elif page == "Retention Strength Score":
    st.title("🛡️ Retention Strength Score")
    st.caption("Part 1: Composite 0-100 rule-based retention health score per customer")

    c1, c2 = st.columns(2)
    c1.metric("Average Score", f"{ret_data['avg_retention_score']}/100")
    c2.metric("Critical Risk (High-Value)", f"{ret_data['critical_risk_priority_count']:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Retention Band Distribution")
        band_dist = ret_data["retention_band_distribution"]
        order = ["Critical Risk", "At Risk", "Stable", "Strong"]
        band_dist = {k: band_dist.get(k, 0) for k in order if k in band_dist}
        fig = px.bar(
            x=list(band_dist.keys()), y=list(band_dist.values()),
            color=list(band_dist.keys()),
            color_discrete_map={
                "Critical Risk": "#E4572E", "At Risk": "#F4B942",
                "Stable": "#7FB3D5", "Strong": "#17A398",
            },
            labels={"x": "Retention Band", "y": "Customer Count"},
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Retention Band")
        bc = ret_data["churn_rate_by_band"]
        bc = {k: bc.get(k, 0) for k in order if k in bc}
        fig = px.bar(
            x=list(bc.keys()), y=list(bc.values()),
            color=list(bc.keys()),
            color_discrete_map={
                "Critical Risk": "#E4572E", "At Risk": "#F4B942",
                "Stable": "#7FB3D5", "Strong": "#17A398",
            },
            labels={"x": "Retention Band", "y": "Churn Rate (%)"},
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Retention Priority List")
    st.caption("High-value customers currently in the Critical Risk band - top priority for retention teams.")
    st.dataframe(
        priority_df.style.format({"EstimatedValue": "${:,.0f}", "RetentionStrengthScore": "{:.1f}"}),
        use_container_width=True, height=350,
    )
    csv = priority_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download priority list (CSV)", csv, "retention_priority_list.csv", "text/csv")

    with st.expander("How is the score calculated?"):
        st.markdown(
            """
The Retention Strength Score is a **transparent, rule-based** score (not the ML model),
so business teams can trust and act on it directly:

| Signal | Weight | Logic |
|---|---|---|
| Active membership | 30% | Active = full credit |
| Product count | 25% | 2 products = full credit (sweet spot), 1 = half, 3-4 = none |
| Tenure | 15% | Normalized, longer = higher |
| Credit score | 10% | Normalized |
| Non-zero balance | 10% | Engaged balance = full credit |
| Credit card ownership | 10% | Has card = full credit |

Bands: **Critical Risk** (0-30), **At Risk** (30-50), **Stable** (50-70), **Strong** (70-100)
            """
        )

# =============================================== CHURN PREDICTION (AI/ML) =
elif page == "Churn Prediction (AI/ML)":
    st.title("🤖 Churn Prediction (AI/ML)")
    st.caption("Part 2: Value-add ML model - compare algorithms, then predict for a single customer")

    tab1, tab2 = st.tabs(["Model Comparison", "Predict a Customer"])

    # ------------------------------------------------------ Model comparison
    with tab1:
        st.subheader("Model Performance Comparison")
        results_df = pd.DataFrame(model_data["results"]).set_index("model")
        st.dataframe(
            results_df.style.highlight_max(axis=0, color="#d4f7ec").format("{:.4f}"),
            use_container_width=True,
        )
        st.success(f"**Selected model: {model_data['best_model']}** (highest ROC-AUC, tie-broken by F1-score)")

        metric_choice = st.selectbox(
            "Compare models on:", ["roc_auc", "accuracy", "precision", "recall", "f1_score"]
        )
        fig = px.bar(
            results_df.reset_index(), x="model", y=metric_choice, color="model",
            labels={"model": "Model", metric_choice: metric_choice.replace("_", " ").title()},
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("ROC Curves")
        try:
            roc_data = load_json("outputs/roc_curve_data.json")
            fig = go.Figure()
            for name, rd in roc_data.items():
                fig.add_trace(go.Scatter(x=rd["fpr"], y=rd["tpr"], mode="lines", name=name))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                      line=dict(dash="dash", color="gray")))
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=420)
            st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.warning("ROC curve data not found - re-run the pipeline to generate it.")

        st.markdown("---")
        st.subheader(f"Feature Importance ({model_data['best_model']})")
        fi = pd.Series(model_data["feature_importance"]).sort_values()
        fig = px.bar(x=fi.values, y=fi.index, orientation="h",
                      labels={"x": "Importance", "y": "Feature"},
                      color_discrete_sequence=[COLORS["neutral"]])
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------- Live prediction
    with tab2:
        st.subheader("Predict Churn Probability for a Customer")
        st.caption("Enter customer attributes below - this is an optional, illustrative prediction tool.")

        try:
            model, scaler, encoders, feature_list, model_name = load_model_artifacts()
            model_ready = True
        except FileNotFoundError:
            model_ready = False
            st.error("Model artifacts not found. Run `python run_pipeline.py` first.")

        if model_ready:
            with st.form("prediction_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    credit_score = st.slider("Credit Score", 300, 900, 650)
                    age = st.slider("Age", 18, 100, 40)
                    tenure = st.slider("Tenure (years)", 0, 20, 5)
                with c2:
                    balance = st.number_input("Balance", 0.0, 300000.0, 60000.0, step=1000.0)
                    salary = st.number_input("Estimated Salary", 0.0, 250000.0, 100000.0, step=1000.0)
                    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
                with c3:
                    geography = st.selectbox("Geography", sorted(encoders["Geography"].classes_))
                    gender = st.selectbox("Gender", sorted(encoders["Gender"].classes_))
                    has_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
                    is_active = st.radio("Active Member?", ["Yes", "No"], horizontal=True)

                submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

            if submitted:
                has_cr_card = 1 if has_card == "Yes" else 0
                is_active_member = 1 if is_active == "Yes" else 0

                balance_salary_ratio = balance / salary if salary > 0 else 0
                has_zero_balance = 1 if balance == 0 else 0

                engagement_score = (
                    is_active_member * 40 + has_cr_card * 15
                    + (tenure / 10) * 20 + (num_products / 4) * 25
                )

                product_score = 1.0 if num_products == 2 else (0.5 if num_products == 1 else 0.0)
                retention_score = (
                    is_active_member * 0.30 + min(tenure / 10, 1) * 0.15
                    + product_score * 0.25 + (credit_score / 900) * 0.10
                    + (1 if balance > 0 else 0) * 0.10 + has_cr_card * 0.10
                ) * 100

                geo_enc = encoders["Geography"].transform([geography])[0]
                gender_enc = encoders["Gender"].transform([gender])[0]

                row = pd.DataFrame([{
                    "CreditScore": credit_score, "Age": age, "Tenure": tenure,
                    "Balance": balance, "NumOfProducts": num_products,
                    "HasCrCard": has_cr_card, "IsActiveMember": is_active_member,
                    "EstimatedSalary": salary, "BalanceSalaryRatio": balance_salary_ratio,
                    "HasZeroBalance": has_zero_balance, "EngagementScore": engagement_score,
                    "RetentionStrengthScore": retention_score,
                    "Geography_enc": geo_enc, "Gender_enc": gender_enc,
                }])[feature_list]

                if model_name == "Logistic Regression":
                    row_input = scaler.transform(row)
                else:
                    row_input = row

                prob = model.predict_proba(row_input)[0][1]
                pred = "Likely to Churn" if prob >= 0.5 else "Likely to Stay"

                st.markdown("---")
                colA, colB = st.columns([1, 2])
                with colA:
                    st.metric("Churn Probability", f"{prob*100:.1f}%")
                    if prob >= 0.5:
                        st.error(f"⚠️ {pred}")
                    else:
                        st.success(f"✅ {pred}")
                with colB:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#E4572E" if prob >= 0.5 else "#17A398"},
                            "steps": [
                                {"range": [0, 30], "color": "#e6f7f2"},
                                {"range": [30, 60], "color": "#fef3d5"},
                                {"range": [60, 100], "color": "#fde3dc"},
                            ],
                        },
                        title={"text": "Churn Risk"},
                    ))
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)

                st.caption(f"Predicted using {model_name} (test-set ROC-AUC: {model_data['best_model_metrics']['roc_auc']:.3f}). "
                           "Use as a prioritization signal alongside the Retention Strength Score, not a sole decision-maker.")

# ================================================== BUSINESS RECOMMENDATIONS
elif page == "Business Recommendations":
    st.title("📋 Business Insights & Recommendations")
    st.caption("Part 1: Synthesized findings and prioritized action plan")

    st.subheader("Key Insights")
    for i, ins in enumerate(insights_data["insights"], 1):
        st.markdown(f"**{i}.** {ins}")

    st.markdown("---")
    st.subheader("Prioritized Recommendations")

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recs_sorted = sorted(insights_data["recommendations"], key=lambda r: priority_order[r["priority"]])

    for r in recs_sorted:
        color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[r["priority"]]
        st.markdown(f"{color} **[{r['priority']}]** {r['recommendation']}")

    st.markdown("---")
    st.subheader("Model Summary")
    best = model_data["best_model_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", model_data["best_model"])
    c2.metric("ROC-AUC", f"{best['roc_auc']:.3f}")
    c3.metric("Recall (churners caught)", f"{best['recall']*100:.1f}%")
    c4.metric("Precision", f"{best['precision']*100:.1f}%")

    st.markdown("---")
    with open("outputs/business_insights.md", "rb") as f:
        st.download_button(
            "Download full report (Markdown)", f, "business_insights.md", "text/markdown"
        )

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, scikit-learn, XGBoost & Streamlit")
