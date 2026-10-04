# Customer Retention & Churn Prediction Analytics

An end-to-end analytics + AI/ML project on a European bank's customer dataset (10,000 customers).

## Structure

```
churn_project/
├── data/
│   └── European_Bank.csv          # raw source data
├── src/
│   ├── m1_preprocessing.py        # cleaning + feature engineering
│   ├── m2_eda_engagement.py       # EDA + customer engagement analysis
│   ├── m3_product_highvalue.py    # product utilization + high-value disengaged detection
│   ├── m4_retention_scoring.py    # rule-based retention strength score
│   ├── m5_ml_modeling.py          # feature engineering + LR/RF/XGBoost training & comparison
│   └── m6_business_insights.py    # insight + recommendation synthesis
├── models/                        # trained model artifacts (generated)
├── outputs/                       # all generated CSVs/JSON (generated)
├── run_pipeline.py                # runs all 6 modules end-to-end
├── app.py                         # Streamlit dashboard (8 pages)
├── requirements.txt
└── README.md
```

## Part 1: Data Analytics
- **Preprocessing** – duplicate/range validation, engineered fields (age/tenure bands, value tiers)
- **EDA** – distributions, churn correlation, geography/gender/age cuts
- **Customer engagement analysis** – composite engagement score, activity status, tenure trends
- **Product utilization analysis** – product-count churn patterns, cross-sell opportunity sizing
- **High-value disengaged customer identification** – top-value customers with disengagement signals
- **Retention strength scoring** – transparent, weighted 0–100 score with Critical/At Risk/Stable/Strong bands
- **Business insights and recommendations** – auto-generated findings + prioritized action list

## Part 2: AI/ML (Value Addition)
- **Feature engineering** – encodes categoricals, builds engagement/retention/ratio features
- **Train-test split** – 80/20 stratified split
- **Multiple models** – Logistic Regression, Random Forest, XGBoost
- **Model comparison** – accuracy, precision, recall, F1, ROC-AUC on held-out test data
- **Best model selection** – by ROC-AUC (tie-break F1), currently XGBoost
- **Streamlit integration** – interactive single-customer prediction with probability gauge

## Running the project

```bash
pip install -r requirements.txt
python run_pipeline.py       # regenerates all outputs/ and models/
streamlit run app.py         # launches the 8-page dashboard
```

## Dashboard pages
1. Home – top-line KPIs and headline charts
2. Dataset Overview – raw data, summary stats, churn correlations
3. Customer Engagement Analytics – engagement score, activity, tenure trends
4. Product Utilization Analytics – product-count churn risk, cross-sell sizing
5. High-Value Customer Detector – filterable list of high-value disengaged customers
6. Retention Strength Score – score distribution, band churn rates, priority list
7. Churn Prediction (AI/ML) – model comparison + live single-customer prediction
8. Business Recommendations – synthesized insights and prioritized actions

## Deployed project - [LINK](https://customer-retention-analyticsgit-qtwf53wqtowghsmffmz3x7.streamlit.app/#customer-retention-and-churn-prediction-analytics)

## Notes
- Churn rate 3–4 products (~83–100%) is a known signature of this classic bank-churn dataset and
  is flagged in the app as an over-selling/service-issue red flag rather than a loyalty signal.
- The Retention Strength Score is intentionally rule-based (not the ML model) so business users can
  trust and act on it without needing to interpret a model.
