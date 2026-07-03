"""
Master pipeline runner.
Executes all analytics + ML modules in order and regenerates every
output artifact consumed by the Streamlit dashboard.

Usage: python run_pipeline.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import m1_preprocessing
import m2_eda_engagement
import m3_product_highvalue
import m4_retention_scoring
import m5_ml_modeling
import m6_business_insights


def main():
    t0 = time.time()
    print("=" * 60)
    print("CUSTOMER RETENTION & CHURN PREDICTION ANALYTICS PIPELINE")
    print("=" * 60)

    print("\n[1/6] Data Preprocessing...")
    m1_preprocessing.run()

    print("\n[2/6] EDA + Customer Engagement Analysis...")
    m2_eda_engagement.run()

    print("\n[3/6] Product Utilization + High-Value Disengaged Customers...")
    m3_product_highvalue.run()

    print("\n[4/6] Retention Strength Scoring...")
    m4_retention_scoring.run()

    print("\n[5/6] AI/ML - Feature Engineering & Model Training...")
    m5_ml_modeling.run()

    print("\n[6/6] Business Insights & Recommendations...")
    m6_business_insights.run()

    print(f"\nPipeline complete in {time.time() - t0:.1f}s")
    print("All outputs written to ./outputs and trained models to ./models")


if __name__ == "__main__":
    main()
