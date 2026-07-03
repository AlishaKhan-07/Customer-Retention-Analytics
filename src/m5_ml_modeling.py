"""
Module 5: AI/ML - Feature Engineering, Model Training & Comparison

Builds churn-prediction features, trains Logistic Regression,
Random Forest, and XGBoost, compares them on held-out test data,
and persists the best-performing model + preprocessing artifacts
for use in the Streamlit "Churn Prediction" page.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

IN_PATH = "outputs/customers_scored_final.csv"
MODEL_DIR = "models"

FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "BalanceSalaryRatio", "HasZeroBalance", "EngagementScore",
    "RetentionStrengthScore", "Geography_enc", "Gender_enc",
]
TARGET = "Exited"


def load():
    return pd.read_csv(IN_PATH)


def feature_engineering(df):
    df = df.copy()

    # Encode categoricals
    geo_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    df["Geography_enc"] = geo_encoder.fit_transform(df["Geography"])
    df["Gender_enc"] = gender_encoder.fit_transform(df["Gender"])

    encoders = {"Geography": geo_encoder, "Gender": gender_encoder}
    return df, encoders


def train_test_prepare(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def evaluate(name, model, X_test, y_test, y_pred, y_prob):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    results = []
    models = {}
    roc_data = {}

    # 1. Logistic Regression (needs scaled features)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate("Logistic Regression", lr, X_test_scaled, y_test, y_pred, y_prob))
    models["Logistic Regression"] = lr
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data["Logistic Regression"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # 2. Random Forest (raw features fine)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, class_weight="balanced", n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", rf, X_test, y_test, y_pred, y_prob))
    models["Random Forest"] = rf
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data["Random Forest"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # 3. XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight, random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", xgb, X_test, y_test, y_pred, y_prob))
    models["XGBoost"] = xgb
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data["XGBoost"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    return results, models, roc_data


def select_best(results, models):
    # Select by ROC-AUC (robust to class imbalance), tie-break on F1
    best = sorted(results, key=lambda r: (r["roc_auc"], r["f1_score"]), reverse=True)[0]
    best_name = best["model"]
    return best_name, models[best_name], best


def run(save=True):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load()
    df, encoders = feature_engineering(df)

    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = train_test_prepare(df)
    results, models, roc_data = train_models(X_train, X_test, y_train, y_test, X_train_s, X_test_s)
    best_name, best_model, best_metrics = select_best(results, models)

    # Feature importance from tree-based best model (or coef for LR)
    if best_name in ("Random Forest", "XGBoost"):
        importances = dict(zip(FEATURES, best_model.feature_importances_.round(4).tolist()))
    else:
        importances = dict(zip(FEATURES, np.abs(best_model.coef_[0]).round(4).tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    comparison = {
        "results": results,
        "best_model": best_name,
        "best_model_metrics": best_metrics,
        "feature_importance": importances,
        "test_set_size": int(len(y_test)),
        "train_set_size": int(len(y_train)),
    }

    if save:
        with open("outputs/model_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        with open("outputs/roc_curve_data.json", "w") as f:
            json.dump(roc_data, f, indent=2)

        # Persist best model + preprocessing artifacts for the Streamlit app
        joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
        joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
        joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
        joblib.dump(best_name, f"{MODEL_DIR}/best_model_name.pkl")
        joblib.dump(FEATURES, f"{MODEL_DIR}/feature_list.pkl")

        # Also save all 3 models so the app can optionally let a user compare
        for name, m in models.items():
            fname = name.lower().replace(" ", "_")
            joblib.dump(m, f"{MODEL_DIR}/{fname}.pkl")

        print("Saved -> outputs/model_comparison.json")
        print("Saved -> outputs/roc_curve_data.json")
        print(f"Saved best model ({best_name}) -> {MODEL_DIR}/best_model.pkl")

    return comparison, models, scaler, encoders


if __name__ == "__main__":
    comparison, models, scaler, encoders = run()
    print(json.dumps(comparison["results"], indent=2))
    print(f"\nBest model: {comparison['best_model']}")
    print(json.dumps(comparison["best_model_metrics"], indent=2))
