"""
Core scoring logic: loads the trained model and provides prediction + explanation functions.
"""
import joblib
import shap
import numpy as np
import pandas as pd

FEATURE_ORDER = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Load model once at import time (not per-request) for performance
model = joblib.load('models/xgboost_fraud_model.pkl')
explainer = shap.TreeExplainer(model)

# Business thresholds — will be refined in the business impact stage
DECLINE_THRESHOLD = 0.7
REVIEW_THRESHOLD = 0.3


def transaction_to_dataframe(transaction_dict: dict) -> pd.DataFrame:
    """Convert a single transaction dict into a properly-ordered DataFrame row."""
    return pd.DataFrame([transaction_dict])[FEATURE_ORDER]


def score_transaction(transaction_dict: dict) -> dict:
    """Score a single transaction: probability, SHAP explanation, business recommendation."""
    X = transaction_to_dataframe(transaction_dict)

    fraud_probability = float(model.predict_proba(X)[0, 1])

    # SHAP explanation — top 5 features by absolute contribution
    shap_values = explainer.shap_values(X)[0]
    feature_contributions = dict(zip(FEATURE_ORDER, shap_values))
    top_5 = dict(sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    top_5 = {k: float(v) for k, v in top_5.items()}

    # Business recommendation based on thresholds
    if fraud_probability >= DECLINE_THRESHOLD:
        recommendation = "decline"
    elif fraud_probability >= REVIEW_THRESHOLD:
        recommendation = "review"
    else:
        recommendation = "approve"

    return {
        "fraud_probability": fraud_probability,
        "shap_explanation": top_5,
        "business_recommendation": recommendation
    }