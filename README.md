# Real-Time Credit Card Fraud Detection System

A production-oriented fraud detection system that goes beyond a probability score — combining imbalanced classification, model explainability (SHAP), calibrated uncertainty (conformal prediction), and drift monitoring, exposed through a FastAPI backend.

**Core thesis:** A fraud detection model that only outputs a probability is incomplete. A production-grade system returns a probability + a human-readable explanation + a confidence interval + a business impact estimate — and monitors itself over time to flag when it's no longer trustworthy.

## Live Demo
*(Coming soon — Streamlit dashboard + hosted API)*

---

## The Problem

Credit card fraud is a severe class imbalance problem: roughly **0.17% of transactions are fraudulent**. A model that predicts "legitimate" for everything achieves 99.83% accuracy while catching zero fraud — which is why this project treats accuracy as a misleading metric throughout, and uses **Precision-Recall AUC** and **business cost** as the real evaluation criteria.

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European transactions (Sept 2013), 492 fraud cases, 30 anonymized PCA features (`V1`–`V28`) + `Amount` + `Time`.

---

## Methodology & Key Findings

### 1. Data Pipeline & EDA
- Identified and removed **1,081 duplicate rows** *before* splitting — a common source of data leakage in tutorial-level fraud projects. This reduced fraud cases from 492 → 473.
- Caught and fixed a subtle bug: the `Class` label column was parsed as a quoted string (`'0'`, `'1'`) rather than an integer, which would have silently broken stratified splitting and `scale_pos_weight` calculations downstream.
- Found fraud transactions follow a **bimodal amount pattern** (median £9.82 vs £22.00 for legitimate, but higher mean £123.87 vs £88.41) — consistent with card-testing behavior (many small probing transactions) mixed with occasional larger fraud.
- Linear correlation ranked `V17`, `V14`, `V12` as top signals — later shown by SHAP to be an incomplete picture (see below).

### 2. Imbalance Handling
Compared three strategies on a Logistic Regression baseline, evaluated at matched recall (~90%) where possible:

| Strategy | Fraud Recall | Fraud Precision | False Positives |
|---|---|---|---|
| Baseline (no handling) | 61.05% | 86.57% | 9 |
| SMOTE | 89.47% | 5.71% | 1,403 |
| **Class Weighting** | **90.53%** | **6.19%** | **1,304** |
| Threshold tuning alone | 90.53% | 2.26% | 3,711 |

**Finding:** Class weighting slightly outperformed SMOTE while avoiding synthetic data generation on anonymized PCA features. Threshold tuning alone — without addressing imbalance during training — performed worst, showing that moving the decision boundary post-hoc is not a substitute for learning a better-separated boundary in the first place.

### 3. Model Comparison
| Model | PR-AUC |
|---|---|
| Logistic Regression | 0.7309 |
| **XGBoost (selected)** | **0.8568** |
| LightGBM | Excluded — reproducible Apple Silicon wheel issue (documented in `03_model_comparison.ipynb`) |
| Isolation Forest (unsupervised) | 0.1507 |

XGBoost was selected as the primary model. Isolation Forest's much lower unsupervised score confirms fraud in this dataset isn't purely a structural anomaly — label information is essential.

### 4. Explainability (SHAP + DiCE)
- **Global:** `V14`, `V4`, `V12`, `V10` are the top SHAP-ranked features — notably, `V17` (the strongest *linear* correlate from EDA) ranked much lower in SHAP importance, demonstrating non-linear effects that Pearson correlation cannot capture.
- **Local:** Waterfall plots explain individual predictions — e.g. for one representative fraud case, `V14` alone contributed +7.28 to the log-odds score, over 4x the next two features combined.
- **Interactions:** `V14` is the dominant interaction hub, appearing in 6 of the top 10 strongest feature-pair interactions — explaining why tree-based XGBoost meaningfully outperforms linear Logistic Regression.
- **Counterfactuals (DiCE):** For the same example transaction, lowering `V4` alone (from 2.07 to below ~0.6) was sufficient to flip the prediction from fraud to legitimate — independently confirming SHAP's attribution via a completely different method.

### 5. Conformal Prediction (MAPIE)
- Wrapped XGBoost with `SplitConformalClassifier`, calibrated on a held-out 56,745-transaction calibration set.
- **Empirical coverage: 89.90%** against a 90% target — validates the statistical guarantee holds on real data.
- At both 90% and 99% confidence, **0% of test transactions** were flagged as high-uncertainty, reflecting XGBoost's very sharp decision boundary on this benchmark dataset. This is a real limitation worth noting: less cleanly-separated production data would likely trigger more human-review flags.

### 6. Drift Monitoring
- Simulated realistic drift via a **chronological** (not random) train/test split — model trained on the first 70% of transactions by time, evaluated on the last 30%.
- **PR-AUC dropped from 0.8568 → 0.7929** (~7.5% relative decrease) on genuinely unseen future data, versus the random-split evaluation.
- **PSI analysis** flagged `V1`, `V3`, `V28`, `V11` as significantly drifted (PSI > 0.25) — while `V14`, the model's most important feature, remained stable (PSI = 0.070), likely explaining why degradation was moderate rather than severe.
- **Evidently AI** independently confirmed drift using a different statistical methodology: 18 of 30 columns (60%) flagged as drifted — validating the manual PSI implementation.
- Built a reusable `check_drift_alert()` function returning OK / WARNING / ALERT status, feeding directly into the API's health endpoint.

### 7. FastAPI Backend
- `POST /score` — single transaction scoring: probability, SHAP top-5 explanation, business recommendation, response time
- `POST /score/batch` — up to 1,000 transactions per request
- `GET /model/health` — drift status and calibration health
- Typed, validated request/response schemas via Pydantic; auto-generated interactive docs at `/docs`
- Verified response time: **~9ms per transaction**, with SHAP values matching offline notebook analysis exactly

---

## Tech Stack
Python · pandas · scikit-learn · XGBoost · SHAP · DiCE · MAPIE · Evidently AI · FastAPI · Pydantic · Streamlit *(in progress)*

## Project Structure
fraud-detection-system/
├── data/ # raw (gitignored) + processed Parquet splits
├── notebooks/ # 01 EDA → 06 drift simulation
├── src/ # reusable training scripts
├── api/ # FastAPI app, schemas, scoring logic
├── models/ # saved model artifacts
├── reports/ # generated drift reports
└── dashboard/ # Streamlit app (in progress)



## Status
✅ Layers 1–6 complete (EDA, imbalance handling, modeling, explainability, conformal prediction, drift monitoring, FastAPI backend)
🔄 In progress: business impact threshold calibration, Streamlit dashboard, tests, CI/CD, Docker, deployment

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download creditcard.csv from Kaggle into data/raw/
python src/train_final_model.py
uvicorn api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.