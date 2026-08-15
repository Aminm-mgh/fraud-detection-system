"""
FastAPI backend for the real-time fraud detection system.
"""
import time
from fastapi import FastAPI, HTTPException
from api.schemas import (
    Transaction, ScoreResponse, BatchScoreRequest, BatchScoreResponse, HealthResponse
)
from api.scoring import score_transaction

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with explainability and business recommendations",
    version="1.0.0"
)


@app.post("/score", response_model=ScoreResponse)
def score(transaction: Transaction):
    """Score a single transaction and return probability, explanation, and recommendation."""
    start = time.time()

    try:
        result = score_transaction(transaction.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scoring failed: {str(e)}")

    elapsed_ms = (time.time() - start) * 1000

    return ScoreResponse(
        fraud_probability=result["fraud_probability"],
        conformal_interval=[0.0, 1.0],  # placeholder — wired up properly in a later step
        shap_explanation=result["shap_explanation"],
        business_recommendation=result["business_recommendation"],
        response_time_ms=elapsed_ms
    )


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest):
    """Score up to 1000 transactions in a single request."""
    start = time.time()

    results = []
    for txn in request.transactions:
        result = score_transaction(txn.model_dump())
        results.append(ScoreResponse(
            fraud_probability=result["fraud_probability"],
            conformal_interval=[0.0, 1.0],  # placeholder
            shap_explanation=result["shap_explanation"],
            business_recommendation=result["business_recommendation"],
            response_time_ms=0.0  # per-item timing omitted in batch mode
        ))

    total_elapsed_ms = (time.time() - start) * 1000

    return BatchScoreResponse(
        results=results,
        total_processed=len(results),
        total_time_ms=total_elapsed_ms
    )


@app.get("/model/health", response_model=HealthResponse)
def health():
    """Return current model health status."""
    return HealthResponse(
        status="ok",
        drift_status="ALERT",  # placeholder — wired up properly in a later step
        empirical_coverage=0.899,
        last_calibration_date="2026-08-14"
    )


@app.get("/")
def root():
    """Simple root endpoint to confirm the API is running."""
    return {"message": "Fraud Detection API is running. Visit /docs for interactive documentation."}