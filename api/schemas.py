"""
Pydantic models defining the request/response schemas for the fraud detection API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Transaction(BaseModel):
    """A single credit card transaction to be scored."""
    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")


class ScoreResponse(BaseModel):
    """Response for a single scored transaction."""
    fraud_probability: float = Field(..., description="Predicted probability of fraud, 0-1")
    conformal_interval: List[float] = Field(..., description="90% confidence interval [lower, upper]")
    shap_explanation: Dict[str, float] = Field(..., description="Top 5 features contributing to this prediction")
    business_recommendation: str = Field(..., description="approve / review / decline")
    response_time_ms: float = Field(..., description="Time taken to score this transaction, in milliseconds")


class BatchScoreRequest(BaseModel):
    """Request for scoring multiple transactions at once."""
    transactions: List[Transaction] = Field(..., max_length=1000, description="Up to 1000 transactions")


class BatchScoreResponse(BaseModel):
    """Response for batch scoring."""
    results: List[ScoreResponse]
    total_processed: int
    total_time_ms: float


class HealthResponse(BaseModel):
    """Response for the model health check endpoint."""
    status: str
    drift_status: str
    empirical_coverage: Optional[float] = None
    last_calibration_date: Optional[str] = None
    