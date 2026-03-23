from pydantic import BaseModel, Field
from typing import List, Optional

from app.config import DEFAULT_FORECAST_HOURS

class PredictionResult(BaseModel):
    datetime: str
    pred_open: float
    pred_close: float
    open_direction: str
    close_direction: str
    high_confidence: bool

class HistoricalResult(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float

class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g., 'TCS.NS' or 'AAPL'")
    forecast_hours: int = Field(default=DEFAULT_FORECAST_HOURS, ge=1, le=100, description="Number of trading hours to forecast")
    force_retrain: bool = Field(default=False, description="Force retraining the model even if cached")

class PredictionResponse(BaseModel):
    ticker: str
    forecast_hours: int
    historical: List[HistoricalResult] = Field(default_factory=list)
    predictions: List[PredictionResult]
    metrics: dict
    status: str

class ErrorResponse(BaseModel):
    detail: str
