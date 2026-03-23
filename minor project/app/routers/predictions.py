from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
import asyncio
import logging

from app.schemas.prediction import PredictionRequest, PredictionResponse, ErrorResponse
from app.services.predictor import predict_for_ticker
from app.config import DEFAULT_FORECAST_HOURS

router = APIRouter(
    prefix="/predict",
    tags=["Predictions"]
)
logger = logging.getLogger(__name__)

@router.post("/", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
async def create_prediction(request: PredictionRequest, req: Request):
    """
    Generate hourly price predictions for a given stock ticker.
    
    If a cached model is available and `force_retrain` is false, it will use the cache.
    Otherwise, it fetches the last 60d of hourly data, trains the BiLSTM+XGBoost ensemble,
    caches it, and generates predictions.
    """
    try:
        semaphore = req.app.state.training_semaphore
        async with semaphore:
            result = await asyncio.to_thread(
                predict_for_ticker,
                request.ticker,
                request.forecast_hours,
                request.force_retrain
            )
        return result
    except ValueError as e:
        logger.error(f"Validation error for {request.ticker}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error predicting {request.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/{ticker}", response_model=PredictionResponse)
async def quick_predict(ticker: str, req: Request, forecast: int = DEFAULT_FORECAST_HOURS):
    """
    Quick prediction endpoint using query parameters.
    """
    try:
        semaphore = req.app.state.training_semaphore
        async with semaphore:
            result = await asyncio.to_thread(
                predict_for_ticker,
                ticker,
                forecast,
                False
            )
        return result
    except ValueError as e:
        logger.error(f"Validation error for {ticker}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error predicting {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

