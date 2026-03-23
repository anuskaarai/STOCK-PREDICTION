"""
Model manager — caching, saving, and loading trained models.

Each ticker gets its own directory under app/models/<TICKER>/
containing the Keras model, XGBoost models, scalers, and metadata.
"""
import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
from tensorflow import keras
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

from app.config import MODELS_DIR, MODEL_CACHE_HOURS

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """Container for all trained model artifacts for a single ticker."""
    keras_model: keras.Model
    xgb_open_dir: XGBClassifier
    xgb_close_dir: XGBClassifier
    scaler_x: MinMaxScaler
    scaler_y: MinMaxScaler
    ticker: str
    trained_at: float = field(default_factory=time.time)
    metrics: dict = field(default_factory=dict)


def _ticker_dir(ticker: str) -> Path:
    """Get the directory path for a ticker's saved models."""
    safe_name = ticker.replace(".", "_").replace("/", "_")
    return MODELS_DIR / safe_name


def save_model(artifacts: ModelArtifacts) -> None:
    """Save all model artifacts to disk."""
    d = _ticker_dir(artifacts.ticker)
    d.mkdir(parents=True, exist_ok=True)

    # Keras model
    artifacts.keras_model.save(str(d / "model.keras"))

    # XGBoost
    artifacts.xgb_open_dir.save_model(str(d / "xgb_open_dir.json"))
    artifacts.xgb_close_dir.save_model(str(d / "xgb_close_dir.json"))

    # Scalers
    joblib.dump(artifacts.scaler_x, str(d / "scaler_x.joblib"))
    joblib.dump(artifacts.scaler_y, str(d / "scaler_y.joblib"))

    # Metadata
    meta = {
        "ticker": artifacts.ticker,
        "trained_at": artifacts.trained_at,
        "metrics": artifacts.metrics,
    }
    with open(str(d / "metadata.json"), "w") as f:
        json.dump(meta, f)

    logger.info(f"Model artifacts saved for {artifacts.ticker} at {d}")


def load_model(ticker: str) -> Optional[ModelArtifacts]:
    """
    Load cached model artifacts for a ticker.
    Returns None if cache is missing or expired.
    """
    d = _ticker_dir(ticker)
    meta_path = d / "metadata.json"

    if not meta_path.exists():
        logger.info(f"No cached model for {ticker}")
        return None

    with open(str(meta_path)) as f:
        meta = json.load(f)

    # Check cache age
    age_hours = (time.time() - meta["trained_at"]) / 3600
    if age_hours > MODEL_CACHE_HOURS:
        logger.info(f"Cached model for {ticker} is {age_hours:.1f}h old (limit={MODEL_CACHE_HOURS}h) — expired")
        return None

    try:
        keras_model = keras.models.load_model(str(d / "model.keras"))

        xgb_open = XGBClassifier()
        xgb_open.load_model(str(d / "xgb_open_dir.json"))

        xgb_close = XGBClassifier()
        xgb_close.load_model(str(d / "xgb_close_dir.json"))

        scaler_x = joblib.load(str(d / "scaler_x.joblib"))
        scaler_y = joblib.load(str(d / "scaler_y.joblib"))

        logger.info(f"Loaded cached model for {ticker} (age={age_hours:.1f}h)")

        return ModelArtifacts(
            keras_model=keras_model,
            xgb_open_dir=xgb_open,
            xgb_close_dir=xgb_close,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            ticker=ticker,
            trained_at=meta["trained_at"],
            metrics=meta.get("metrics", {}),
        )
    except Exception as e:
        logger.error(f"Failed to load cached model for {ticker}: {e}")
        return None


def is_model_cached(ticker: str) -> bool:
    """Check if a fresh cached model exists for the ticker."""
    return load_model(ticker) is not None
