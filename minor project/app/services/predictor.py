"""
Core prediction orchestrator.

Handles combining data fetching, feature engineering,
model training (if needed), and generating predictions.
Replicates v4 ML logic.
"""
import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any

from app.config import (
    LOOKBACK, FEATURE_COLS, TRAIN_SPLIT,
    EPOCHS, BATCH_SIZE, PATIENCE, RETURN_SCALE
)
from app.services.data_fetcher import fetch_stock_data
from app.services.feature_engineer import engineer_features
from app.services.model_builder import build_model
from app.services.model_manager import (
    ModelArtifacts, is_model_cached, load_model, save_model
)
from tensorflow import keras

logger = logging.getLogger(__name__)


def train_models_for_ticker(ticker: str, df: pd.DataFrame) -> ModelArtifacts:
    """Train the multi-head BiLSTM and XGBoost models for a given ticker."""
    logger.info(f"Starting training pipeline for {ticker}")

    # Prepare data
    X_raw = df[FEATURE_COLS].values
    
    # Target in percent space matching v4 notebook
    y_returns = df[["open_log_return", "log_return"]].values * RETURN_SCALE

    # Compute training standard deviation for clipping bounds
    train_split_idx = int(len(X_raw) * TRAIN_SPLIT)
    train_ret_std_open = float(np.std(y_returns[:train_split_idx, 0]))
    train_ret_std_close = float(np.std(y_returns[:train_split_idx, 1]))
    logger.info(f"Target std deviations — Open: {train_ret_std_open:.4f}, Close: {train_ret_std_close:.4f}")

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    # Create sequences
    n_samples = len(X_raw) - LOOKBACK
    X_seq = np.zeros((n_samples, LOOKBACK, len(FEATURE_COLS)))
    y_ret_seq = np.zeros((n_samples, 2))
    y_dir_seq = np.zeros((n_samples, 2))

    for i in range(n_samples):
        window = X_raw[i: i + LOOKBACK]
        X_seq[i] = window

        # Target is the next step
        next_ret = y_returns[i + LOOKBACK]
        y_ret_seq[i] = next_ret

        # Direction: 1 if return > 0 else 0
        y_dir_seq[i, 0] = int(next_ret[0] > 0)
        y_dir_seq[i, 1] = int(next_ret[1] > 0)

    # Train/Val split
    split_idx = int(n_samples * TRAIN_SPLIT)

    # Fit scalers ONLY on training data to prevent leakage
    scaler_X.fit(X_raw[:split_idx + LOOKBACK])
    scaler_y.fit(y_returns[:split_idx + LOOKBACK])

    # Scale sequences
    for i in range(n_samples):
        X_seq[i] = scaler_X.transform(X_seq[i])

    y_ret_scaled = scaler_y.transform(y_ret_seq)

    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_ret_train, y_ret_val = y_ret_scaled[:split_idx], y_ret_scaled[split_idx:]
    y_dir_train, y_dir_val = y_dir_seq[:split_idx], y_dir_seq[split_idx:]

    # Build & Train Keras Model
    keras_model = build_model(LOOKBACK, len(FEATURE_COLS))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True
        )
    ]

    logger.info("Training Keras model...")
    history = keras_model.fit(
        X_train,
        {
            "return_output": y_ret_train,
            "open_dir_output": y_dir_train[:, 0],
            "close_dir_output": y_dir_train[:, 1],
        },
        validation_data=(
            X_val,
            {
                "return_output": y_ret_val,
                "open_dir_output": y_dir_val[:, 0],
                "close_dir_output": y_dir_val[:, 1],
            }
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )

    best_epoch = int(np.argmin(history.history["val_loss"]))
    metrics = {
        "val_loss": history.history["val_loss"][best_epoch],
        "val_open_dir_accuracy": history.history["val_open_dir_output_accuracy"][best_epoch],
        "val_close_dir_accuracy": history.history["val_close_dir_output_accuracy"][best_epoch],
        "train_ret_std_open": train_ret_std_open,
        "train_ret_std_close": train_ret_std_close,
    }
    logger.info(f"Keras training complete. Best Val Loss (Epoch {best_epoch+1}): {metrics['val_loss']:.4f}")

    # Train XGBoost Models
    logger.info("Training XGBoost ensemble...")
    # XGBoost trains on just the tabular features at time T to predict T+1
    X_raw_scaled = scaler_X.transform(X_raw)
    X_train_xgb = X_raw_scaled[LOOKBACK: LOOKBACK + split_idx]
    
    xgb_params = dict(
        n_estimators=1000, learning_rate=0.01, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.05,
        random_state=42
    )

    xgb_open = XGBClassifier(**xgb_params)
    xgb_open.fit(X_train_xgb, y_dir_train[:, 0])

    xgb_close = XGBClassifier(**xgb_params)
    xgb_close.fit(X_train_xgb, y_dir_train[:, 1])

    artifacts = ModelArtifacts(
        keras_model=keras_model,
        xgb_open_dir=xgb_open,
        xgb_close_dir=xgb_close,
        scaler_x=scaler_X,
        scaler_y=scaler_y,
        ticker=ticker,
        metrics=metrics
    )

    # Save to disk
    save_model(artifacts)
    return artifacts


def generate_forecast(
    raw_df: pd.DataFrame,
    engineered_df: pd.DataFrame, 
    artifacts: ModelArtifacts, 
    forecast_hours: int
) -> list[Dict[str, Any]]:
    """Iteratively forecast future prices using the v4 sliding buffer methodology."""
    logger.info(f"Generating {forecast_hours}-hour forecast for {artifacts.ticker}...")

    # Initial state limits
    BUFFER_SIZE = 70
    buf = raw_df[["Open", "High", "Low", "Close", "Volume"]].iloc[-BUFFER_SIZE:].copy()
    
    # We use engineered_df essentially just to initialize tracking variables and seed sequence
    X_raw = engineered_df[FEATURE_COLS].values
    X_scaled = artifacts.scaler_x.transform(X_raw)
    current_seq = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(FEATURE_COLS))
    
    last_atr_pct = float(engineered_df["ATR_pct"].iloc[-1])
    last_close = float(raw_df["Close"].iloc[-1])
    last_time = raw_df.index[-1]
    
    model = artifacts.keras_model
    xgb_open = artifacts.xgb_open_dir
    xgb_close = artifacts.xgb_close_dir
    scaler_x = artifacts.scaler_x
    scaler_y = artifacts.scaler_y

    train_ret_std_open = artifacts.metrics.get("train_ret_std_open", 1.0)
    train_ret_std_close = artifacts.metrics.get("train_ret_std_close", 1.0)
    
    records = []

    while len(records) < forecast_hours:
        last_time += pd.Timedelta(hours=1)

        # Indian market hours 9:15 - 15:30 (Monday to Friday)
        if last_time.weekday() >= 5:
            continue
        if last_time.hour < 9 or (last_time.hour == 9 and last_time.minute < 15):
            continue
        if last_time.hour > 15 or (last_time.hour == 15 and last_time.minute > 30):
            continue

        # Inference from LSTM
        ret_pred_s, open_dir_pred, close_dir_pred = model.predict(current_seq, verbose=0)
        ret_pred = scaler_y.inverse_transform(ret_pred_s)[0]
        
        # XGBoost inference
        current_tab = current_seq[0, -1, :].reshape(1, -1)
        xgb_open_d = int(xgb_open.predict(current_tab)[0])
        xgb_close_d = int(xgb_close.predict(current_tab)[0])

        # Direction confidences
        lstm_open_conf = float(open_dir_pred.squeeze())
        lstm_close_conf = float(close_dir_pred.squeeze())
        
        open_dir_agree = int(xgb_open_d) == int(lstm_open_conf > 0.5)
        close_dir_agree = int(xgb_close_d) == int(lstm_close_conf > 0.5)

        # Returns in percent space
        open_ret_pct = float(ret_pred[0])
        close_ret_pct = float(ret_pred[1])

        # ATR Noise-floor replacement
        atr_floor_pct = last_atr_pct * 100 * 0.3
        if abs(open_ret_pct) < atr_floor_pct:
            sign = 1 if xgb_open_d == 1 else -1
            open_ret_pct = sign * last_atr_pct * 100 * 0.5

        if abs(close_ret_pct) < atr_floor_pct:
            sign = 1 if xgb_close_d == 1 else -1
            close_ret_pct = sign * last_atr_pct * 100 * 0.5

        # Clip predictions to ±3 std deviations
        open_ret_pct = float(np.clip(open_ret_pct, -3 * train_ret_std_open, 3 * train_ret_std_open))
        close_ret_pct = float(np.clip(close_ret_pct, -3 * train_ret_std_close, 3 * train_ret_std_close))

        # Reconstruct Absolute prices
        pred_open  = float(last_close * np.exp(open_ret_pct / RETURN_SCALE))
        pred_close = float(last_close * np.exp(close_ret_pct / RETURN_SCALE))
        pred_high  = float(max(pred_open, pred_close) * (1 + last_atr_pct * 0.5))
        pred_low   = float(min(pred_open, pred_close) * (1 - last_atr_pct * 0.5))
        avg_vol    = float(buf["Volume"].iloc[-5:].mean())

        records.append({
            "datetime": last_time.isoformat(),
            "pred_open": float(round(pred_open, 2)),
            "pred_close": float(round(pred_close, 2)),
            "open_direction": "UP" if xgb_open_d == 1 else "DOWN",
            "close_direction": "UP" if xgb_close_d == 1 else "DOWN",
            "high_confidence": bool(open_dir_agree and close_dir_agree)
        })

        # Slide sequence using engineered dataframe buffer logic
        last_close = pred_close
        
        new_row = pd.DataFrame([{
            "Open": pred_open, "High": pred_high, 
            "Low": pred_low, "Close": pred_close, "Volume": avg_vol
        }], index=[last_time])
        
        buf = pd.concat([buf, new_row]).iloc[-BUFFER_SIZE:]
        
        # We re-engineer features on `buf` directly
        next_features_df = engineer_features(buf)
        next_feat_vec = next_features_df[FEATURE_COLS].iloc[-1].values
        
        last_atr_pct = float(next_features_df["ATR_pct"].iloc[-1])
        next_scaled = scaler_x.transform(next_feat_vec.reshape(1, -1))
        
        current_seq = np.append(current_seq[:, 1:, :], next_scaled.reshape(1, 1, -1), axis=1)

    return records


def predict_for_ticker(ticker: str, forecast_hours: int, force_retrain: bool = False) -> Dict[str, Any]:
    """Orchestrate data fetch, optional train, and forecast."""
    logger.info(f"Prediction requested for {ticker}")

    # 1. Fetch live data & engineer features
    raw_df = fetch_stock_data(ticker)
    engineered_df = engineer_features(raw_df)

    # 2. Get model (load or train)
    artifacts = None
    if not force_retrain:
        artifacts = load_model(ticker)
    
    if artifacts is None:
        logger.info(f"Training required for {ticker}. This may take a few minutes.")
        artifacts = train_models_for_ticker(ticker, engineered_df)
    else:
        logger.info(f"Using cached model for {ticker}")

    # 3. Forecast
    predictions = generate_forecast(raw_df, engineered_df, artifacts, forecast_hours)

    # 4. Extract Historical Data (last 100 rows for better context)
    historical_df = engineered_df.tail(100)
    historical_data = []
    
    # Needs accessing original price matching indices
    raw_hist_df = raw_df.loc[historical_df.index]
    for timestamp, row in raw_hist_df.iterrows():
        historical_data.append({
            "datetime": timestamp.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"])
        })

    return {
        "ticker": ticker,
        "forecast_hours": forecast_hours,
        "historical": historical_data,
        "predictions": predictions,
        "metrics": artifacts.metrics,
        "status": "success"
    }
