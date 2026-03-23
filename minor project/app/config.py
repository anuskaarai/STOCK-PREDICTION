"""
Application configuration and constants.

Replicates the exact hyperparameters and feature columns
from the 18 hourly.ipynb (v4) notebook.
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────
DATA_PERIOD = "60d"  # Reduced to 60d for fast prediction. (v4 uses 730d max available)
DATA_INTERVAL = "1h"

# ── Model Architecture ───────────────────────────────────────────────
LOOKBACK = 24          # sequence length (hours)
LSTM_UNITS_1 = 128     # first BiLSTM units
LSTM_UNITS_2 = 64      # second BiLSTM units
ATTENTION_HEADS = 4
ATTENTION_KEY_DIM = 24
DENSE_UNITS = 64
DROPOUT_LSTM = 0.2
DROPOUT_DENSE = 0.15

# ── Training ─────────────────────────────────────────────────────────
EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 20          # v4 early stopping patience
TRAIN_SPLIT = 0.8      # 80% train, 20% test
LEARNING_RATE = 1e-3

# Loss weights (v4 config)
LOSS_WEIGHT_RETURN    = 2.0
LOSS_WEIGHT_OPEN_DIR  = 1.0
LOSS_WEIGHT_CLOSE_DIR = 1.0

# ── Target scaling ───────────────────────────────────────────────────
RETURN_SCALE = 100.0   # Scale targets to percent space

# ── Forecast ─────────────────────────────────────────────────────────
DEFAULT_FORECAST_HOURS = 18   # v4 forecast hours

# ── Model cache ──────────────────────────────────────────────────────
MODEL_CACHE_HOURS = 24

# ── Feature columns ──────────────────────────────────────────────────
FEATURE_COLS = [
    "log_return", "open_log_return", "HL_pct", "OC_pct",
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag6", "ret_lag12", "ret_lag24",
    "ret_mean_6", "ret_mean_12", "ret_std_6", "ret_std_12",
    "RSI_14", "RSI_6",
    "Stoch_K", "Stoch_D",
    "ROC_5", "ROC_10",
    "SMA_10_ratio", "SMA_20_ratio", "EMA_10_ratio",
    "MACD_diff", "MACD_cross",
    "BB_pct", "BB_width",
    "ATR_pct",
    "vol_ratio", "vol_log",
    "OBV_norm",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_first_hour", "is_last_hour"
]
