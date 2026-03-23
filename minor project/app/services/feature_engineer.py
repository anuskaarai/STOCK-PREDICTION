"""
Feature engineering service.

Replicates the exact feature engineering pipeline from 18 hourly.ipynb (v4).
Produces 37 feature columns from raw OHLCV data using the `ta` library.
"""
import numpy as np
import pandas as pd
import logging
import ta

from app.config import FEATURE_COLS

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline from the notebook (v4).

    Args:
        df: Raw OHLCV DataFrame with columns: Open, High, Low, Close, Volume

    Returns:
        DataFrame with engineered feature columns, NaN rows dropped.
    """
    logger.info("Engineering features (v4)...")
    df = df.copy()

    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    open_  = df["Open"].squeeze()

    # ── Core returns & Price structure ──────────────────────────────
    df["log_return"]      = np.log(close  / close.shift(1))
    df["open_log_return"] = np.log(open_  / close.shift(1))
    df["HL_pct"]          = (high - low)   / close
    df["OC_pct"]          = (close - open_) / open_

    # ── Lagged returns ──────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"ret_lag{lag}"] = df["log_return"].shift(lag)

    # ── Rolling return stats ────────────────────────────────────────
    df["ret_mean_6"]  = df["log_return"].rolling(6).mean()
    df["ret_mean_12"] = df["log_return"].rolling(12).mean()
    df["ret_std_6"]   = df["log_return"].rolling(6).std()
    df["ret_std_12"]  = df["log_return"].rolling(12).std()

    # ── Momentum ────────────────────────────────────────────────────
    df["RSI_14"]  = ta.momentum.rsi(close, window=14)
    df["RSI_6"]   = ta.momentum.rsi(close, window=6)
    stoch         = ta.momentum.StochasticOscillator(high, low, close)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    df["ROC_5"]   = ta.momentum.roc(close, window=5)
    df["ROC_10"]  = ta.momentum.roc(close, window=10)

    # ── Trend ───────────────────────────────────────────────────────
    df["SMA_10_ratio"] = close / ta.trend.sma_indicator(close, window=10)
    df["SMA_20_ratio"] = close / ta.trend.sma_indicator(close, window=20)
    df["EMA_10_ratio"] = close / ta.trend.ema_indicator(close, window=10)

    macd = ta.trend.MACD(close)
    df["MACD_diff"]  = macd.macd_diff()
    df["MACD_cross"] = (macd.macd() > macd.macd_signal()).astype(int)

    # ── Volatility ──────────────────────────────────────────────────
    bb            = ta.volatility.BollingerBands(close)
    df["BB_pct"]  = bb.bollinger_pband()
    df["BB_width"]= (bb.bollinger_hband() - bb.bollinger_lband()) / close
    df["ATR_pct"] = (ta.volatility.AverageTrueRange(high, low, close).average_true_range() / close)

    # ── Volume ──────────────────────────────────────────────────────
    df["vol_ratio"] = volume / volume.rolling(20).mean()
    df["vol_log"]   = np.log1p(volume)

    obv_raw        = ta.volume.on_balance_volume(close, volume)
    obv_med        = obv_raw.rolling(50).median().replace(0, np.nan)
    df["OBV_norm"] = obv_raw / obv_med

    # ── Time (cyclical) ─────────────────────────────────────────────
    # Ensure index is datetime for hour/dayofweek extraction
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    df["hour_sin"]      = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]       = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["dow_cos"]       = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df["is_first_hour"] = (df.index.hour == 9).astype(int)
    df["is_last_hour"]  = (df.index.hour == 15).astype(int)

    # ── Drop NaN rows ───────────────────────────────────────────────
    initial_len = len(df)
    df.dropna(subset=FEATURE_COLS, inplace=True)
    dropped = initial_len - len(df)
    logger.info(
        f"Feature engineering complete: {len(df)} rows "
        f"(dropped {dropped} NaN rows), {len(FEATURE_COLS)} features"
    )

    return df

