"""
Data fetching service using yfinance.

Downloads hourly OHLCV data for any given stock ticker.
"""
import yfinance as yf
import pandas as pd
import logging

from app.config import DATA_PERIOD, DATA_INTERVAL

logger = logging.getLogger(__name__)


def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """
    Fetch hourly OHLCV data for the given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., "TCS.NS", "AAPL")

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index is datetime (timezone-aware removed).

    Raises:
        ValueError: If ticker is invalid or returns no data.
    """
    logger.info(f"Fetching data for {ticker} (period={DATA_PERIOD}, interval={DATA_INTERVAL})")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=DATA_PERIOD, interval=DATA_INTERVAL)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for ticker '{ticker}': {e}")

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Please verify the ticker symbol is correct."
        )

    # Clean up: keep only OHLCV, remove timezone info for consistency
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in data for '{ticker}'.")

    df = df[required_cols].copy()

    # Remove timezone info to avoid issues downstream
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Drop any rows with NaN in OHLCV
    df.dropna(subset=required_cols, inplace=True)

    if len(df) < 50:
        raise ValueError(
            f"Insufficient data for ticker '{ticker}' "
            f"(got {len(df)} rows, need at least 50)."
        )

    logger.info(f"Fetched {len(df)} rows for {ticker}")
    return df
