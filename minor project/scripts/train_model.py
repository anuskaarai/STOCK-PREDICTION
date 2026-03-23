#!/usr/bin/env python3
"""
Standalone script to train and cache a model for a specific ticker.
"""
import argparse
import logging
import sys

from app.services.predictor import predict_for_ticker

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_script")

def main():
    parser = argparse.ArgumentParser(description="Train and cache a stock prediction model.")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., TCS.NS)")
    parser.add_argument("--force", action="store_true", help="Force retrain even if cached")
    
    args = parser.parse_args()
    
    logger.info(f"Starting script for ticker: {args.ticker}")
    
    try:
        # predict_for_ticker naturally handles the data fetch, train, and forecast.
        # It's an easy way to trigger a full training run and see the output.
        result = predict_for_ticker(args.ticker, forecast_hours=1, force_retrain=args.force)
        
        logger.info(f"Successfully trained model for {args.ticker}!")
        logger.info(f"Metrics: {result['metrics']}")
        
    except Exception as e:
        logger.error(f"Failed to train model for {args.ticker}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
