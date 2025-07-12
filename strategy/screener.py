# refactored_fyers_swing/strategy/screener.py

import os
import pandas as pd
from datetime import datetime
from config import DAILY_DATA_FILE, MODEL_FILE,FEATURE_COLUMNS
from data.fetch_stock_list import load_symbols
from features.engineer_features import add_technical_indicators
from model.prediction import load_model, predict
from strategy.signal_generator import generate_signals
from utils.logger import get_logger

logger = get_logger(__name__)

def run_screener():
    """
    Main screener function that loads today's data, predicts targets,
    applies filters, and saves bullish stock suggestions.
    """
    try:
        logger.info("=== Starting Daily Screener ===")

        # Load today's data (assumed pre-downloaded in DAILY_DATA_FILE)
        df_today = pd.read_csv(DAILY_DATA_FILE, parse_dates=["date"])
        logger.info(f"Loaded today's data: {df_today.shape}")

        # Feature engineering
        df_features = add_technical_indicators(df_today)

        # Load model and predict
        model = load_model()
        predictions = predict(model, df_features)

        # Generate bullish signals
        signals = generate_signals(predictions, df_features)

        # Save results
        today_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"outputs/reports/signals_{today_str}.csv"
        signals.to_csv(output_file, index=False)
        logger.info(f"Signals saved to {output_file}")

    except Exception as e:
        logger.exception("Error in daily screener")
