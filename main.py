# refactored_fyers_swing/main.py

import argparse
import pandas as pd
from config import HISTORICAL_DATA_FILE, LATEST_DATA_FILE
from data.fetch_token import launch_browser_login
from data.fetch_stock_list import load_symbols
from data.fetch_historical_data import fetch_and_store_all
from features.engineer_features import add_technical_indicators
from model.train_model import train_model
from strategy.screener import run_screener
from data.fetch_auth import generate_access_token
from utils.logger import get_logger

logger = get_logger("Main")

def main():
    parser = argparse.ArgumentParser(description="Swing Trading AI System")
    parser.add_argument("--auth", action="store_true", help="Run authentication flow")
    parser.add_argument("--fetch-history", action="store_true", help="Fetch historical OHLCV data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--screener", action="store_true", help="Run daily screener")
    parser.add_argument("--token", type=str, help="Generate access token using auth code")
    
    args = parser.parse_args()

    if args.auth:
        launch_browser_login()

    elif args.token:
        logger.info("Generating access token from auth code...")
        generate_access_token(args.token)

    elif args.fetch_history:
        symbols = load_symbols()
        fetch_and_store_all(symbols)

    elif args.train:
        df = pd.read_parquet(HISTORICAL_DATA_FILE)
        df = add_technical_indicators(df)
        
    

        if 'target_hit' not in df.columns:
            logger.error("'target_hit' column not found. Cannot train model.")
        else:
            train_model(df)

        df_latest = df.copy()
        latest_date = df_latest["date"].max()
        df_latest = df_latest[df_latest["date"] == latest_date].copy()
        print('latest'*30)
        print(df_latest)
        df_latest.to_parquet(LATEST_DATA_FILE, index=False)


    elif args.screener:
        run_screener()

    else:
        logger.info("No argument provided. Use --help for options.")

if __name__ == "__main__":
    main()
