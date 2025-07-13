import os
import time
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from data.fetch_auth import get_saved_access_token
from config import HISTORICAL_DATA_FILE
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize Fyers API client
def get_fyers_client():
    token = get_saved_access_token()
    if not token:
        raise ValueError("Access token is missing. Please authenticate.")
    return fyersModel.FyersModel(token=token, is_async=False, client_id=None)

def fetch_ohlcv_data_range(fyers, symbol: str, from_date: datetime, to_date: datetime, resolution: str = "1D"):
    """Fetch OHLCV data for a given date range"""
    try:
        start_str = from_date.strftime("%Y-%m-%d")
        end_str = to_date.strftime("%Y-%m-%d")
        print(f"Fetching {symbol} from {start_str} to {end_str}")

        data = fyers.history({
            "symbol": symbol.strip(),
            "resolution": resolution,
            "date_format": "1",
            "range_from": start_str,
            "range_to": end_str,
            "cont_flag": "1"
        })

        if data["s"] == "ok":
            df = pd.DataFrame(data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["volume"] = df["volume"] / 100000 
            df["symbol"] = symbol
            df["date"] = pd.to_datetime(df["timestamp"], unit="s")
            return df.drop(columns=["timestamp"])
        else:
            logger.warning(f" Failed for {symbol} ({start_str} to {end_str}): {data.get('message')}")
    except Exception as e:
        logger.exception(f" Error fetching {symbol} from {start_str} to {end_str}: {e}")
    return pd.DataFrame()

def fetch_and_store_all(symbols: list, years: int = 3):
    """Fetch and save historical data for all symbols, incrementally"""
    fyers = get_fyers_client()
    today = datetime.now().date()
    start_date_default = today - timedelta(days=365 * years)

    # Load existing data if available
    if os.path.exists(HISTORICAL_DATA_FILE):
        existing_data = pd.read_parquet(HISTORICAL_DATA_FILE)
        existing_data['date'] = pd.to_datetime(existing_data['date']).dt.date
        logger.info(f" Loaded existing historical data from {HISTORICAL_DATA_FILE}")
    else:
        existing_data = pd.DataFrame()

    updated_data = []

    for symbol in symbols:
        logger.info(f"Checking updates for {symbol}")
        
        symbol_data = existing_data[existing_data["symbol"] == symbol] if not existing_data.empty else pd.DataFrame()

        if not symbol_data.empty:
            latest_date = symbol_data["date"].max()
            fetch_from = latest_date + timedelta(days=1)
        else:
            fetch_from = start_date_default

        if fetch_from >= today:
            logger.info(f" Up-to-date: {symbol}")
            updated_data.append(symbol_data)
            continue

        # Fetch only missing range
        new_data = fetch_ohlcv_data_range(fyers, symbol, fetch_from, today)
        time.sleep(0.5)

        if not new_data.empty:
            combined = pd.concat([symbol_data, new_data], ignore_index=True)
        else:
            combined = symbol_data

        updated_data.append(combined)

    # Merge all updated data and save
    if updated_data:
        full_data = pd.concat(updated_data, ignore_index=True)
        full_data = full_data.drop_duplicates(subset=["symbol", "date"]).sort_values(by=["symbol", "date"])
        full_data.to_parquet(HISTORICAL_DATA_FILE, index=False)
        logger.info(f"Updated historical data saved to {HISTORICAL_DATA_FILE}")
    else:
        logger.warning(" No new data to update.")

