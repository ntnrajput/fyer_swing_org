# refactored_fyers_swing/config.py

import os
from pathlib import Path

# === FYERS API ===
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "VE3CCLJZWA-100" )
FYERS_SECRET_ID = os.getenv("FYERS_SECRET_ID", "QEGA69PVUL")
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "https://www.google.com" )
FYERS_APP_ID_HASH = os.getenv("FYERS_APP_ID_HASH", "b209632623b60de416ea3bcbd2b780ef11ebdbb652b3f06f63ffdd34366faa18")

TOKEN_PATH = Path("outputs/fyers_access_token.txt")


#=====Good Stock Filter====
AVG_VOL = 1.0
AVG_PRICE = 50

# === DATA SETTINGS ===
SYMBOLS_FILE = Path("symbols.csv")
HISTORICAL_DATA_FILE = Path("outputs/all_symbols_history.parquet")
LATEST_DATA_FILE = Path("outputs/latest_full_data.parquet")
DAILY_DATA_FILE = Path("outputs/today_data.csv")


# === INDICATOR SETTINGS ===
EMA_PERIODS = [20, 50, 200]
RSI_PERIOD = 14
VOLUME_LOOKBACK = 20

# === MODEL ===
MODEL_FILE = Path("model/model.pkl")

# === LOGGING ===
LOG_FILE = Path("outputs/logs/system.log")

#====Feature Columns====
FEATURE_COLUMNS = [
    'daily_return', 'range_pct', 'volatility_5', 'volatility_10', 'volatility_20',
    'return_mean_5', 'return_mean_10', 'return_mean_20',
    'price_above_ema20', 'price_above_ema50', 'ema20_above_ema50',
    'ema20_distance', 'ema50_distance', 'ema_spread',
    'norm_dist_to_support', 'norm_dist_to_resistance',
    'volume_ma', 'volume_ratio', 'price_volume',
    'rsi', 'bb_position', 'trend_strength',
    'is_doji', 'is_hammer'
]
