# features/engineer_features.py

import numpy as np
import pandas as pd
from utils.helper import calculate_ema, calculate_rsi, calculate_atr
from features.swing_utils import (
    calculate_bb_position,
    add_candle_features,
    get_support_resistance,
    add_nearest_sr,
    generate_swing_labels
)
from config import EMA_PERIODS, RSI_PERIOD
from utils.logger import get_logger

logger = get_logger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA, RSI, ATR, candlestick, SR, and swing features.
    """
    all_dfs = []

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df.sort_values("date", inplace=True)

        # ➕ Technical Indicators
        for period in EMA_PERIODS:
            symbol_df[f"ema{period}"] = calculate_ema(symbol_df["close"], period)
        symbol_df["rsi"] = calculate_rsi(symbol_df["close"], RSI_PERIOD)
        symbol_df["atr"] = calculate_atr(symbol_df)

        # ➕ Candlestick Pattern Features
        symbol_df = add_candle_features(symbol_df)

        # ➕ Bollinger Band position
        symbol_df["bb_position"] = calculate_bb_position(symbol_df["close"], 20)

        # ➕ Support & Resistance
        support_levels, resistance_levels = get_support_resistance(symbol_df)
        symbol_df = add_nearest_sr(symbol_df, support_levels, resistance_levels)

        # ➕ Swing Labels (based on future price movement)
        symbol_df = generate_swing_labels(symbol_df)

        # ➕ Additional Engineered Features
        symbol_df = add_model_features(symbol_df)

        all_dfs.append(symbol_df)
        logger.info(f"✅ Features added for {symbol}")

    return pd.concat(all_dfs, ignore_index=True)


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived swing-trading features."""
    df = df.copy()
    df['daily_return'] = df['close'].pct_change()
    df['range_pct'] = (df['high'] - df['low']) / df['close']

    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['daily_return'].rolling(period).std()
        df[f'return_mean_{period}'] = df['daily_return'].rolling(period).mean()

    # EMA relationships
    df['price_above_ema20'] = (df['close'] > df['ema20']).astype(int)
    df['price_above_ema50'] = (df['close'] > df['ema50']).astype(int)
    df['ema20_above_ema50'] = (df['ema20'] > df['ema50']).astype(int)

    df['ema20_distance'] = (df['close'] - df['ema20']) / df['close']
    df['ema50_distance'] = (df['close'] - df['ema50']) / df['close']
    df['ema_spread'] = (df['ema20'] - df['ema50']) / df['close']

    # Normalize support/resistance distances
    df['norm_dist_to_support'] = df['dist_to_support'] / df['close']
    df['norm_dist_to_resistance'] = df['dist_to_resistance'] / df['close']

    # Volume-based features
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1)
        df['price_volume'] = df['close'] * df['volume']

    # Trend strength: slope of price over 10 days
    df['trend_strength'] = df['close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

    # Handle NaNs
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    return df
