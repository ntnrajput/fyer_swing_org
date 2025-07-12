# features/swing_utils.py

import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bb_position(prices, period=20):
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return bb_position.clip(0, 1)

def add_candle_features(df):
    df = df.copy()
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_bearish'] = (df['open'] > df['close']).astype(int)
    df['body_to_range'] = df['body'] / (df['range'] + 1e-8)
    df['upper_shadow_to_range'] = df['upper_shadow'] / (df['range'] + 1e-8)
    df['lower_shadow_to_range'] = df['lower_shadow'] / (df['range'] + 1e-8)

    df['is_doji'] = (df['body_to_range'] < 0.1).astype(int)
    df['is_hammer'] = ((df['body_to_range'] < 0.3) & (df['lower_shadow_to_range'] > 0.6)).astype(int)

    return df

def get_support_resistance(df, window=5):
    df = df.copy()
    rolling_max = df['high'].rolling(window=2 * window + 1, center=True).max()
    rolling_min = df['low'].rolling(window=2 * window + 1, center=True).min()

    df['Swing_High'] = (df['high'] == rolling_max)
    df['Swing_Low'] = (df['low'] == rolling_min)

    support_levels = []
    resistance_levels = []

    for idx, row in df[df['Swing_Low']].iterrows():
        level_price = row['low']
        strength = sum(abs(df['low'] - level_price) / level_price < 0.01)
        support_levels.append((idx, level_price, strength))

    for idx, row in df[df['Swing_High']].iterrows():
        level_price = row['high']
        strength = sum(abs(df['high'] - level_price) / level_price < 0.01)
        resistance_levels.append((idx, level_price, strength))

    return support_levels, resistance_levels

def add_nearest_sr(df, support_levels, resistance_levels):
    support_data = [(level[1], level[2]) for level in support_levels]
    resistance_data = [(level[1], level[2]) for level in resistance_levels]

    nearest_support = []
    nearest_resistance = []
    support_strength = []
    resistance_strength = []

    for close in df['close']:
        if support_data:
            nearest_sup = min(support_data, key=lambda x: abs(x[0] - close))
            nearest_support.append(nearest_sup[0])
            support_strength.append(nearest_sup[1])
        else:
            nearest_support.append(close * 0.95)
            support_strength.append(1)

        if resistance_data:
            nearest_res = min(resistance_data, key=lambda x: abs(x[0] - close))
            nearest_resistance.append(nearest_res[0])
            resistance_strength.append(nearest_res[1])
        else:
            nearest_resistance.append(close * 1.05)
            resistance_strength.append(1)

    df['nearest_support'] = nearest_support
    df['nearest_resistance'] = nearest_resistance
    df['support_strength'] = support_strength
    df['resistance_strength'] = resistance_strength
    df['dist_to_support'] = df['close'] - df['nearest_support']
    df['dist_to_resistance'] = df['nearest_resistance'] - df['close']

    return df

def generate_swing_labels(df, target_pct=0.05, window=10, stop_loss_pct=0.03):
    df = df.copy()
    target_hit = []
    max_return = []
    min_return = []

    for i in range(len(df)):
        if i + window >= len(df):
            target_hit.append(None)
            max_return.append(None)
            min_return.append(None)
        else:
            entry_price = df['close'].iloc[i]
            future_high = df['high'].iloc[i+1:i+1+window].max()
            future_low = df['low'].iloc[i+1:i+1+window].min()

            max_ret = (future_high - entry_price) / entry_price
            min_ret = (future_low - entry_price) / entry_price

            max_return.append(max_ret)
            min_return.append(min_ret)

            if max_ret >= target_pct and min_ret > -stop_loss_pct:
                target_hit.append(1)
            else:
                target_hit.append(0)

    df['target_hit'] = target_hit
    df['max_return'] = max_return
    df['min_return'] = min_return

    return df
