# features/swing_utils.py

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(prices, period=14):
    """Calculate RSI with improved smoothing."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Use Wilder's smoothing method
    gain_smooth = gain.ewm(alpha=1/period, adjust=False).mean()
    loss_smooth = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain_smooth / loss_smooth
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bb_position(prices, period=20, std_dev=2):
    """Calculate Bollinger Band position with configurable parameters."""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # Calculate position within bands
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    
    # Add squeeze detection
    bb_width = (upper_band - lower_band) / sma
    bb_squeeze = bb_width < bb_width.rolling(20).quantile(0.1)
    
    return bb_position.clip(0, 1)


def add_candle_features(df):
    """Enhanced candlestick pattern recognition."""
    df = df.copy()
    
    # Vectorized basic candle components
    df['body'] = np.abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    
    # Vectorized max/min operations
    ohlc_max = np.maximum(df['open'], df['close'])
    ohlc_min = np.minimum(df['open'], df['close'])
    
    df['upper_shadow'] = df['high'] - ohlc_max
    df['lower_shadow'] = ohlc_min - df['low']
    
    # Candle direction
    df['is_bullish'] = (df['close'] > df['open']).astype(np.int8)
    df['is_bearish'] = (df['open'] > df['close']).astype(np.int8)
    
    # Ratios with epsilon for numerical stability
    range_safe = df['range'] + 1e-8
    df['body_to_range'] = df['body'] / range_safe
    df['upper_shadow_to_range'] = df['upper_shadow'] / range_safe
    df['lower_shadow_to_range'] = df['lower_shadow'] / range_safe
    
    # Vectorized basic patterns
    df['is_doji'] = (df['body_to_range'] < 0.1).astype(np.int8)
    df['is_hammer'] = ((df['body_to_range'] < 0.3) & 
                       (df['lower_shadow_to_range'] > 0.6) & 
                       (df['upper_shadow_to_range'] < 0.1)).astype(np.int8)
    df['is_shooting_star'] = ((df['body_to_range'] < 0.3) & 
                             (df['upper_shadow_to_range'] > 0.6) & 
                             (df['lower_shadow_to_range'] < 0.1)).astype(np.int8)
    
    # Advanced patterns
    df = add_advanced_candle_patterns(df)
    
    # Pattern strength
    df['pattern_strength'] = calculate_pattern_strength(df)
    
    return df


def add_advanced_candle_patterns(df):
    """Add advanced candlestick patterns."""
    df = df.copy()
    
    # Vectorized spinning tops
    df['spinning_top'] = ((df['body_to_range'] < 0.3) & 
                         (df['upper_shadow_to_range'] > 0.3) & 
                         (df['lower_shadow_to_range'] > 0.3)).astype(np.int8)
    
    # Vectorized Marubozu
    df['marubozu_bull'] = ((df['body_to_range'] > 0.95) & 
                          (df['close'] > df['open'])).astype(np.int8)
    df['marubozu_bear'] = ((df['body_to_range'] > 0.95) & 
                          (df['close'] < df['open'])).astype(np.int8)
    
    # Pre-compute shifts for multi-candle patterns
    close_s1 = df['close'].shift(1)
    close_s2 = df['close'].shift(2)
    is_bullish_s1 = df['is_bullish'].shift(1)
    is_bullish_s2 = df['is_bullish'].shift(2)
    is_bearish_s1 = df['is_bearish'].shift(1)
    is_bearish_s2 = df['is_bearish'].shift(2)
    body_s1 = df['body'].shift(1)
    body_s2 = df['body'].shift(2)
    open_s2 = df['open'].shift(2)
    
    # Three white soldiers / Three black crows
    df['three_white_soldiers'] = (
        (df['is_bullish'] == 1) & 
        (is_bullish_s1 == 1) & 
        (is_bullish_s2 == 1) & 
        (df['close'] > close_s1) & 
        (close_s1 > close_s2) & 
        (df['body'] > body_s1 * 0.8) & 
        (body_s1 > body_s2 * 0.8)
    ).astype(np.int8)
    
    df['three_black_crows'] = (
        (df['is_bearish'] == 1) & 
        (is_bearish_s1 == 1) & 
        (is_bearish_s2 == 1) & 
        (df['close'] < close_s1) & 
        (close_s1 < close_s2) & 
        (df['body'] > body_s1 * 0.8) & 
        (body_s1 > body_s2 * 0.8)
    ).astype(np.int8)
    
    # Morning star / Evening star
    midpoint_s2 = (open_s2 + close_s2) / 2
    df['morning_star'] = (
        (is_bearish_s2 == 1) & 
        (body_s1 < body_s2 * 0.3) & 
        (df['is_bullish'] == 1) & 
        (df['close'] > midpoint_s2)
    ).astype(np.int8)
    
    df['evening_star'] = (
        (is_bullish_s2 == 1) & 
        (body_s1 < body_s2 * 0.3) & 
        (df['is_bearish'] == 1) & 
        (df['close'] < midpoint_s2)
    ).astype(np.int8)
    
    # Harami patterns
    df['bullish_harami'] = (
        (is_bearish_s1 == 1) & 
        (df['is_bullish'] == 1) & 
        (df['high'] < close_s1) & 
        (df['low'] > open_s2)
    ).astype(np.int8)
    
    df['bearish_harami'] = (
        (is_bullish_s1 == 1) & 
        (df['is_bearish'] == 1) & 
        (df['high'] < open_s2) & 
        (df['low'] > close_s1)
    ).astype(np.int8)
    
    return df


def calculate_pattern_strength(df):
    """Calculate overall pattern strength score."""
    pattern_columns = [
        'is_doji', 'is_hammer', 'is_shooting_star', 'spinning_top',
        'marubozu_bull', 'marubozu_bear', 'three_white_soldiers',
        'three_black_crows', 'morning_star', 'evening_star',
        'bullish_harami', 'bearish_harami'
    ]
    
    # Weight patterns by reliability
    weights = np.array([
        0.5, 2.0, 2.0, 0.5, 1.5, 1.5, 3.0, 3.0, 2.5, 2.5, 1.5, 1.5
    ])
    
    # Vectorized calculation
    pattern_strength = np.zeros(len(df))
    for i, col in enumerate(pattern_columns):
        if col in df.columns:
            pattern_strength += df[col].values * weights[i]
    
    return pattern_strength


def get_support_resistance(df, window=10, min_strength=2):
    """Enhanced support and resistance detection with fractal analysis."""
    df = df.copy()
    
    # Use scipy to find local maxima and minima
    highs = df['high'].values
    lows = df['low'].values
    
    # Find local maxima (resistance)
    resistance_indices = argrelextrema(highs, np.greater, order=window)[0]
    
    # Find local minima (support)  
    support_indices = argrelextrema(lows, np.less, order=window)[0]
    
    # Vectorized distance calculations for strength
    tolerance = 0.02
    
    # Calculate support levels with strength
    support_levels = []
    for idx in support_indices:
        if idx < len(df):
            level_price = df.iloc[idx]['low']
            
            # Vectorized touch calculation
            low_touches = np.sum(np.abs(lows - level_price) / level_price < tolerance)
            high_touches = np.sum(np.abs(highs - level_price) / level_price < tolerance)
            touches = low_touches + high_touches
            
            if touches >= min_strength:
                support_levels.append((idx, level_price, touches))
    
    # Calculate resistance levels with strength
    resistance_levels = []
    for idx in resistance_indices:
        if idx < len(df):
            level_price = df.iloc[idx]['high']
            
            # Vectorized touch calculation
            high_touches = np.sum(np.abs(highs - level_price) / level_price < tolerance)
            low_touches = np.sum(np.abs(lows - level_price) / level_price < tolerance)
            touches = high_touches + low_touches
            
            if touches >= min_strength:
                resistance_levels.append((idx, level_price, touches))
    
    # Add dynamic support/resistance based on moving averages
    support_levels.extend(get_dynamic_support_resistance(df, 'support'))
    resistance_levels.extend(get_dynamic_support_resistance(df, 'resistance'))
    
    return support_levels, resistance_levels


def get_dynamic_support_resistance(df, level_type='support'):
    """Get dynamic support/resistance from moving averages and trend lines."""
    dynamic_levels = []
    
    # EMAs as dynamic support/resistance
    ema_periods = [20, 50, 100, 200]
    current_close = df['close'].iloc[-1]
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        if ema_col in df.columns:
            ema_values = df[ema_col].dropna()
            if len(ema_values) > 0:
                recent_ema = ema_values.iloc[-1]
                strength = period // 10
                
                if level_type == 'support' and recent_ema < current_close:
                    dynamic_levels.append((len(df)-1, recent_ema, strength))
                elif level_type == 'resistance' and recent_ema > current_close:
                    dynamic_levels.append((len(df)-1, recent_ema, strength))
    
    # Trend line support/resistance
    trend_levels = calculate_trend_lines(df, level_type)
    dynamic_levels.extend(trend_levels)
    
    return dynamic_levels


def calculate_trend_lines(df, level_type='support', lookback=50):
    """Calculate trend lines as dynamic support/resistance."""
    if len(df) < lookback:
        return []
    
    trend_levels = []
    recent_df = df.tail(lookback)
    
    if level_type == 'support':
        # Find upward trend line through recent lows
        lows = recent_df['low'].values
        
        # Get bottom 20% of lows for trend line
        n_lows = max(2, len(lows) // 5)
        low_indices = np.argpartition(lows, n_lows)[:n_lows]
        
        if len(low_indices) >= 2:
            slope, intercept, r_value, _, _ = linregress(low_indices, lows[low_indices])
            
            # Project trend line to current position
            current_trend_level = slope * (len(lows) - 1) + intercept
            
            # Only consider as support if trend is upward and correlation is good
            if slope > 0 and r_value > 0.5:
                strength = int(abs(r_value) * 5)
                trend_levels.append((len(df)-1, current_trend_level, strength))
    
    else:  # resistance
        # Find downward trend line through recent highs
        highs = recent_df['high'].values
        
        # Get top 20% of highs for trend line
        n_highs = max(2, len(highs) // 5)
        high_indices = np.argpartition(highs, -n_highs)[-n_highs:]
        
        if len(high_indices) >= 2:
            slope, intercept, r_value, _, _ = linregress(high_indices, highs[high_indices])
            
            # Project trend line to current position
            current_trend_level = slope * (len(highs) - 1) + intercept
            
            # Only consider as resistance if trend is downward and correlation is good
            if slope < 0 and r_value > 0.5:
                strength = int(abs(r_value) * 5)
                trend_levels.append((len(df)-1, current_trend_level, strength))
    
    return trend_levels


def add_nearest_sr(df, support_levels, resistance_levels):
    """Enhanced support/resistance distance calculation."""
    df = df.copy()
    
    # Convert to arrays for vectorized operations
    support_data = np.array([(level[1], level[2]) for level in support_levels]) if support_levels else np.array([]).reshape(0, 2)
    resistance_data = np.array([(level[1], level[2]) for level in resistance_levels]) if resistance_levels else np.array([]).reshape(0, 2)
    
    close_prices = df['close'].values
    n_rows = len(df)
    
    # Pre-allocate arrays
    nearest_support = np.zeros(n_rows)
    nearest_resistance = np.zeros(n_rows)
    support_strength = np.ones(n_rows)
    resistance_strength = np.ones(n_rows)
    multiple_support_levels = np.zeros(n_rows, dtype=int)
    multiple_resistance_levels = np.zeros(n_rows, dtype=int)
    
    # Vectorized processing
    for i in range(n_rows):
        close = close_prices[i]
        
        # Process support levels
        if len(support_data) > 0:
            # Find supports within 10% of current price
            distances = np.abs(support_data[:, 0] - close) / close
            close_mask = distances < 0.1
            
            if np.any(close_mask):
                close_supports = support_data[close_mask]
                weights = close_supports[:, 1] / (distances[close_mask] + 0.01)
                best_idx = np.argmax(weights)
                
                nearest_support[i] = close_supports[best_idx, 0]
                support_strength[i] = close_supports[best_idx, 1]
                multiple_support_levels[i] = np.sum(close_supports[:, 0] <= close)
            else:
                nearest_support[i] = close * 0.95
        else:
            nearest_support[i] = close * 0.95
        
        # Process resistance levels
        if len(resistance_data) > 0:
            # Find resistances within 10% of current price
            distances = np.abs(resistance_data[:, 0] - close) / close
            close_mask = distances < 0.1
            
            if np.any(close_mask):
                close_resistances = resistance_data[close_mask]
                weights = close_resistances[:, 1] / (distances[close_mask] + 0.01)
                best_idx = np.argmax(weights)
                
                nearest_resistance[i] = close_resistances[best_idx, 0]
                resistance_strength[i] = close_resistances[best_idx, 1]
                multiple_resistance_levels[i] = np.sum(close_resistances[:, 0] >= close)
            else:
                nearest_resistance[i] = close * 1.05
        else:
            nearest_resistance[i] = close * 1.05
    
    # Add to dataframe
    df['nearest_support'] = nearest_support
    df['nearest_resistance'] = nearest_resistance
    df['support_strength'] = support_strength
    df['resistance_strength'] = resistance_strength
    df['multiple_support_levels'] = multiple_support_levels
    df['multiple_resistance_levels'] = multiple_resistance_levels
    
    # Vectorized calculations
    df['dist_to_support'] = df['close'] - df['nearest_support']
    df['dist_to_resistance'] = df['nearest_resistance'] - df['close']
    df['pct_to_support'] = (df['close'] - df['nearest_support']) / df['close'] * 100
    df['pct_to_resistance'] = (df['nearest_resistance'] - df['close']) / df['close'] * 100
    
    # Support/Resistance zones
    df['in_support_zone'] = (df['pct_to_support'] < 2).astype(np.int8)
    df['in_resistance_zone'] = (df['pct_to_resistance'] < 2).astype(np.int8)
    
    # Confluence zones
    df['support_confluence'] = (df['multiple_support_levels'] >= 2).astype(np.int8)
    df['resistance_confluence'] = (df['multiple_resistance_levels'] >= 2).astype(np.int8)
    
    return df


def generate_swing_labels(df, target_pct=0.05, window=10, stop_loss_pct=0.03):
    """Enhanced swing label generation with multiple target levels."""
    df = df.copy()
    
    # Multiple target levels
    targets = [0.03, 0.05, 0.08, 0.10]
    
    n_rows = len(df)
    
    # Pre-allocate arrays
    target_hit = np.full(n_rows, np.nan)
    max_return = np.full(n_rows, np.nan)
    min_return = np.full(n_rows, np.nan)
    days_to_target = np.full(n_rows, np.nan)
    risk_reward_ratio = np.full(n_rows, np.nan)
    
    # Multi-target arrays
    target_arrays = {}
    days_arrays = {}
    for target in targets:
        target_key = f'target_hit_{int(target*100)}'
        days_key = f'days_to_target_{int(target*100)}'
        target_arrays[target_key] = np.full(n_rows, np.nan)
        days_arrays[days_key] = np.full(n_rows, np.nan)
    
    # Vectorized processing
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    valid_indices = np.arange(n_rows - window)
    
    for i in valid_indices:
        entry_price = close_prices[i]
        future_highs = high_prices[i+1:i+1+window]
        future_lows = low_prices[i+1:i+1+window]
        
        if len(future_highs) == 0:
            continue
            
        max_high = np.max(future_highs)
        min_low = np.min(future_lows)
        
        max_ret = (max_high - entry_price) / entry_price
        min_ret = (min_low - entry_price) / entry_price
        
        max_return[i] = max_ret
        min_return[i] = min_ret
        
        # Check primary target
        if max_ret >= target_pct and min_ret > -stop_loss_pct:
            target_hit[i] = 1
            
            # Find days to target using vectorized operation
            target_reached = (future_highs - entry_price) / entry_price >= target_pct
            if np.any(target_reached):
                days_to_target[i] = np.argmax(target_reached) + 1
            else:
                days_to_target[i] = window
        else:
            target_hit[i] = 0
        
        # Risk-reward ratio
        if min_ret < 0:
            risk_reward_ratio[i] = max_ret / abs(min_ret)
        else:
            risk_reward_ratio[i] = max_ret / 0.01
        
        # Check multiple targets
        for target in targets:
            target_key = f'target_hit_{int(target*100)}'
            days_key = f'days_to_target_{int(target*100)}'
            
            if max_ret >= target and min_ret > -stop_loss_pct:
                target_arrays[target_key][i] = 1
                
                # Find days to this target
                target_reached = (future_highs - entry_price) / entry_price >= target
                if np.any(target_reached):
                    days_arrays[days_key][i] = np.argmax(target_reached) + 1
            else:
                target_arrays[target_key][i] = 0
    
    # Add to dataframe
    df['target_hit'] = target_hit
    df['max_return'] = max_return
    df['min_return'] = min_return
    df['days_to_target'] = days_to_target
    df['risk_reward_ratio'] = risk_reward_ratio
    
    # Add multi-target columns
    for target in targets:
        target_key = f'target_hit_{int(target*100)}'
        days_key = f'days_to_target_{int(target*100)}'
        df[target_key] = target_arrays[target_key]
        df[days_key] = days_arrays[days_key]
    
    # Advanced labeling features
    df = add_advanced_swing_labels(df, window)
    
    return df


def add_advanced_swing_labels(df, window=10):
    """Add advanced swing trading labels and features."""
    df = df.copy()
    
    # Trend following vs mean reversion classification
    n_rows = len(df)
    trend_trade = np.full(n_rows, np.nan)
    mean_reversion_trade = np.full(n_rows, np.nan)
    breakout_trade = np.full(n_rows, np.nan)
    
    # Vectorized processing where possible
    target_hit_mask = ~pd.isna(df['target_hit'])
    valid_indices = np.where(target_hit_mask)[0]
    valid_indices = valid_indices[valid_indices < n_rows - window]
    
    for i in valid_indices:
        # Get current market state
        current_trend = df['trend_direction'].iloc[i] if 'trend_direction' in df.columns else 0
        near_support = df['in_support_zone'].iloc[i] if 'in_support_zone' in df.columns else 0
        near_resistance = df['in_resistance_zone'].iloc[i] if 'in_resistance_zone' in df.columns else 0
        
        # Classify trade type
        if current_trend == 1 and near_support:
            trend_trade[i] = 1
            mean_reversion_trade[i] = 0
            breakout_trade[i] = 0
        elif current_trend == 0 and (near_support or near_resistance):
            mean_reversion_trade[i] = 1
            trend_trade[i] = 0
            breakout_trade[i] = 0
        else:
            breakout_trade[i] = 1
            trend_trade[i] = 0
            mean_reversion_trade[i] = 0
    
    df['trend_trade'] = trend_trade
    df['mean_reversion_trade'] = mean_reversion_trade
    df['breakout_trade'] = breakout_trade
    
    # Success rate by trade type
    df['trade_type_success'] = np.nan
    
    # Calculate win rate for different market conditions
    df = add_conditional_success_rates(df)
    
    return df


def add_conditional_success_rates(df):
    """Add success rates based on market conditions."""
    df = df.copy()
    
    # Define market conditions
    conditions = {
        'high_volume': 'volume_ratio > 1.5' if 'volume_ratio' in df.columns else None,
        'low_volatility': 'volatility_5 < volatility_20' if 'volatility_5' in df.columns else None,
        'bullish_regime': 'bull_regime == 1' if 'bull_regime' in df.columns else None,
        'bearish_regime': 'bear_regime == 1' if 'bear_regime' in df.columns else None,
        'high_rsi': 'rsi > 70' if 'rsi' in df.columns else None,
        'low_rsi': 'rsi < 30' if 'rsi' in df.columns else None,
    }
    
    # Calculate success rates for each condition
    for condition_name, condition_query in conditions.items():
        if condition_query is None:
            df[f'{condition_name}_success_rate'] = 0.5
            continue
            
        try:
            condition_mask = df.query(condition_query).index
            if len(condition_mask) > 0:
                success_rate = df.loc[condition_mask, 'target_hit'].mean()
                if pd.isna(success_rate):
                    success_rate = 0.5
            else:
                success_rate = 0.5
            df[f'{condition_name}_success_rate'] = success_rate
        except:
            df[f'{condition_name}_success_rate'] = 0.5
    
    return df


def calculate_swing_quality_score(df):
    """Calculate a comprehensive swing quality score."""
    df = df.copy()
    
    # Initialize score
    swing_score = np.zeros(len(df))
    
    # Technical alignment (30% weight)
    if 'bullish_confluence' in df.columns:
        max_confluence = df['bullish_confluence'].max()
        if max_confluence > 0:
            swing_score += (df['bullish_confluence'] / max_confluence) * 0.3
    
    # Support/Resistance context (25% weight)
    if 'in_support_zone' in df.columns:
        swing_score += df['in_support_zone'].values * 0.25
    
    # Pattern strength (20% weight)
    if 'pattern_strength' in df.columns:
        max_pattern = df['pattern_strength'].max()
        if max_pattern > 0:
            swing_score += (df['pattern_strength'] / max_pattern) * 0.2
    
    # Volume confirmation (15% weight)
    if 'volume_ratio' in df.columns:
        volume_score = np.clip(df['volume_ratio'].values - 1, 0, 1)
        swing_score += volume_score * 0.15
    
    # Risk-reward potential (10% weight)
    if 'risk_reward_ratio' in df.columns:
        rr_score = np.clip(df['risk_reward_ratio'].values / 3, 0, 1)
        swing_score += rr_score * 0.1
    
    df['swing_quality_score'] = swing_score
    
    return df