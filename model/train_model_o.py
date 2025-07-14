# model/train_model.py

import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from config import FEATURE_COLUMNS, MODEL_FILE

logger = get_logger(__name__)

def validate_features(df):
    """Validate and report on available features"""
    logger.info(" Validating available features...")
    
    # Categorize features
    feature_categories = {
        'Technical Indicators': ['rsi', 'atr', 'bb_position'],
        'EMA Features': [col for col in df.columns if col.startswith('ema')],
        'Candlestick Patterns': ['is_doji', 'is_hammer', 'is_bullish', 'is_bearish', 
                                'body_to_range', 'upper_shadow_to_range', 'lower_shadow_to_range'],
        'Support/Resistance': ['norm_dist_to_support', 'norm_dist_to_resistance', 
                              'support_strength', 'resistance_strength', 'nearest_support', 'nearest_resistance'],
        'Volatility & Returns': ['daily_return', 'range_pct', 'volatility_5', 'volatility_10', 
                                'volatility_20', 'return_mean_5', 'return_mean_10', 'return_mean_20'],
        'Volume Features': ['volume_ratio', 'price_volume', 'volume_ma', 'price_volume_correlation'],
        'Trend Features': ['trend_strength', 'ema20_distance', 'ema50_distance', 'ema_spread'],
        'Advanced Features': ['rsi_oversold', 'rsi_overbought', 'gap_up', 'gap_down', 
                             'max_drawdown', 'sharpe_proxy', 'trend_consistency']
    }
    
    available_features = {}
    missing_features = {}
    
    for category, features in feature_categories.items():
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        
        if available:
            available_features[category] = available
        if missing:
            missing_features[category] = missing
    
    # Report available features
    logger.info("Available Feature Categories:")
    for category, features in available_features.items():
        logger.info(f"    {category}: {len(features)} features")
        for feature in features[:5]:  # Show first 5 features
            logger.info(f"      - {feature}")
        if len(features) > 5:
            logger.info(f"      ... and {len(features) - 5} more")
    
    # Report missing features (if any)
    if missing_features:
        logger.info(" Missing Feature Categories:")
        for category, features in missing_features.items():
            logger.info(f"    {category}: {len(features)} missing features")
    
    total_available = sum(len(features) for features in available_features.values())
    logger.info(f" Total Features Available: {total_available}")
    
    return available_features, missing_features

def create_advanced_features(df):
    """Create advanced technical indicators and features for swing trading"""
    logger.info(" Creating advanced features...")
    
    # Check if existing features from engineer_features.py are already present
    existing_features = set(df.columns)
    
    # Price momentum features (avoid duplicates with daily_return)
    if 'daily_return' not in existing_features:
        df['daily_return'] = df['close'].pct_change()
    
    for window in [5, 10, 20]:
        if f'price_momentum_{window}' not in existing_features:
            df[f'price_momentum_{window}'] = df['close'].pct_change(window)
        if 'volume' in df.columns and f'volume_momentum_{window}' not in existing_features:
            df[f'volume_momentum_{window}'] = df['volume'].pct_change(window)
    
    # Volatility features (complement existing volatility_5, volatility_10, volatility_20)
    if 'price_volatility_10' not in existing_features:
        df['price_volatility_10'] = df['close'].rolling(10).std()
    if 'price_volatility_20' not in existing_features:
        df['price_volatility_20'] = df['close'].rolling(20).std()
    if 'volatility_ratio' not in existing_features:
        df['volatility_ratio'] = df['price_volatility_10'] / (df['price_volatility_20'] + 1e-8)
    
    # Support/Resistance levels (complement existing nearest_support/resistance)
    if 'support_level' not in existing_features:
        df['support_level'] = df['low'].rolling(20).min()
    if 'resistance_level' not in existing_features:
        df['resistance_level'] = df['high'].rolling(20).max()
    
    # Use existing support/resistance distances if available, otherwise create new ones
    if 'norm_dist_to_support' in existing_features:
        df['support_distance'] = df['norm_dist_to_support']
    elif 'support_distance' not in existing_features:
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
    
    if 'norm_dist_to_resistance' in existing_features:
        df['resistance_distance'] = df['norm_dist_to_resistance']
    elif 'resistance_distance' not in existing_features:
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
    
    # Market strength indicators
    if 'higher_high' not in existing_features:
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    if 'higher_low' not in existing_features:
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
    if 'bullish_pattern' not in existing_features:
        df['bullish_pattern'] = df['higher_high'] & df['higher_low']
    
    # Volume analysis (complement existing volume features)
    if 'volume' in df.columns:
        if 'volume_ma' not in existing_features:
            df['volume_ma'] = df['volume'].rolling(20).mean()
        if 'volume_ratio' not in existing_features:
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        if 'price_volume_correlation' not in existing_features:
            df['price_volume_correlation'] = df['close'].rolling(20).corr(df['volume'])
    
    # Gap analysis
    if 'gap_up' not in existing_features:
        df['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1) > 0.02).astype(int)
    if 'gap_down' not in existing_features:
        df['gap_down'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1) < -0.02).astype(int)
    
    # Trend strength (complement existing trend_strength if different calculation)
    if 'trend_strength' not in existing_features:
        df['trend_strength_ma'] = np.where(df['close'] > df['close'].rolling(20).mean(), 1, 
                                          np.where(df['close'] < df['close'].rolling(20).mean(), -1, 0))
    
    # Risk metrics
    if 'max_drawdown' not in existing_features:
        df['max_drawdown'] = (df['close'] / df['close'].rolling(20).max() - 1) * 100
    if 'sharpe_proxy' not in existing_features:
        df['sharpe_proxy'] = df['close'].pct_change().rolling(20).mean() / (df['close'].pct_change().rolling(20).std() + 1e-8)
    
    # Additional advanced features that complement existing ones
    if 'rsi' in existing_features:
        # RSI-based features
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_momentum'] = df['rsi'].diff()
    
    if 'atr' in existing_features:
        # ATR-based features
        df['atr_ratio'] = df['atr'] / df['close']
        df['price_atr_position'] = (df['close'] - df['close'].shift(1)) / df['atr']
    
    # Bollinger Band enhancements (if bb_position exists)
    if 'bb_position' in existing_features:
        df['bb_squeeze'] = (df['bb_position'] > 0.8).astype(int) | (df['bb_position'] < 0.2).astype(int)
        df['bb_expansion'] = (df['bb_position'].rolling(5).std() > 0.1).astype(int)
    
    # Candlestick pattern enhancements (if candle features exist)
    if 'is_doji' in existing_features and 'is_hammer' in existing_features:
        df['reversal_pattern'] = df['is_doji'] | df['is_hammer']
        df['consecutive_bullish'] = df['is_bullish'].rolling(3).sum()
        df['consecutive_bearish'] = df['is_bearish'].rolling(3).sum()
    
    # EMA-based momentum (if EMAs exist)
    ema_cols = [col for col in existing_features if col.startswith('ema')]
    if ema_cols:
        # EMA convergence/divergence
        if 'ema20' in existing_features and 'ema50' in existing_features:
            df['ema_convergence'] = abs(df['ema20'] - df['ema50']) / df['close']
            df['ema_momentum'] = (df['ema20'] - df['ema20'].shift(1)) / df['ema20']
    
    # Multi-timeframe strength
    df['short_term_strength'] = (
        df['close'].rolling(5).mean() / df['close'].rolling(20).mean() - 1
    ) * 100
    
    df['medium_term_strength'] = (
        df['close'].rolling(10).mean() / df['close'].rolling(50).mean() - 1
    ) * 100 if len(df) > 50 else 0
    
    # Price pattern recognition
    df['higher_highs_count'] = df['higher_high'].rolling(10).sum()
    df['higher_lows_count'] = df['higher_low'].rolling(10).sum()
    df['trend_consistency'] = (df['higher_highs_count'] + df['higher_lows_count']) / 20
    
    # Relative strength vs market (if multiple symbols)
    if len(df['symbol'].unique()) > 1:
        market_return = df.groupby('date')['close'].pct_change().groupby(df['date']).mean()
        df['relative_strength'] = df['daily_return'] - df['date'].map(market_return)

    
    return df

def optimize_hyperparameters(X_train, y_train):
    """Optimize hyperparameters using TimeSeriesSplit"""
    logger.info(" Optimizing hyperparameters...")
    
    # Define parameter grids

    #  rf_params = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [5, 10, 15, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['sqrt', 'log2', None]
    # }

    rf_params = {
        'n_estimators': [100],
        'max_depth': [10,None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1],
        'max_features': ['sqrt']
    }
    

    # gb_params = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'subsample': [0.8, 0.9, 1.0]
    # }


    gb_params = {
        'n_estimators': [100],
        'max_depth': [3] ,
        'learning_rate': [0.1],
        'subsample': [0.9]
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Random Forest optimization
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    # Gradient Boosting optimization
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    rf_grid.fit(X_train, y_train)
    gb_grid.fit(X_train, y_train)
    
    logger.info(f" Best RF AUC: {rf_grid.best_score_:.3f}")
    logger.info(f" Best GB AUC: {gb_grid.best_score_:.3f}")
    
    return rf_grid.best_estimator_, gb_grid.best_estimator_

def create_ensemble_model(rf_model, gb_model, X_train, y_train):
    """Create ensemble model with multiple algorithms"""
    logger.info(" Creating ensemble model...")
    
    # Logistic Regression for ensemble
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble

def evaluate_model_performance(model, X_test, y_test, X_train, y_train):
    """Comprehensive model evaluation for trading"""
    logger.info(" Evaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification metrics
    report = classification_report(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Trading-specific metrics
    true_positives = conf_matrix[1, 1]
    false_positives = conf_matrix[0, 1]
    false_negatives = conf_matrix[1, 0]
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Win rate and profit estimation
    win_rate = precision * 100
    
    logger.info(f"\nTRADING PERFORMANCE METRICS:")
    logger.info(f"    Win Rate: {win_rate:.1f}%")
    logger.info(f"    AUC Score: {auc_score:.3f}")
    logger.info(f"    Precision: {precision:.3f}")
    logger.info(f"    Recall: {recall:.3f}")
    logger.info(f"\n Classification Report:\n{report}")
    
    # Feature importance for top model
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            logger.info(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    return {
        'auc_score': auc_score,
        'win_rate': win_rate,
        'precision': precision,
        'recall': recall,
        'classification_report': report
    }

def train_model(df):
    """Advanced model training for swing trading predictions"""
    try:
        logger.info(" Loading data for advanced model training...")
        df = df.copy()

       
        
        # Validate existing features before adding new ones
        available_features, missing_features = validate_features(df)

        # Create advanced features (complementing existing ones)
        df = create_advanced_features(df)

    
        
        logger.info(" Filtering and preparing training data...")
        
        # Check if target_hit column exists
        if 'target_hit' not in df.columns:
            logger.error("target_hit column not found in dataframe")
            raise ValueError("target_hit column not found in dataframe")
        
        # Check data before filtering
        logger.info(f" Initial data shape: {df.shape}")
        logger.info(f" Target column info: {df['target_hit'].value_counts()}")
        
        # Filter out rows with NaN target values
        df_before_filter = df.copy()
        df = df.dropna(subset=["target_hit"])
        
        logger.info(f" After filtering NaN targets: {df.shape}")
        
        # Ensure target is binary
        df["target_hit"] = df["target_hit"].astype(int)
        
        # Clean target variable - ensure it's 0 or 1
        logger.info(" Cleaning target variable...")
        unique_targets = df["target_hit"].unique()
        logger.info(f" Unique target values: {unique_targets}")
        
        # Filter to only keep 0 and 1 values
        df = df[df["target_hit"].isin([0, 1])]
        logger.info(f" After filtering to binary targets: {df.shape}")

        cols_to_remove = [
            'days_to_target', 
            'days_to_target_3', 
            'days_to_target_7', 
            'days_to_target_10', 
            'days_to_target_12', 
            'trade_type_success',
            'price_atr_position',
            'price_volume_correlation',
            'volume_momentum_10',
            'volume_momentum_5',
            'volume_momentum_5',

        ]

        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])

        feature_cols = [col for col in df.columns if col not in ['target_hit', 'symbol', 'date']]
        df = df.dropna(subset=feature_cols)

        logger.info(f" After removing NaN values from feature columns: {df.shape}")
        
        # Check if we have any data left
        if len(df) == 0:
            logger.error("No data remaining after filtering. Check your data preprocessing.")
            raise ValueError("No data remaining after filtering")
        
        # Enhanced feature selection - ensure we use all available features
        exclude_cols = ['target_hit', 'date', 'symbol', 'max_return', 'min_return', 
                       'Swing_High', 'Swing_Low']  # Exclude target and intermediate columns
        all_features = [col for col in df.columns if col not in exclude_cols]
        available_features = [col for col in all_features if col in df.columns]
        
        # Prioritize existing engineered features and add new advanced features
        existing_engineered_features = [
            # Technical indicators
            'rsi', 'atr', 'bb_position',
            # EMA features
            'ema20_distance', 'ema50_distance', 'ema_spread',
            'price_above_ema20', 'price_above_ema50', 'ema20_above_ema50',
            # Candlestick features
            'is_doji', 'is_hammer', 'is_bullish', 'is_bearish',
            'body_to_range', 'upper_shadow_to_range', 'lower_shadow_to_range',
            # Support/Resistance
            'norm_dist_to_support', 'norm_dist_to_resistance',
            'support_strength', 'resistance_strength',
            # Volatility and returns
            'daily_return', 'range_pct', 'volatility_5', 'volatility_10', 'volatility_20',
            'return_mean_5', 'return_mean_10', 'return_mean_20',
            # Volume features
            'volume_ratio', 'price_volume',
            # Trend features
            'trend_strength'
        ]
        
        # Get features that actually exist in the dataframe
        base_features = [col for col in existing_engineered_features if col in df.columns]
        
        # Add EMA columns dynamically
        ema_features = [col for col in df.columns if col.startswith('ema') and col not in base_features]
        base_features.extend(ema_features)
        
        # Add new advanced features
        new_features = [col for col in available_features if col not in base_features]
        final_features = base_features + new_features
        
        # If FEATURE_COLUMNS is defined in config, respect it but add our features
        if 'FEATURE_COLUMNS' in globals() and FEATURE_COLUMNS:
            config_features = [col for col in FEATURE_COLUMNS if col in df.columns]
            # Combine config features with our engineered features
            final_features = list(set(config_features + final_features))
        
        logger.info(f" Using {len(final_features)} features for training")
        

        
        # Prepare features and target
        X = df[final_features]
        y = df["target_hit"]

        print(y)
        
        # Final check for any remaining NaN values
        if X.isna().any().any():
            logger.warning("Found NaN values in features, filling with 0")
            X = X.fillna(0)
        
        # Handle class imbalance
        logger.info(" Handling class imbalance...")
        class_counts = y.value_counts()
        logger.info(f"   Class distribution: {class_counts.to_dict()}")
        
        # Check if we have both classes
        if len(class_counts) < 2:
            logger.error("Only one class found in target variable. Cannot train model.")
            raise ValueError("Only one class found in target variable")
        
        # Calculate class weights
        try:
            classes = np.unique(y)
            logger.info(f"   Unique classes (cleaned): {classes}")
            logger.info(f"   Classes dtype: {classes.dtype}")
            
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight_dict = dict(zip(classes, class_weights))
            logger.info(f"   Class weights: {class_weight_dict}")
        except Exception as e:
            logger.warning(f"   Could not compute class weights: {e}")
            class_weight_dict = None
        
        # Time-aware split (important for time series data)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X.to_csv('check.csv')

        print("Any NaN:", np.isnan(X_train).any())
        print("Any Inf:", np.isinf(X_train).any())
        print("Max Value:", np.nanmax(X_train))
        print("Min Value:", np.nanmin(X_train))
        
        logger.info(f" Training advanced model on {len(X_train)} samples, testing on {len(X_test)} samples...")
        
        # Check if we have enough data for training
        if len(X_train) == 0 or len(X_test) == 0:
            logger.error("Insufficient data for training/testing split")
            raise ValueError("Insufficient data for training/testing split")
        
        # Feature scaling
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(50, len(final_features)))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_features = X_train_scaled.columns[selector.get_support()]
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        logger.info(f" Selected {len(selected_features)} most important features")

    
        
        # Hyperparameter optimization
        rf_best, gb_best = optimize_hyperparameters(X_train_selected, y_train)
        
        # Create ensemble model
        ensemble_model = create_ensemble_model(rf_best, gb_best, X_train_selected, y_train)
        
        # Evaluate model
        metrics = evaluate_model_performance(ensemble_model, X_test_selected, y_test, X_train_selected, y_train)
        
        # Create model pipeline for deployment
        model_pipeline = {
            'scaler': scaler,
            'selector': selector,
            'model': ensemble_model,
            'selected_features': list(selected_features),
            'feature_columns': final_features,
            'metrics': metrics
        }
        
        # Save comprehensive model
        joblib.dump(model_pipeline, MODEL_FILE)
        logger.info(f" Advanced model pipeline saved to {MODEL_FILE}")
        
        # Trading recommendations
        logger.info("\n TRADING RECOMMENDATIONS:")
        if metrics['win_rate'] >= 60:
            logger.info("    Model shows strong predictive power - suitable for live trading")
        elif metrics['win_rate'] >= 55:
            logger.info("   Model shows moderate predictive power - use with caution")
        else:
            logger.info("    Model needs improvement - not recommended for live trading")
        
        logger.info(f"    Recommended position sizing: Conservative (model win rate: {metrics['win_rate']:.1f}%)")
        logger.info(f"    Suitable for swing trading timeframes: 2-10 days")
        
        return model_pipeline
        
    except Exception as e:
        logger.error(f" Failed to train advanced model: {e}", exc_info=True)
        raise e