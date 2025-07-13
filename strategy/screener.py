# # refactored_fyers_swing/strategy/screener.py

# import os
# import pandas as pd
# from datetime import datetime
# from config import DAILY_DATA_FILE, MODEL_FILE,FEATURE_COLUMNS, LATEST_DATA_FILE
# from data.fetch_stock_list import load_symbols
# from features.engineer_features import add_technical_indicators
# from model.prediction import load_model, predict
# from strategy.signal_generator import generate_signals
# from utils.logger import get_logger

# logger = get_logger(__name__)

# def run_screener():
#     """
#     Main screener function that loads today's data, predicts targets,
#     applies filters, and saves bullish stock suggestions.
#     """
#     try:
#         logger.info("=== Starting Daily Screener ===")

#         # Load today's data (assumed pre-downloaded in DAILY_DATA_FILE)
#         # df_today = pd.read_csv(DAILY_DATA_FILE, parse_dates=["date"])
#         # logger.info(f"Loaded today's data: {df_today.shape}")

#         # Feature engineering
#         df_features =  df = pd.read_parquet(LATEST_DATA_FILE)

#         # Load model and predict
#         model = load_model()
#         predictions = predict(model, df_features)

#         # Generate bullish signals
#         signals = generate_signals(predictions, df_features)

#         # Save results
#         today_str = datetime.now().strftime("%Y-%m-%d")
#         output_file = f"outputs/reports/signals_{today_str}.csv"
#         signals.to_csv(output_file, index=False)
#         logger.info(f"Signals saved to {output_file}")

#     except Exception as e:
#         logger.exception("Error in daily screener")









# refactored_fyers_swing/strategy/screener.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import DAILY_DATA_FILE, MODEL_FILE, FEATURE_COLUMNS, LATEST_DATA_FILE
from data.fetch_stock_list import load_symbols
from features.engineer_features import add_technical_indicators
from model.prediction import load_model, predict
from strategy.signal_generator import generate_signals
from utils.logger import get_logger
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

def load_advanced_model():
    """Load the advanced model pipeline"""
    try:
        if not os.path.exists(MODEL_FILE):
            logger.error(f"❌ Model file not found: {MODEL_FILE}")
            return None
            
        model_pipeline = joblib.load(MODEL_FILE)
        logger.info("✅ Advanced model pipeline loaded successfully")
        
        # Log model information
        if isinstance(model_pipeline, dict):
            logger.info(f"📊 Model contains: {list(model_pipeline.keys())}")
            if 'metrics' in model_pipeline:
                metrics = model_pipeline['metrics']
                logger.info(f"🎯 Model Win Rate: {metrics.get('win_rate', 'N/A'):.1f}%")
                logger.info(f"📈 Model AUC: {metrics.get('auc_score', 'N/A'):.3f}")
        
        return model_pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

def advanced_predict(model_pipeline, df_features):
    """Make predictions using the advanced model pipeline"""
    try:
        if not isinstance(model_pipeline, dict):
            logger.error("❌ Invalid model pipeline format")
            return None
            
        # Get required components
        scaler = model_pipeline.get('scaler')
        selector = model_pipeline.get('selector')
        model = model_pipeline.get('model')
        selected_features = model_pipeline.get('selected_features', [])
        
        if not all([scaler, selector, model]):
            logger.error("❌ Missing required model components")
            return None
            
        logger.info("🔮 Making predictions with advanced model...")
        
        # Prepare features (excluding target and metadata columns)
        exclude_cols = ['target_hit', 'date', 'symbol', 'max_return', 'min_return', 
                       'Swing_High', 'Swing_Low']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        # Ensure we have the required features
        available_features = [col for col in feature_cols if col in df_features.columns]
        
        if not available_features:
            logger.error("❌ No valid features found for prediction")
            return None
            
        X = df_features[available_features].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(0)
        
        # Scale features
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Select features
        X_selected = selector.transform(X_scaled)
        X_selected = pd.DataFrame(
            X_selected,
            columns=selected_features,
            index=X.index
        )
        
        # Make predictions
        predictions = model.predict(X_selected)
        prediction_probabilities = model.predict_proba(X_selected)
        
        # Create results dataframe
        results = pd.DataFrame({
            'symbol': df_features['symbol'],
            'date': df_features['date'],
            'close': df_features['close'],
            'prediction': predictions,
            'probability': prediction_probabilities[:, 1],  # Probability of positive class
            'confidence': np.max(prediction_probabilities, axis=1)
        })
        
        logger.info(f"✅ Predictions completed for {len(results)} stocks")
        logger.info(f"📊 Bullish signals: {sum(predictions)} out of {len(predictions)}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return None

def apply_advanced_filters(predictions, df_features):
    """Apply advanced filtering criteria for swing trading"""
    try:
        logger.info("🔍 Applying advanced screening filters...")
        
        # Merge predictions with features
        filtered_df = predictions.merge(
            df_features[['symbol', 'date', 'close', 'volume', 'rsi', 'atr', 'bb_position',
                        'volume_ratio', 'trend_strength', 'ema20_distance', 'ema50_distance']],
            on=['symbol', 'date'],
            how='left'
        )
        
        initial_count = len(filtered_df)
        
        # Filter 1: Only bullish predictions with high confidence
        filtered_df = filtered_df[
            (filtered_df['prediction'] == 1) & 
            (filtered_df['probability'] >= 0.6)
        ]
        logger.info(f"📈 After bullish + confidence filter: {len(filtered_df)} stocks")
        
        # Filter 2: Technical criteria
        technical_filters = (
            # RSI not overbought
            (filtered_df['rsi'] < 75) &
            # Above EMA20 or close to it (uptrend)
            (filtered_df['ema20_distance'] >= -0.02) &
            # Good volume (above average)
            (filtered_df['volume_ratio'] >= 1.2) &
            # Not in extreme overbought territory (BB position)
            (filtered_df['bb_position'] < 0.9) &
            # Reasonable volatility (ATR)
            (filtered_df['atr'] / filtered_df['close'] < 0.05)
        )
        
        filtered_df = filtered_df[technical_filters]
        logger.info(f"🎯 After technical filters: {len(filtered_df)} stocks")
        
        # Filter 3: Risk management filters
        risk_filters = (
            # Minimum price (avoid penny stocks)
            (filtered_df['close'] >= 10) &
            # Maximum price (liquidity concerns)
            (filtered_df['close'] <= 5000) &
            # Trend alignment
            (filtered_df['trend_strength'] >= 0)
        )
        
        filtered_df = filtered_df[risk_filters]
        logger.info(f"⚖️ After risk filters: {len(filtered_df)} stocks")
        
        # Sort by probability (highest first)
        filtered_df = filtered_df.sort_values('probability', ascending=False)
        
        # Add screening score
        filtered_df['screening_score'] = (
            filtered_df['probability'] * 0.4 +
            (filtered_df['confidence'] * 0.3) +
            (np.clip(filtered_df['volume_ratio'], 0, 3) / 3 * 0.2) +
            (np.clip((80 - filtered_df['rsi']) / 50, 0, 1) * 0.1)
        )
        
        # Re-sort by screening score
        filtered_df = filtered_df.sort_values('screening_score', ascending=False)
        
        logger.info(f"🎉 Final filtered stocks: {len(filtered_df)}")
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"❌ Filtering failed: {e}")
        return predictions

def generate_trading_insights(filtered_stocks, df_features):
    """Generate detailed trading insights for filtered stocks"""
    try:
        logger.info("💡 Generating trading insights...")
        
        insights = []
        
        for _, stock in filtered_stocks.iterrows():
            symbol = stock['symbol']
            
            # Get additional features for this stock
            stock_data = df_features[df_features['symbol'] == symbol].iloc[-1]
            
            # Generate insight
            insight = {
                'symbol': symbol,
                'current_price': stock['close'],
                'prediction_probability': stock['probability'],
                'screening_score': stock['screening_score'],
                'rsi': stock_data.get('rsi', 0),
                'trend_strength': stock_data.get('trend_strength', 0),
                'volume_ratio': stock['volume_ratio'],
                'ema20_distance': stock['ema20_distance'],
                'bb_position': stock['bb_position'],
                
                # Trading recommendations
                'entry_price': stock['close'],
                'target_price': stock['close'] * 1.05,  # 5% target
                'stop_loss': stock['close'] * 0.97,    # 3% stop loss
                'risk_reward_ratio': 1.67,  # 5% target / 3% stop loss
                
                # Trade rationale
                'rationale': f"High probability signal ({stock['probability']:.1%}) with favorable technical setup"
            }
            
            insights.append(insight)
        
        return pd.DataFrame(insights)
        
    except Exception as e:
        logger.error(f"❌ Insight generation failed: {e}")
        return pd.DataFrame()

def save_screening_results(signals, insights, today_str):
    """Save screening results in multiple formats"""
    try:
        # Create output directory
        output_dir = "outputs/reports"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main signals
        signals_file = f"{output_dir}/signals_{today_str}.csv"
        signals.to_csv(signals_file, index=False)
        logger.info(f"📊 Signals saved to {signals_file}")
        
        # Save detailed insights
        insights_file = f"{output_dir}/insights_{today_str}.csv"
        insights.to_csv(insights_file, index=False)
        logger.info(f"💡 Insights saved to {insights_file}")
        
        # Save summary report
        summary_file = f"{output_dir}/summary_{today_str}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"SWING TRADING SCREENER REPORT - {today_str}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Stocks Screened: {len(signals)}\n")
            f.write(f"Bullish Signals Generated: {len(insights)}\n")
            f.write(f"Average Prediction Probability: {insights['prediction_probability'].mean():.2%}\n")
            f.write(f"Average Screening Score: {insights['screening_score'].mean():.2f}\n\n")
            
            f.write("TOP 5 RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            for i, (_, stock) in enumerate(insights.head(5).iterrows(), 1):
                f.write(f"{i}. {stock['symbol']} - ₹{stock['current_price']:.2f} ")
                f.write(f"(Probability: {stock['prediction_probability']:.1%})\n")
        
        logger.info(f"📋 Summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")

def run_screener():
    """
    Advanced screener function that loads today's data, predicts targets,
    applies sophisticated filters, and saves actionable swing trading signals.
    """
    try:
        logger.info("=== 🚀 Starting Advanced Daily Screener ===")
        
        # Load today's data
        if not os.path.exists(LATEST_DATA_FILE):
            logger.error(f"❌ Data file not found: {LATEST_DATA_FILE}")
            return
            
        df_today = pd.read_parquet(LATEST_DATA_FILE)
        logger.info(f"📦 Loaded today's data: {df_today.shape}")
        
        # Ensure we have recent data
        if 'date' in df_today.columns:
            latest_date = pd.to_datetime(df_today['date']).max()
            days_old = (datetime.now() - latest_date).days
            if days_old > 5:
                logger.warning(f"⚠️ Data is {days_old} days old. Consider updating.")
        
        # Feature engineering (assuming this is already done if using LATEST_DATA_FILE)
        logger.info("🔧 Using pre-engineered features...")
        df_features = df_today.copy()
        
        # If features are not engineered, do it now
        required_features = ['rsi', 'atr', 'bb_position', 'ema20_distance']
        missing_features = [f for f in required_features if f not in df_features.columns]
        
        if missing_features:
            logger.info(f"🔧 Adding missing features: {missing_features}")
            df_features = add_technical_indicators(df_features)
        
        # Load advanced model
        model_pipeline = load_advanced_model()
        if model_pipeline is None:
            logger.error("❌ Cannot proceed without model")
            return
        
        # Make predictions
        predictions = advanced_predict(model_pipeline, df_features)
        if predictions is None:
            logger.error("❌ Cannot proceed without predictions")
            return
        
        # Apply advanced filters
        filtered_signals = apply_advanced_filters(predictions, df_features)
        
        # Generate trading insights
        insights = generate_trading_insights(filtered_signals, df_features)
        
        # Save results
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_screening_results(filtered_signals, insights, today_str)
        
        # Log final summary
        if len(insights) > 0:
            logger.info(f"🎯 SCREENING COMPLETE!")
            logger.info(f"📊 Found {len(insights)} high-probability swing trading opportunities")
            logger.info(f"🏆 Top recommendation: {insights.iloc[0]['symbol']} "
                       f"(Probability: {insights.iloc[0]['prediction_probability']:.1%})")
        else:
            logger.info("📊 No qualifying stocks found today. Market conditions may not be favorable.")
        
        logger.info("=== ✅ Advanced Screener Complete ===")
        
    except Exception as e:
        logger.error(f"❌ Error in advanced screener: {e}", exc_info=True)

if __name__ == "__main__":
    run_screener()
