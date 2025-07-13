# model/train_model.py

import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from config import FEATURE_COLUMNS, MODEL_FILE

logger = get_logger(__name__)

def validate_features(df):
    """Validate and report on available features"""
    logger.info("Validating available features...")
    
    # Focus on core feature categories only
    core_features = {
        'Technical Indicators': ['rsi', 'atr', 'bb_position'],
        'Price Movement': ['daily_return', 'range_pct'],
        'Volume': ['volume_ratio'],
        'Trend': ['trend_strength'],
        'Support/Resistance': ['norm_dist_to_support', 'norm_dist_to_resistance']
    }
    
    available_features = {}
    for category, features in core_features.items():
        available = [f for f in features if f in df.columns]
        if available:
            available_features[category] = available
    
    # Report available features
    logger.info("Available Core Features:")
    for category, features in available_features.items():
        logger.info(f"  {category}: {features}")
    
    total_available = sum(len(features) for features in available_features.values())
    logger.info(f"Total Core Features Available: {total_available}")
    
    return available_features

def create_basic_features(df):
    """Create only essential features to reduce overfitting"""
    logger.info("Creating basic features...")
    
    # Only add features that don't already exist
    existing_features = set(df.columns)
    
    # Essential price features
    if 'daily_return' not in existing_features:
        df['daily_return'] = df['close'].pct_change()
    
    if 'range_pct' not in existing_features:
        df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Simple volume feature
    if 'volume' in df.columns and 'volume_ratio' not in existing_features:
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # Basic trend feature
    if 'trend_strength' not in existing_features:
        df['price_ma_10'] = df['close'].rolling(10).mean()
        df['price_ma_20'] = df['close'].rolling(20).mean()
        df['trend_strength'] = (df['price_ma_10'] - df['price_ma_20']) / df['close']
    
    # Simple RSI-based features (if RSI exists)
    if 'rsi' in existing_features:
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    return df

def create_simple_model(X_train, y_train, class_weight_dict=None):
    """Create a simple, less complex model to reduce overfitting"""
    logger.info("Creating simple model...")
    
    # Simple Random Forest with conservative parameters
    rf_model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100-300
        max_depth=5,      # Limited depth
        min_samples_split=10,  # Increased to prevent overfitting
        min_samples_leaf=5,    # Increased to prevent overfitting
        max_features='sqrt',   # Reduced feature subset
        class_weight=class_weight_dict,
        random_state=42
    )
    
    # Simple Logistic Regression as backup
    lr_model = LogisticRegression(
        C=1.0,  # Not too regularized, not too complex
        max_iter=1000,
        class_weight=class_weight_dict,
        random_state=42
    )
    
    # Use cross-validation to select best model
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
    lr_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='roc_auc')
    
    logger.info(f"RF Cross-validation AUC: {rf_scores.mean():.3f} (+/- {rf_scores.std() * 2:.3f})")
    logger.info(f"LR Cross-validation AUC: {lr_scores.mean():.3f} (+/- {lr_scores.std() * 2:.3f})")
    
    # Select best model
    if rf_scores.mean() > lr_scores.mean():
        logger.info("Selected Random Forest model")
        best_model = rf_model
    else:
        logger.info("Selected Logistic Regression model")
        best_model = lr_model
    
    best_model.fit(X_train, y_train)
    return best_model

def evaluate_model_performance(model, X_test, y_test, X_train, y_train):
    """Simplified model evaluation"""
    logger.info("Evaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    report = classification_report(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Trading metrics
    true_positives = conf_matrix[1, 1]
    false_positives = conf_matrix[0, 1]
    false_negatives = conf_matrix[1, 0]
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    win_rate = precision * 100
    
    # Check for overfitting
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfitting_gap = train_score - test_score
    
    logger.info(f"\nMODEL PERFORMANCE METRICS:")
    logger.info(f"  Win Rate: {win_rate:.1f}%")
    logger.info(f"  AUC Score: {auc_score:.3f}")
    logger.info(f"  Precision: {precision:.3f}")
    logger.info(f"  Recall: {recall:.3f}")
    logger.info(f"  Train Accuracy: {train_score:.3f}")
    logger.info(f"  Test Accuracy: {test_score:.3f}")
    logger.info(f"  Overfitting Gap: {overfitting_gap:.3f}")
    
    if overfitting_gap > 0.1:
        logger.warning("  WARNING: Potential overfitting detected!")
    else:
        logger.info("  Model shows good generalization")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 5 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    return {
        'auc_score': auc_score,
        'win_rate': win_rate,
        'precision': precision,
        'recall': recall,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'overfitting_gap': overfitting_gap,
        'classification_report': report
    }

def train_model(df):
    """Simplified model training to reduce overfitting"""
    try:
        logger.info("Loading data for simplified model training...")
        df = df.copy()
        
        # Validate existing features
        available_features = validate_features(df)
        
        # Create only basic features
        df = create_basic_features(df)
        
        logger.info("Filtering and preparing training data...")
        
        # Check target column
        if 'target_hit' not in df.columns:
            logger.error("target_hit column not found in dataframe")
            raise ValueError("target_hit column not found in dataframe")
        
        # Data cleaning
        logger.info(f"Initial data shape: {df.shape}")
        df = df.dropna(subset=["target_hit"])
        df["target_hit"] = df["target_hit"].astype(int)
        
        # Filter to binary targets only
        df = df[df["target_hit"].isin([0, 1])]
        logger.info(f"After filtering to binary targets: {df.shape}")
        
        # Remove problematic columns
        cols_to_remove = [
            'days_to_target', 'days_to_target_3', 'days_to_target_5', 
            'days_to_target_8', 'days_to_target_10', 'trade_type_success',
            'price_atr_position', 'price_volume_correlation',
            'volume_momentum_10', 'volume_momentum_5',
            'price_ma_10', 'price_ma_20', 'volume_ma'  # Remove intermediate calculations
        ]
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])
        
        # Select only core features to prevent overfitting
        core_features = [
            'rsi', 'atr', 'bb_position',  # Technical indicators
            'daily_return', 'range_pct',  # Price movement
            'volume_ratio',  # Volume
            'trend_strength',  # Trend
            'norm_dist_to_support', 'norm_dist_to_resistance',  # Support/Resistance
            'rsi_oversold', 'rsi_overbought'  # Simple RSI features
        ]
        
        # Add any existing EMA features (but limit to main ones)
        ema_features = ['ema20_distance', 'ema50_distance', 'ema_spread']
        for ema_feat in ema_features:
            if ema_feat in df.columns:
                core_features.append(ema_feat)
        
        # Filter to features that actually exist
        final_features = [col for col in core_features if col in df.columns]
        
        # Remove features with too many missing values
        for feature in final_features.copy():
            if df[feature].isna().sum() / len(df) > 0.3:  # Remove if >30% missing
                logger.warning(f"Removing feature {feature} due to high missing values")
                final_features.remove(feature)
        
        logger.info(f"Using {len(final_features)} core features for training")
        logger.info(f"Selected features: {final_features}")
        
        # Prepare data
        df = df.dropna(subset=final_features)
        X = df[final_features]
        y = df["target_hit"]
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Check class distribution
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        if len(class_counts) < 2:
            logger.error("Only one class found in target variable")
            raise ValueError("Only one class found in target variable")
        
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Time-aware split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Simple feature scaling
        scaler = StandardScaler()
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
        
        # Feature selection - limit to top features only
        max_features = min(10, len(final_features))  # Limit to 10 features max
        selector = SelectKBest(score_func=f_classif, k=max_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_features = X_train_scaled.columns[selector.get_support()]
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        logger.info(f"Selected {len(selected_features)} most important features: {list(selected_features)}")
        
        # Train simple model
        model = create_simple_model(X_train_selected, y_train, class_weight_dict)
        
        # Evaluate model
        metrics = evaluate_model_performance(model, X_test_selected, y_test, X_train_selected, y_train)
        
        # Create simplified model pipeline
        model_pipeline = {
            'scaler': scaler,
            'selector': selector,
            'model': model,
            'selected_features': list(selected_features),
            'feature_columns': final_features,
            'metrics': metrics
        }
        
        # Save model
        joblib.dump(model_pipeline, MODEL_FILE)
        logger.info(f"Simplified model pipeline saved to {MODEL_FILE}")
        
        # Trading recommendations
        logger.info("\nTRADING RECOMMENDATIONS:")
        if metrics['overfitting_gap'] < 0.05:
            logger.info("  Model shows good generalization")
        else:
            logger.warning("  Model may still be overfitting - use with caution")
        
        if metrics['win_rate'] >= 55:
            logger.info("  Model shows reasonable predictive power")
        else:
            logger.info("  Model needs improvement - consider more data or different features")
        
        logger.info(f"  Conservative approach recommended (win rate: {metrics['win_rate']:.1f}%)")
        
        return model_pipeline
        
    except Exception as e:
        logger.error(f"Failed to train simplified model: {e}", exc_info=True)
        raise e