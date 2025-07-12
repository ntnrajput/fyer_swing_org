# model/train_model.py

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.logger import get_logger
from config import FEATURE_COLUMNS, MODEL_FILE

logger = get_logger(__name__)

def train_model(df):
    try:
        logger.info("📦 Loading data for model training...")
        df = df.copy()

        logger.info("🔍 Filtering training data...")
        df = df.dropna(subset=["target_hit"])
        df["target_hit"] = df["target_hit"].astype(int)

        # Split features & target
        X = df[FEATURE_COLUMNS]
        y = df["target_hit"]

        logger.info(f"🧠 Training model on {len(X)} samples...")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info("\n" + report)

        # Save model
        joblib.dump(model, MODEL_FILE)
        logger.info(f"✅ Model saved to {MODEL_FILE}")

    except Exception as e:
        logger.error(f"❌ Failed to train model: {e}", exc_info=True)
