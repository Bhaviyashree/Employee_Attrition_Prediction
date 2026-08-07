"""
Train Random Forest and XGBoost classifiers for employee attrition prediction.
Saves best model and preprocessing artifacts to model/ directory.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing import (  # noqa: E402
    MODEL_DIR,
    clean_dataset,
    encode_categorical_features,
    load_dataset,
    prepare_features,
    save_preprocessing_artifacts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_models(test_size: float = 0.2, random_state: int = 42) -> dict:
    """Train RF and XGBoost, compare metrics, save best model."""
    logger.info("Loading and preprocessing dataset...")
    df = clean_dataset(load_dataset())
    encoded_df, label_encoders = encode_categorical_features(df, fit=True)

    # Target: Attrition Yes=1, No=0
    if "Attrition" in encoded_df.columns:
        target = encoded_df["Attrition"]
        if target.dtype == object or target.max() > 1:
            le_attr = label_encoders.get("Attrition")
            if le_attr:
                target = pd.Series(le_attr.transform(df["Attrition"].astype(str)))
            else:
                target = df["Attrition"].map({"Yes": 1, "No": 0})
    else:
        raise ValueError("Attrition column required for training")

    feature_cols = [c for c in encoded_df.columns if c != "Attrition"]
    X = encoded_df[feature_cols]
    y = target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
    }

    results: dict = {}
    best_model = None
    best_name = ""
    best_auc = 0.0

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        if hasattr(model, "feature_importances_"):
            metrics["feature_importances"] = {
                col: float(imp)
                for col, imp in zip(feature_cols, model.feature_importances_)
            }

        results[name] = metrics
        logger.info(
            "%s - Accuracy: %.4f, F1: %.4f, ROC AUC: %.4f",
            name,
            metrics["accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = model
            best_name = name

    # Persist artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "attrition_model.pkl")
    save_preprocessing_artifacts(scaler, label_encoders, feature_cols)

    # Save model metadata
    metadata = {
        "best_model": best_name,
        "metrics": results,
        "feature_names": feature_cols,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save test predictions for dashboard charts
    np.savez(
        MODEL_DIR / "test_predictions.npz",
        y_true=y_test.values,
        y_pred=best_model.predict(X_test_scaled),
        y_prob=best_model.predict_proba(X_test_scaled)[:, 1],
    )

    logger.info("Best model: %s (ROC AUC: %.4f)", best_name, best_auc)
    logger.info("Artifacts saved to %s", MODEL_DIR)
    return metadata


if __name__ == "__main__":
    train_models()
