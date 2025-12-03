"""
Model training utility module.
Provides helper functions for splitting data, evaluating models, and saving/loading.
"""

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    accuracy_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

# REMOVIDO: MODEL_SAVE_PATH (não existe mais no config)
from .config import (
    EARLY_STOPPING_ROUNDS,
    RANDOM_STATE,
    TEST_SIZE,
    VERBOSE_EVAL,
)
from .utils import print_separator, setup_logger

logger = setup_logger(__name__)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Split data into training and validation sets.
    """
    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() < 10 else None
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Validation set size: {X_val.shape[0]}")
    
    return X_train, X_val, y_train, y_val


def evaluate_classifier(
    model: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> dict:
    """
    Evaluate the Feasibility Classifier (Stage 1).
    Metrics: AUC, Accuracy, Confusion Matrix.
    """
    print_separator("CLASSIFIER EVALUATION")
    logger.info("Evaluating classifier on validation set...")
    
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")
    
    return {"accuracy": acc, "auc": auc, "confusion_matrix": cm}


def evaluate_regressor(
    model: lgb.LGBMRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val_original: pd.DataFrame = None,
) -> dict:
    """
    Evaluate the Steel Area Regressor (Stage 2).
    Metrics: RMSE, MAE (for both rho and As).
    """
    print_separator("REGRESSOR EVALUATION")
    logger.info("Evaluating regressor on validation set...")
    
    y_pred_rho = model.predict(X_val)
    
    rmse_rho = np.sqrt(mean_squared_error(y_val, y_pred_rho))
    mae_rho = mean_absolute_error(y_val, y_pred_rho)
    
    print(f"\n--- Rho (Steel Ratio) Metrics ---")
    print(f"RMSE (rho): {rmse_rho:.6f}")
    print(f"MAE (rho):  {mae_rho:.6f}")
    
    metrics = {"rmse_rho": rmse_rho, "mae_rho": mae_rho}
    
    if df_val_original is not None and "Ac" in df_val_original.columns and "As" in df_val_original.columns:
        Ac = df_val_original["Ac"].values
        As_actual = df_val_original["As"].values
        As_pred = y_pred_rho * Ac
        
        rmse_as = np.sqrt(mean_squared_error(As_actual, As_pred))
        mae_as = mean_absolute_error(As_actual, As_pred)
        
        metrics["rmse_as"] = rmse_as
        metrics["mae_as"] = mae_as
        
        print(f"\n--- As (Reinforcement Area) Metrics ---")
        print(f"RMSE (As): {rmse_as:.2f} cm²")
        print(f"MAE (As):  {mae_as:.2f} cm²")
    
    return metrics


def print_feature_importance(model, feature_columns: list, top_n: int = 15) -> None:
    """Print feature importances."""
    print_separator("FEATURE IMPORTANCE")
    
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop {top_n} Features:")
    print(feature_importance_df.head(top_n).to_string(index=False))


def save_model(model, file_path: str) -> None:
    """Save trained model to disk (path is now mandatory)."""
    logger.info(f"Saving model to: {file_path}")
    try:
        joblib.dump(model, file_path)
        print(f"\n✓ Model saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(file_path: str):
    """Load trained model from disk."""
    logger.info(f"Loading model from: {file_path}")
    try:
        return joblib.load(file_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise