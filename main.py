"""
Main script: Two-Stage Training Pipeline (Classifier + Regressor)
"""
import pandas as pd
import lightgbm as lgb
from src.config import (
    FEATURE_COLUMNS, CLASSIFIER_PARAMS, REGRESSOR_PARAMS,
    MODEL_PATH_CLASSIFIER, MODEL_PATH_REGRESSOR
)
from src.data_loader import get_data_info, load_dataset
from src.feature_engineering import (
    create_engineered_features,
    create_target_variable,
    prepare_features,
)
from src.model_trainer import (
    evaluate_classifier,
    evaluate_regressor,
    split_data, 
    save_model, 
    print_feature_importance
)
from src.utils import print_separator, setup_logger

logger = setup_logger(__name__)


def train_classifier(df: pd.DataFrame, X: pd.DataFrame) -> lgb.LGBMClassifier:
    """
    Trains the Feasibility Classifier.
    Predicts if a pillar configuration is viable (1) or will fail (0).
    """
    print_separator("STAGE 1: TRAINING CLASSIFIER (FEASIBILITY)")
    
    # Target is the feasibility flag
    y = df['is_feasible']
    
    # Split data (using stratify to maintain class balance)
    X_train, X_val, y_train, y_val = split_data(X, y)
    
    # Initialize and Train
    model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50)
        ]
    )
    
    # Evaluate
    evaluate_classifier(model, X_val, y_val)
    
    # Save model
    save_model(model, str(MODEL_PATH_CLASSIFIER))
    
    # Feature Importance (Optional for classifier)
    print("Classifier Feature Importance:")
    print_feature_importance(model, FEATURE_COLUMNS, top_n=5)
    
    return model


def train_regressor(df: pd.DataFrame, X: pd.DataFrame) -> lgb.LGBMRegressor:
    """
    Trains the Steel Area Regressor.
    Predicts 'rho' (steel ratio) ONLY for feasible pillars.
    """
    print_separator("STAGE 2: TRAINING REGRESSOR (STEEL AREA)")
    
    # FILTER: Only train on feasible pillars (is_feasible == 1)
    mask_feasible = df['is_feasible'] == 1
    
    X_feasible = X[mask_feasible].copy()
    y_feasible = df.loc[mask_feasible, 'rho'].copy() # Use rho as target
    df_original_feasible = df[mask_feasible].copy()
    
    print(f"Training Regressor on {len(X_feasible)} feasible samples.")
    
    if len(X_feasible) < 10:
        logger.warning("Not enough feasible samples to train regressor!")
        return None

    # Split data
    X_train, X_val, y_train, y_val = split_data(X_feasible, y_feasible)
    
    # Keep original data for validation to calculate error in cm²
    df_val_original = df_original_feasible.loc[X_val.index]
    
    # Initialize and Train
    model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50)
        ]
    )
    
    # Evaluate
    evaluate_regressor(model, X_val, y_val, df_val_original)
    
    # Feature Importance
    print("Regressor Feature Importance:")
    print_feature_importance(model, FEATURE_COLUMNS, top_n=10)
    
    # Save model
    save_model(model, str(MODEL_PATH_REGRESSOR))
    return model


def main() -> None:
    """
    Main execution pipeline.
    """
    try:
        print_separator("STARTING PILLAR DESIGN AI TRAINING")
        
        # =====================================================================
        # 1. LOAD & ENGINEER DATA
        # =====================================================================
        df = load_dataset()
        get_data_info(df)
        
        # Create engineered features
        df = create_engineered_features(df)
        
        # Create target variable (rho)
        df = create_target_variable(df)
        
        # Prepare feature matrix X (common to both models)
        X, _ = prepare_features(df, FEATURE_COLUMNS)
        
        # =====================================================================
        # 2. TRAIN MODELS
        # =====================================================================
        
        # A. Train Classifier (The "Inspector")
        train_classifier(df, X)
        
        # B. Train Regressor (The "Engineer")
        train_regressor(df, X)
        
        logger.info("Pipeline Finished Successfully.")
        print_separator("TRAINING COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()