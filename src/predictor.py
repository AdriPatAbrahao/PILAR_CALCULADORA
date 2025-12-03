"""
Predictor module for Two-Stage Inference (Classifier + Regressor).
"""

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, MODEL_PATH_CLASSIFIER, MODEL_PATH_REGRESSOR
from .feature_engineering import create_engineered_features
from .model_trainer import load_model
from .utils import setup_logger

logger = setup_logger(__name__)


class PillarPredictor:
    """
    Predictor class handling the 2-stage pipeline:
    1. Feasibility Check (Classifier)
    2. Steel Area Calculation (Regressor)
    """
    
    def __init__(self, classifier_path=None, regressor_path=None):
        """
        Initialize predictor by loading both models.
        """
        logger.info("Initializing PillarPredictor...")
        
        path_clf = classifier_path or MODEL_PATH_CLASSIFIER
        path_reg = regressor_path or MODEL_PATH_REGRESSOR
        
        self.classifier = load_model(path_clf)
        self.regressor = load_model(path_reg)
        
        logger.info("Both models loaded successfully.")
    
    def predict_single(self, pillar_data: dict) -> dict:
        """
        Predict for a single pillar.
        Returns 'status': 'Infeasible' or 'Feasible'.
        """
        try:
            # Prepare Data
            df = pd.DataFrame([pillar_data])
            df_eng = self._process_pillar_data(df)
            X = df_eng[FEATURE_COLUMNS]
            Ac = df_eng['Ac'].values[0]
            
            # --- STAGE 1: CLASSIFIER ---
            is_feasible = self.classifier.predict(X)[0]
            prob_feasible = self.classifier.predict_proba(X)[0][1]
            
            result = {
                'status': 'Feasible' if is_feasible == 1 else 'Infeasible',
                'feasibility_prob': prob_feasible,
                'Ac': Ac,
                'As_actual': pillar_data.get('As', 0)
            }
            
            if is_feasible == 0:
                # Se não passa, não calculamos aço (ou retornamos infinito/zero)
                result['rho_predicted'] = 0.0
                result['As_predicted'] = 0.0 # Indica falha
                result['message'] = "Pillar geometry/loads failed feasibility check."
            else:
                # --- STAGE 2: REGRESSOR ---
                rho_pred = self.regressor.predict(X)[0]
                As_pred = rho_pred * Ac
                
                result['rho_predicted'] = rho_pred
                result['As_predicted'] = As_pred
                
                # Calculate Error if actual As exists
                if result['As_actual'] > 0:
                    result['error'] = As_pred - result['As_actual']
                    result['error_pct'] = (result['error'] / result['As_actual']) * 100
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single prediction: {e}", exc_info=True)
            raise

    def predict_batch(self, pillars_data: list) -> pd.DataFrame:
        """
        Predict for multiple pillars efficiently.
        """
        try:
            df = pd.DataFrame(pillars_data)
            df_eng = self._process_pillar_data(df)
            X = df_eng[FEATURE_COLUMNS]
            Ac = df_eng['Ac'].values
            
            # 1. Classify all
            feasibility = self.classifier.predict(X)
            probs = self.classifier.predict_proba(X)[:, 1]
            
            # 2. Regress all (we can filter later, but predicting all is vector-efficient)
            rho_preds = self.regressor.predict(X)
            As_preds = rho_preds * Ac
            
            # 3. Mask unfeasible results
            # If not feasible, set As to 0 (or NaN)
            final_As = np.where(feasibility == 1, As_preds, 0)
            final_rho = np.where(feasibility == 1, rho_preds, 0)
            
            results_df = pd.DataFrame({
                'is_feasible': feasibility,
                'prob_feasible': probs,
                'rho_predicted': final_rho,
                'As_predicted': final_As,
                'As_actual': df.get('As', np.zeros(len(df))),
                'Ac': Ac
            })
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}", exc_info=True)
            raise
    
    def _process_pillar_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        df_processed = create_engineered_features(df_processed)
        return df_processed