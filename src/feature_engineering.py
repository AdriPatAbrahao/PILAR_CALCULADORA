"""
Feature engineering module for pillar design prediction.
Creates engineered features from raw data.
"""

import numpy as np
import pandas as pd

from .utils import setup_logger

logger = setup_logger(__name__)


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features based on Interaction Diagram (Abacus) variables.
    
    Features created:
    - N_med, Mx_med, My_med: Average efforts
    - dN, dMx, dMy: Effort variations
    - nu: Dimensionless axial force (Nd / (Ac * fcd))
    - mu_x: Dimensionless moment X (Mxd / (Ac * h_x * fcd))
    - mu_y: Dimensionless moment Y (Myd / (Ac * h_y * fcd))
    - lambda_x, lambda_y: Slenderness
    """
    df = df.copy()
    
    logger.info("Creating engineered features (Physics-Informed + Variations)...")
    
    try:
        # 1. Recuperar valores brutos
        largura = df["largura"].values   # Dimensão X (b)
        altura = df["Altura"].values     # Dimensão Y (h)
        pe_direito = df["PeDireito"].values
        fck = df["fck"].values
        
        N_top = df["N_top"].values
        N_base = df["N_base"].values
        Mx_top = df["Mx_top"].values
        Mx_base = df["Mx_base"].values
        My_top = df["My_top"].values
        My_base = df["My_base"].values
        
        # --- CÁLCULOS DE MÉDIA E VARIAÇÃO (REINSERIDOS) ---
        # Necessários pois estão no config.py
        df["N_med"] = (N_top + N_base) / 2
        df["Mx_med"] = (Mx_top + Mx_base) / 2
        df["My_med"] = (My_top + My_base) / 2
        
        df["dN"] = N_base - N_top
        df["dMx"] = Mx_base - Mx_top
        df["dMy"] = My_base - My_top
        # --------------------------------------------------

        # Esforços Máximos (para o cálculo físico)
        N_max = np.maximum(np.abs(N_top), np.abs(N_base))
        Mx_max = np.maximum(np.abs(Mx_top), np.abs(Mx_base))
        My_max = np.maximum(np.abs(My_top), np.abs(My_base))
        
        # 2. Definições de Cálculo (Norma)
        GAMMA_F = 1.4  # Majoração de cargas
        GAMMA_C = 1.4  # Minoração do concreto
        
        # fcd = fck / 1.4 (Convertendo MPa para kN/cm²: divide por 10)
        fcd = (fck / GAMMA_C) / 10.0
        
        Ac = largura * altura
        df["Ac"] = Ac
        
        # 3. Variável NU (Normal Reduzida)
        # v = Nd / (Ac * fcd)
        Nd = N_max * GAMMA_F
        df["nu"] = Nd / (Ac * fcd)
        
        # 4. Variáveis MU (Momentos Reduzidos)
        Mxd = Mx_max * GAMMA_F * 100 # *100 para passar kNm para kNcm
        Myd = My_max * GAMMA_F * 100
        
        # mu_x usa a dimensão 'Altura' como braço de alavanca (gira em torno de X)
        df["mu_x"] = Mxd / (Ac * altura * fcd)
        
        # mu_y usa a dimensão 'largura' como braço de alavanca (gira em torno de Y)
        df["mu_y"] = Myd / (Ac * largura * fcd)
        
        # 5. Esbeltez (Lambda)
        df["lambda_x"] = 3.46 * pe_direito / largura
        df["lambda_y"] = 3.46 * pe_direito / altura
        
        # 6. Variável Composta
        df["mu_total"] = np.sqrt(df["mu_x"]**2 + df["mu_y"]**2)
        
        # Features originais de excentricidade
        df["e_x"] = np.where(N_max != 0, Mx_max / N_max, 0)
        df["e_y"] = np.where(N_max != 0, My_max / N_max, 0)

        logger.info("Physics-informed features created successfully.")

        # --- SUGESTÕES AVANÇADAS DE ENGENHARIA ---
        
        # 1. Gradiente de Momento (Razão Topo/Base)
        # Adicionamos um epsilon para não dividir por zero
        eps = 1e-6
        df['ratio_M_x'] = df['Mx_top'] / (df['Mx_base'] + eps)
        df['ratio_M_y'] = df['My_top'] / (df['My_base'] + eps)
        
        # 2. Proxy de Efeito de 2ª Ordem (Interação N-Linear Carga x Esbeltez)
        # O momento de 2ª ordem é proporcional a N * lambda²
        df['index_2nd_order_x'] = df['nu'] * (df['lambda_x']**2)
        df['index_2nd_order_y'] = df['nu'] * (df['lambda_y']**2)
        
        # 3. Geometria (Retangularidade)
        # Pilares parede vs Pilares quadrados têm confinamento diferente
        df['aspect_ratio'] = np.maximum(altura/largura, largura/altura)
        
        # 4. Direção da Flexão (Ângulo da carga no ábaco)
        # Ajuda a entender se estamos na ponta da "roseta" de interação ou na face
        df['theta_moment'] = np.arctan2(df['mu_y'], df['mu_x'])
        
        logger.info("Advanced structural features created: ratios, 2nd_order, aspect, theta")
        
    except Exception as e:
        logger.error(f"Error creating engineered features: {e}", exc_info=True)
        raise
    
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable rho (steel ratio).
    """
    df = df.copy()
    logger.info("Creating target variable (rho)...")
    try:
        As = df["As"].values
        Ac = df["Ac"].values
        df["rho"] = As / Ac
        
        logger.info(f"Target variable created. Mean rho: {df['rho'].mean():.6f}")
    except Exception as e:
        logger.error(f"Error creating target variable: {e}", exc_info=True)
        raise
    return df


def prepare_features(df: pd.DataFrame, feature_columns: list) -> tuple:
    """
    Prepare feature matrix and target vector.
    """
    try:
        # Ensure Ac is NOT in features
        features_clean = [col for col in feature_columns if col != 'Ac']
        
        logger.info(f"Features for training: {features_clean}")
        
        # Check if all columns exist
        missing = set(features_clean) - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns in DataFrame: {missing}")
        
        X = df[features_clean].copy()
        y = df["rho"].copy()
        
        logger.info(f"Features prepared. Shape: {X.shape}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}", exc_info=True)
        raise