"""
Configuration file for pillar design prediction model.
"""
from pathlib import Path

# =====================================================================
# PATHS
# =====================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dados_pilares.csv"
LOGS_DIR = PROJECT_ROOT / "logs"

# Paths for the two models
MODEL_PATH_CLASSIFIER = PROJECT_ROOT / "models" / "modelo_classificador.pkl"
MODEL_PATH_REGRESSOR = PROJECT_ROOT / "models" / "modelo_regressor.pkl"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
(PROJECT_ROOT / "models").mkdir(exist_ok=True)

# =====================================================================
# DATA CONFIGURATION
# =====================================================================
# Required columns from the dataset (Raw CSV inputs)
REQUIRED_COLUMNS = [
    'fck', 'PeDireito', 'largura', 'Altura','Cobrimento',
    'N_top', 'Mx_top', 'My_top', 'N_base', 'Mx_base', 'My_base', 'As'
]

# =====================================================================
# FEATURE COLUMNS FOR MODEL
# =====================================================================
# Features used for training
FEATURE_COLUMNS = [
    # --- 1. Dados Básicos ---
    'fck', 'PeDireito', 'largura', 'Altura', 'Cobrimento',
    
    # --- 2. Esforços Brutos ---
    'N_top', 'Mx_top', 'My_top', 'N_base', 'Mx_base', 'My_base',
    
    # --- 3. Variações (Gradientes Lineares) ---
    'N_med', 'Mx_med', 'My_med', 
    'dN', 'dMx', 'dMy',
    
    # --- 4. Variáveis Físicas (Ábacos) ---
    'nu',           # Carga Normal Adimensional
    'mu_x',         # Momento Adimensional X
    'mu_y',         # Momento Adimensional Y
    'mu_total',     # Magnitude do Momento
    'lambda_x',     # Esbeltez X
    'lambda_y',     # Esbeltez Y
    
    # --- 5. Excentricidades Geométricas ---
    'e_x', 'e_y',

    # --- 6. NOVAS FEATURES AVANÇADAS (Sugeridas) ---
    'ratio_M_x',         # Razão de momentos (Curvatura Simples/Dupla X)
    'ratio_M_y',         # Razão de momentos (Curvatura Simples/Dupla Y)
    'index_2nd_order_x', # Indicador de P-Delta (Carga x Esbeltez^2)
    'index_2nd_order_y',
    'aspect_ratio',      # Formato da seção (Retangularidade)
    'theta_moment'       # Ângulo da flexão oblíqua
]

# =====================================================================
# MODEL CONFIGURATION
# =====================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 50

# === CLASSIFIER PARAMETERS (Feasibility) ===
CLASSIFIER_PARAMS = {
    'objective': 'binary',        # Binary classification (Pass/Fail)
    'metric': 'auc',              # Area Under Curve (good for imbalanced classes)
    'is_unbalance': True,         # Handle class imbalance (if many more pass than fail)
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}

# === REGRESSOR PARAMETERS (Steel Area) ===
REGRESSOR_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 100,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 2000
}