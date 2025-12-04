"""
Data loading module for pillar design dataset.
"""
import pandas as pd
import numpy as np
from .config import DATA_PATH, REQUIRED_COLUMNS
from .utils import setup_logger

logger = setup_logger(__name__)

def load_dataset() -> pd.DataFrame:
    """
    Load dataset, fix column names and flag unfeasible pillars (As=0).
    """
    logger.info(f"Loading dataset from: {DATA_PATH}")
    
    try:
        # 1. Carrega pulando a linha de metadados incorreta
        df = pd.read_csv(
            DATA_PATH,
            sep=';',           
            decimal=',',       
            skiprows=1,
            encoding='latin-1' 
        )
        
        # 2. Renomeia colunas
        rename_map = {
            'Pe direito': 'PeDireito',
            'N': 'N_top', 'Mx': 'Mx_top', 'My': 'My_top',
            'N.1': 'N_base', 'Mx.1': 'Mx_base', 'My.1': 'My_base'
        }
        df = df.rename(columns=rename_map)
        
        # 3. Seleciona colunas
        df = df[REQUIRED_COLUMNS].copy()
        
        # 4. Converte numéricos
        numeric_columns = [
            'fck', 'PeDireito', 'largura', 'Altura', 'Cobrimento',
            'N_top', 'Mx_top', 'My_top', 'N_base', 'Mx_base', 'My_base', 'As'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove erros de conversão (NaN)
        df = df.dropna()

        # === FLAG DE VIABILIDADE ===
        # Se As > 0, o pilar é viável (1).
        # Se As == 0, o pilar não passou (0).
        df['is_feasible'] = np.where(df['As'] > 0, 1, 0)
        
        count_fail = len(df[df['is_feasible'] == 0])
        count_ok = len(df[df['is_feasible'] == 1])
        
        logger.info(f"Data Loaded: {count_ok} Feasible (As>0), {count_fail} Failed (As=0)")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        raise

def get_data_info(df: pd.DataFrame) -> None:
    print(f"\n--- Data Info ---")
    print(f"Total Rows: {len(df)}")
    print(f"Feasible (Pass): {df['is_feasible'].sum()}")
    print(f"Unfeasible (Fail): {len(df) - df['is_feasible'].sum()}")