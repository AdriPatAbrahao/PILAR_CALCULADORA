"""
Optimizer module for Pillar Design.
Encontra a largura/seção que minimiza o Custo Global (Concreto + Aço).
"""

import numpy as np
import pandas as pd
import itertools
from .predictor import PillarPredictor
from .utils import setup_logger, print_separator

logger = setup_logger(__name__)

class PillarOptimizer:
    def __init__(self, predictor: PillarPredictor):
        self.predictor = predictor
        
    def find_optimal_width(self, fixed_params: dict, loads: dict, 
                          constraints: dict, costs: dict) -> pd.DataFrame:
        """
        Para um conjunto de cargas e altura fixos, encontra a LARGURA ideal.
        
        Args:
            fixed_params: Dict com 'fck', 'PeDireito', 'Altura' (se fixa), etc.
            loads: Dict com vetor de cargas {'N_top', 'Mx_top', ...}
            constraints: Limites {'min_largura': 15, 'max_largura': 80, 'step': 5}
            costs: Preços {'aco_kg': 12.0, 'concreto_m3': 450.0}
            
        Returns:
            DataFrame com todas as opções ordenadas pelo menor custo.
        """
        logger.info("Iniciando otimização de custo...")
        
        # 1. Gerar Grid de Larguras
        # Vamos variar a 'largura' conforme os limites (ex: 15, 20, 25... 80)
        larguras = range(constraints['min_largura'], 
                         constraints['max_largura'] + 1, 
                         constraints['step'])
        
        candidates = []
        for b in larguras:
            # Monta o candidato mesclando Cargas + Parâmetros Fixos
            candidate = {**fixed_params, **loads}
            
            # Define a geometria variável
            candidate['largura'] = b
            
            # Se 'Altura' não foi fixada, assumimos que é igual à largura (pilar quadrado)
            # ou o usuário deve passar um range de Alturas também (aqui simplificado para largura)
            if 'Altura' not in candidate:
                candidate['Altura'] = b 
                
            # Dummy As para o modelo rodar (será previsto depois)
            candidate['As'] = 0 
            candidates.append(candidate)
            
        # 2. Predição em Massa (IA)
        # O modelo calcula a viabilidade e a área de aço para todas as larguras de uma vez
        df_results = self.predictor.predict_batch(candidates)
        
        # Recupera as dimensões para cálculo de custo
        df_results['largura'] = [c['largura'] for c in candidates]
        df_results['Altura']  = [c['Altura'] for c in candidates]
        # PeDireito e fck são constantes neste loop, pegamos do primeiro
        pe_direito = fixed_params['PeDireito']
        
        # 3. Cálculo de Quantitativos e Custos
        
        # A. Volume de Concreto (m³)
        # V = b * h * L (tudo convertido para metros)
        vol_concreto = (df_results['largura']/100) * (df_results['Altura']/100) * (pe_direito/100)
        cost_concreto = vol_concreto * costs['concreto_m3']
        
        # B. Peso de Aço (kg)
        # Peso = As (cm²) * Comprimento (m) * Densidade Linear Aprox
        # Densidade do aço ~ 7850 kg/m³. 
        # 1 cm² de aço em 1 m de barra = 1e-4 m² * 1 m * 7850 kg/m³ = 0.785 kg
        peso_aco = df_results['As_predicted'] * (pe_direito/100) * 0.785
        cost_aco = peso_aco * costs['aco_kg']
        
        # 4. Composição do Custo Total
        df_results['custo_concreto'] = cost_concreto
        df_results['custo_aco'] = cost_aco
        df_results['custo_total'] = cost_concreto + cost_aco
        
        # 5. Penalização (Pilar Inviável)
        # Se o classificador (Fiscal) disse que não passa, custo vira infinito
        # Se a probabilidade for muito baixa (<50%), também penalizamos
# 5. Penalização (Pilar Inviável)
        mask_inviavel = (df_results['is_feasible'] == 0) | (df_results['prob_feasible'] < 0.5)
        
        # === DEBUG: VER O QUE ESTÁ ACONTECENDO ===
        print("\n--- DEBUG OTIMIZADOR (Primeiras 10 tentativas) ---")
        # Mostra largura, probabilidade e se o modelo achou viável (antes de penalizar)
        cols_debug = ['largura', 'Altura', 'prob_feasible', 'is_feasible', 'As_predicted']
        print(df_results[cols_debug].head(10).to_string(index=False))
        print("-" * 50)
        # ==========================================

        df_results.loc[mask_inviavel, 'custo_total'] = float('inf')
        
        # 6. Ordenação
        df_final = df_results.sort_values('custo_total').reset_index(drop=True)
        
        return df_final