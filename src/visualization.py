"""
Visualization module for Pillar Design.
Generates decision boundary plots (Interaction Diagrams and Design Maps).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .predictor import PillarPredictor
from .utils import setup_logger, print_separator

logger = setup_logger(__name__)

def plot_interaction_diagram(predictor, base_pillar, n_range, m_range, n_points=50):
    """
    Gera um Diagrama de Interação (Normal x Momento) para um pilar fixo.
    Mostra a região de segurança (Viável) vs Falha.
    """
    print_separator("GERANDO DIAGRAMA DE INTERAÇÃO (N x M)")
    
    # 1. Criar o Grid de Cargas
    N_values = np.linspace(n_range[0], n_range[1], n_points)
    M_values = np.linspace(m_range[0], m_range[1], n_points)
    
    batch_data = []
    for N in N_values:
        for M in M_values:
            # Copia o pilar base e altera as cargas
            pilar = base_pillar.copy()
            # Aplica a carga no topo e base (simplificação para o gráfico)
            pilar['N_top'] = N
            pilar['N_base'] = N
            pilar['Mx_top'] = M
            pilar['Mx_base'] = M # Considerando momento constante para simplificar visualização
            batch_data.append(pilar)
            
    # 2. Fazer Predição em Lote
    df_results = predictor.predict_batch(batch_data)
    
    # 3. Preparar dados para o Heatmap
    # Queremos uma matriz onde Z = Probabilidade de Viabilidade
    Z = df_results['prob_feasible'].values.reshape(n_points, n_points)
    
    # 4. Plotar
    plt.figure(figsize=(10, 8))
    
    # Heatmap de Probabilidade
    plt.contourf(M_values, N_values, Z, levels=20, cmap='RdYlGn', alpha=0.8)
    plt.colorbar(label='Probabilidade de Sucesso (%)')
    
    # Linha de Fronteira (Probabilidade = 50%)
    cs = plt.contour(M_values, N_values, Z, levels=[0.5], colors='black', linewidths=2)
    plt.clabel(cs, fmt='Fronteira (50%%)', inline=True)
    
    plt.title(f"Fronteira de Resistência - Pilar {base_pillar['largura']}x{base_pillar['Altura']} cm")
    plt.xlabel('Momento Fletor (kNm)')
    plt.ylabel('Carga Axial (kN)')
    plt.grid(True, alpha=0.3)
    
    filename = "grafico_interacao_nm.png"
    plt.savefig(filename)
    print(f"Gráfico salvo como: {filename}")
    plt.close()


def plot_section_boundary(predictor, base_loads, w_range, h_range, n_points=50):
    """
    Gera um Mapa de Otimização (Largura x Altura) para cargas fixas.
    Mostra qual seção mínima é necessária.
    """
    print_separator("GERANDO MAPA DE OTIMIZAÇÃO (Seção B x H)")
    
    # 1. Criar o Grid de Geometria
    W_values = np.linspace(w_range[0], w_range[1], n_points) # Larguras
    H_values = np.linspace(h_range[0], h_range[1], n_points) # Alturas
    
    batch_data = []
    for H in H_values: # Y axis (rows)
        for W in W_values: # X axis (cols)
            pilar = base_loads.copy()
            pilar['largura'] = W
            pilar['Altura'] = H
            # As precisa existir no dicionário, mesmo que seja dummy para predição
            if 'As' not in pilar: pilar['As'] = 0 
            batch_data.append(pilar)
            
    # 2. Fazer Predição
    df_results = predictor.predict_batch(batch_data)
    
    # 3. Preparar Matriz Z
    Z_prob = df_results['prob_feasible'].values.reshape(n_points, n_points)
    
    # 4. Plotar
    plt.figure(figsize=(10, 8))
    
    # Heatmap
    # Nota: Usamos 'RdYlGn' (Vermelho=Ruim, Verde=Bom)
    plt.contourf(W_values, H_values, Z_prob, levels=20, cmap='RdYlGn', alpha=0.8)
    plt.colorbar(label='Probabilidade de Viabilidade')
    
    # Linha de Decisão
    cs = plt.contour(W_values, H_values, Z_prob, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    plt.clabel(cs, fmt='Limiar 50%%', inline=True)
    
    plt.title(f"Fronteira de Design - Carga N={base_loads['N_top']}kN, M={base_loads['Mx_top']}kNm")
    plt.xlabel('Largura (cm)')
    plt.ylabel('Altura (cm)')
    plt.grid(True, alpha=0.3)
    
    filename = "grafico_fronteira_secao.png"
    plt.savefig(filename)
    print(f"Gráfico salvo como: {filename}")
    plt.close()

if __name__ == "__main__":
    # Teste rápido se rodar o arquivo diretamente
    predictor = PillarPredictor()
    
    # Cargas de Teste (Pilar Real do seu CSV)
    cargas_reais = {
        'fck': 50, 'PeDireito': 200,
        'N_top': 27, 'Mx_top': 3, 'My_top': -28,
        'N_base': 27, 'Mx_base': -19, 'My_base': 0,
        'As': 0
    }
    
    # 1. Mapa de Seção: Variando de 10x10 até 60x60
    plot_section_boundary(predictor, cargas_reais, w_range=(10, 60), h_range=(10, 60))
    
    # 2. Diagrama N-M: Para um pilar fixo de 20x20
    pilar_base = cargas_reais.copy()
    pilar_base['largura'] = 20
    pilar_base['Altura'] = 20
    # Variando Normal de 0 a 2000 kN, Momento de 0 a 200 kNm
    plot_interaction_diagram(predictor, pilar_base, n_range=(0, 3000), m_range=(0, 300))