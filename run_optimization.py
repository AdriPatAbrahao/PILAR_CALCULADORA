"""
Script de Calibração do Otimizador.
Objetivo: Verificar se o otimizador funciona corretamente usando um caso real (conhecido) do dataset.
"""
from src.predictor import PillarPredictor
from src.optimizer import PillarOptimizer
from src.utils import print_separator

def main():
    # --- 1. DADOS REAIS DO CSV (Pilar Conhecido) ---
    # Este pilar existe no seu arquivo e tem As = 20.1 cm²
    # Seção Real: 30 x 95 cm
    FIXED_PARAMS = {
        'fck': 50,           # Concreto forte do exemplo
        'PeDireito': 235,    # 2.35m
        'Altura': 95,        # Fixamos a altura em 95cm (ex: parede) e vamos otimizar a largura
        'Cobrimento': 2.5,
    }
    
    # Cargas Reais deste pilar
    LOAD_VECTOR = {
        'N_top': 392,   'Mx_top': 129,   'My_top': -92,
        'N_base': 392,  'Mx_base': 205,  'My_base': 430,
    }
    
    # --- 2. CONFIGURAÇÃO DO TESTE ---
    # Vamos pedir para o otimizador testar larguras de 20cm a 50cm.
    # Como a largura real é 30cm, o otimizador DEVE encontrar uma solução viável aqui.
    CONSTRAINTS = {
        'min_largura': 20,
        'max_largura': 50,
        'step': 5
    }
    
    COSTS = {
        'aco_kg': 12.00,       
        'concreto_m3': 450.00 
    }

    # --- 3. EXECUÇÃO ---
    print_separator("TESTE DE CALIBRAÇÃO DO OTIMIZADOR")
    print(f"Alvo: Otimizar largura para suportar cargas do pilar real (30x95).")
    print(f"Expectativa: A largura de 30cm (ou próxima) deve ser VIÁVEL e ECONÔMICA.")
    
    predictor = PillarPredictor()
    optimizer = PillarOptimizer(predictor)
    
    # Roda a otimização
    df_result = optimizer.find_optimal_width(FIXED_PARAMS, LOAD_VECTOR, CONSTRAINTS, COSTS)
    
    # --- 4. ANÁLISE DO RESULTADO ---
    print("\n--- RESULTADO DA VARREDURA (GRID SEARCH) ---")
    cols = ['largura', 'As_predicted', 'custo_total', 'is_feasible', 'prob_feasible']
    
    # Mostra todas as tentativas para você ver a curva de decisão
    print(df_result.sort_values('largura')[cols].to_string(index=False, float_format="%.2f"))
    
    best = df_result.iloc[0]
    if best['is_feasible'] == 1:
        print(f"\n✅ SUCESSO! O otimizador encontrou solução.")
        print(f"Melhor Largura: {best['largura']:.0f} cm (Custo: R$ {best['custo_total']:.2f})")
        
        if best['largura'] == 30:
            print("🎯 NA MOSCA! A IA convergiu exatamente para a largura do projeto original.")
        elif best['largura'] < 30:
            print("🚀 OTIMIZOU! A IA achou uma largura menor que a original que também passa.")
        else:
            print("⚠️ CONSERVADOR. A IA sugeriu uma largura maior que a original.")
    else:
        print("\n❌ FALHA. Mesmo com dados conhecidos, o modelo não aprovou nenhuma largura.")
        print("Isso indicaria que o Classificador está rejeitando o próprio dado de treino (Overfitting reverso ou erro de dados).")

if __name__ == "__main__":
    main()