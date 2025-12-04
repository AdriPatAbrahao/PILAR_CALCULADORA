"""
Example usage of the Two-Stage PillarPredictor using REAL DATA from CSV.
"""

from .predictor import PillarPredictor
from .utils import print_separator, setup_logger

logger = setup_logger(__name__)

def example_single_prediction():
    """
    Test: Predict for a SINGLE REAL PILLAR from the dataset.
    We use a known row to compare the Model vs Reality.
    """
    print_separator("TEST 1: SINGLE PILLAR PREDICTION (REAL DATA)")
    
    predictor = PillarPredictor()
    
    # === DADOS REAIS DO CSV (Linha com As=4.7) ===
    # Pilar: 200x45x15 (PeDireito x Largura x Altura)
    # Fck: 50 MPa
    pillar_real = {
        'fck': 50, 
        'PeDireito': 235, 
        'largura': 30, 
        'Altura': 95,
        'Cobrimento': 2.5,
        'N_top': 392,   'Mx_top': 129,   'My_top': -92,
        'N_base': 392,  'Mx_base': 205, 'My_base': 430,
        'As': 20.1 # Valor alvo real
    }
    
    result = predictor.predict_single(pillar_real)
    
    print("\n--- Pilar Real do CSV (30x95 cm) ---")
    print(f"Status:       {result['status']} (Prob: {result['feasibility_prob']:.1%})")
    
    if result['status'] == 'Feasible':
        print(f"As Calculado: {result['As_predicted']:.2f} cm²")
        print(f"As Real (CSV):{result['As_actual']:.2f} cm²")
        
        diff = result.get('error', 0)
        pct = result.get('error_pct', 0)
        print(f"Diferença:    {diff:+.2f} cm² ({pct:+.1f}%)")
        
        if abs(diff) < 1.0:
            print(">> EXCELENTE PRECISÃO! O modelo reproduziu o cálculo original.")
        elif abs(diff) < 3.0:
            print(">> BOA PRECISÃO. Diferença aceitável para estimativa.")
    else:
        print("ALERTA: O modelo classificou este pilar real como INVIÁVEL.")


def example_batch_test():
    """
    Test: Predict for a batch including the real pillar and a hypothetical fail case.
    """
    print_separator("TEST 2: BATCH PREDICTION")
    predictor = PillarPredictor()
    pillars = [
        # 1. O Pilar Real (deve passar)
        {'fck': 50, 'PeDireito': 235, 'largura': 30, 'Altura': 95, 'Cobrimento': 2.5,   
         'N_top': 392, 'Mx_top': 129, 'My_top': -92,
         'N_base': 392, 'Mx_base': 205, 'My_base': 430, 'As': 20.1},
         
        # 2. Pilar Hipotético: Mesma geometria, mas CARGA EXTREMA (deve falhar)
        {'fck': 50, 'PeDireito': 200, 'largura': 45, 'Altura': 15, 'Cobrimento': 2.5,
         'N_top': 5000, 'Mx_top': 300, 'My_top': 200, 
         'N_base': 5000, 'Mx_base': 300, 'My_base': 200, 'As': 0}
    ]
    
    df_res = predictor.predict_batch(pillars)
    
    print(f"{'Caso':<10} {'Status':<12} {'Prob':<8} {'As_Pred':<10} {'As_Real':<10}")
    print("-" * 55)
    
    cases = ["Real", "Extremo"]
    for i, row in df_res.iterrows():
        status = "Feasible" if row['is_feasible'] == 1 else "Infeasible"
        print(f"{cases[i]:<10} {status:<12} {row['prob_feasible']:.1%}   {row['As_predicted']:<10.2f} {row['As_actual']:<10.2f}")


def example_compare_variations():
    """
    Test: Compare predictions for same loads with different dimensions.
    Using the REAL LOADS from the CSV pillar.
    """
    print_separator("TEST 3: OTIMIZAÇÃO (VARIAÇÃO DE SEÇÃO)")
    
    predictor = PillarPredictor()
    
    # Cargas do Pilar Real
    base_loads = {
        'fck': 30, 'Cobrimento': 3.0,
        'N_top': 500, 'Mx_top': 10, 'My_top': 80, 
        'N_base': 500, 'Mx_base': 10, 'My_base': 80,
    }
    
    # Tentando otimizar a seção original (45x15)
    variations = [
        {'PeDireito': 200, 'largura': 45, 'Altura': 15, 'As': 4.7}, # Original
        {'PeDireito': 200, 'largura': 30, 'Altura': 15, 'As': 0},   # Reduzindo largura
        {'PeDireito': 200, 'largura': 20, 'Altura': 15, 'As': 0},   # Muito fino (deve falhar?)
        {'PeDireito': 200, 'largura': 15, 'Altura': 15, 'As': 0},   # Quadrado pequeno (deve falhar)
    ]
    
    print(f"{'Dimensao':<15} {'Status':<12} {'Prob':<8} {'As_Pred':<10}")
    print("-" * 50)
    
    for var in variations:
        pillar = {**base_loads, **var}
        result = predictor.predict_single(pillar)
        
        dim_str = f"{var['largura']}x{var['Altura']}cm"
        status = result['status']
        prob = f"{result['feasibility_prob']:.0%}"
        as_pred = f"{result.get('As_predicted', 0):.2f}"
        
        print(f"{dim_str:<15} {status:<12} {prob:<8} {as_pred:<10}")

if __name__ == "__main__":
    example_single_prediction()
    example_batch_test()
    example_compare_variations()