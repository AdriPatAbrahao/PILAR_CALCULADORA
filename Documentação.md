# 🏗️ PILAR CALCULADORA IA - Otimização Estrutural Inteligente

Este projeto implementa um sistema de Inteligência Artificial para o pré-dimensionamento e otimização de custo de pilares de concreto armado. O sistema utiliza uma arquitetura de dois estágios ("Fiscal" e "Engenheiro") combinada com features baseadas na física do concreto (Ábacos de Interação) para garantir precisão e segurança.

---

## 📋 Índice
1. [Visão Geral da Arquitetura](#-visão-geral-da-arquitetura)
2. [Estrutura do Projeto](#-estrutura-do-projeto)
3. [Instalação e Configuração](#-instalação-e-configuração)
4. [Como Usar](#-como-usar)
    - [Treinamento da IA](#1-treinamento-mainpy)
    - [Inferência (Teste)](#2-inferência-inference_demopy)
    - [Otimização de Custo](#3-otimização-run_optimizationpy)
5. [Detalhes dos Módulos](#-detalhes-dos-módulos)
6. [Metodologia de Engenharia](#-metodologia-de-engenharia)

---

## 🧠 Visão Geral da Arquitetura

O sistema não utiliza uma única rede neural, mas sim um **Pipeline de 2 Estágios** para imitar o processo de decisão de um engenheiro:

1.  **Estágio 1: O "Fiscal" (Classificador - LightGBM Binary)**
    * **Função:** Analisa se a geometria e as cargas propostas são *fisicamente viáveis*.
    * **Saída:** Probabilidade de Sucesso (0 a 100%) e Status (`Feasible`/`Infeasible`).
    * **Objetivo:** Impedir que o sistema dimensione aço para pilares que colapsariam por esmagamento ou flambagem excessiva.

2.  **Estágio 2: O "Engenheiro" (Regressor - LightGBM Regression L1)**
    * **Função:** Calcula a área de aço necessária ($A_s$) para os pilares aprovados pelo Fiscal.
    * **Treinamento:** Focado em minimizar o Erro Médio Absoluto (MAE) para precisão centimétrica.
    * **Saída:** Área de Aço em cm².

3.  **Otimizador (Grid Search + Physics Override)**
    * **Função:** Varre milhares de combinações de largura/altura para encontrar a seção que minimiza o custo total (Concreto + Aço).
    * **Segurança:** Possui uma "Rede de Segurança" (`Physics Override`) que usa a taxa de carga normalizada ($\nu$) para validar pilares robustos que a IA possa ter rejeitado indevidamente.

---

## 📂 Estrutura do Projeto

```text
PILAR_CALCULADORA/
├── data/
│   └── dados_pilares.csv       # Dataset de treinamento (CSV com ; e decimais .)
├── models/
│   ├── modelo_classificador.pkl # Modelo treinado do Estágio 1
│   └── modelo_regressor.pkl     # Modelo treinado do Estágio 2
├── logs/                       # Logs de execução (treinamento e erros)
├── src/
│   ├── config.py               # Configurações globais (Caminhos, Parâmetros, Features)
│   ├── data_loader.py          # Carregamento e limpeza de dados (Trata erros e flags)
│   ├── feature_engineering.py  # Criação de variáveis físicas (nu, mu, lambda, p-delta)
│   ├── model_trainer.py        # Funções de treino, avaliação e split de dados
│   ├── predictor.py            # Classe de inferência (Carrega modelos e prevê)
│   ├── optimizer.py            # Motor de otimização de custo e geometria
│   └── utils.py                # Utilitários (Logs, prints)
├── main.py                     # Script principal para TREINAR a IA
├── inference_demo.py           # Script para TESTAR a IA (Inferência)
├── run_optimization.py         # Script para OTIMIZAR um pilar específico
└── requirements.txt            # Dependências do Python


⚙️ Instalação e ConfiguraçãoPré-requisitos: Python 3.8+ instalado.Instalar dependências:Bashpip install -r requirements.txt
(Principais libs: pandas, numpy, scikit-learn, lightgbm, joblib)Configuração:Edite o arquivo src/config.py para ajustar parâmetros como caminhos de arquivo ou hiperparâmetros dos modelos (num_leaves, learning_rate).🚀 Como Usar1. Treinamento (main.py)Executa o pipeline completo: carrega dados, cria features, treina os dois modelos e salva em /models.Bashpython main.py
Saída esperada: Relatórios de acurácia (AUC, RMSE, MAE) e importância das features no terminal.2. Inferência (inference_demo.py)Usa os modelos treinados para prever o aço de pilares de teste (reais e hipotéticos). Útil para validar se a IA está "pensando" certo.Bashpython inference_demo.py
3. Otimização (run_optimization.py)A ferramenta final. Você insere as cargas e parâmetros fixos no script, e ele busca a melhor largura.Como configurar:Abra run_optimization.py e edite o dicionário LOAD_VECTOR e FIXED_PARAMS com os dados da sua obra.Bashpython run_optimization.py
Saída esperada: Tabela com as melhores opções de seção, custo de concreto, custo de aço e custo total.🛠️ Detalhes dos Módulossrc/feature_engineering.pyEste é o cérebro físico do projeto. Ele converte dados brutos (N, M, b, h) em variáveis de engenharia estrutural:nu (Normal Reduzida): Taxa de utilização da compressão do concreto.mu_x / mu_y (Momentos Reduzidos): Taxa de utilização da flexão.lambda (Esbeltez): Indicador de risco de flambagem.index_2nd_order: Indicador composto ($\nu \cdot \lambda^2$) que detecta risco crítico de efeitos de 2ª ordem (P-Delta).aspect_ratio: Formato da seção (Retangularidade).src/optimizer.pyImplementa uma busca em grade inteligente:Gera candidatos variando a largura (ex: 15 a 80 cm).Calcula viabilidade e aço para todos via IA (predictor.py).Calcula custos reais:Aço: Peso (kg) calculada via densidade linear ($A_s \cdot L \cdot 0.785$).Concreto: Volume ($m^3$).Physics Override: Se a IA reprovar um pilar com carga muito baixa ($\nu < 0.4$), o otimizador força a aprovação e calcula armadura mínima, corrigindo possíveis vieses conservadores do modelo.📊 Metodologia de EngenhariaTratamento de Dados "Sujos"O dataset original contém pilares que falharam no software de origem (marcados com $A_s=0$ ou valores absurdos).No Treino: O data_loader.py identifica esses casos e cria a flag is_feasible.O Classificador aprende a identificar o padrão desses erros.O Regressor é treinado apenas com os dados viáveis, garantindo que ele não aprenda a prever "zero aço" ou "aço infinito".Consideração de CustosA função objetivo de otimização é:$$ Custo_{Total} = (V_{conc} \times Preço_{m^3}) + (Peso_{aço} \times Preço_{kg}) $$Onde o peso do aço é derivado diretamente da previsão da IA, garantindo que a solução ótima balanceie a economia de concreto (pilares finos) com a economia de aço (pilares robustos).