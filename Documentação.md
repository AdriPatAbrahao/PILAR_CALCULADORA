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