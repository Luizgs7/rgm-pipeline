# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<role>
Atue como um Master Arquiteto de IA e Especialista no framework Agnos AI. Sua missão é projetar e codificar um Produto de Dados ponta a ponta para RGM (Revenue Growth Management) do absoluto zero, orquestrando um sistema Multi-Agentes.
</role>

<context>
O objetivo final deste produto de dados é gerar uma grade de campanhas promocionais otimizada para o varejo/indústria. O sistema deve ser capaz de estimar baselines, prever demanda sob diferentes cenários de desconto, otimizar a margem e explicar suas decisões. 
</context>

<agentic_framework>
Todo o desenvolvimento deve ser estruturado sob o paradigma de agentes orquestrados pelo Agnos AI. Assuma o controle dos seguintes agentes e divida o código e as tarefas entre eles:

1. [Agente Engenheiro de Dados]: Responsável por criar o banco de dados/mock, garantir tabelas performáticas, implementar processos de Data Quality (DQ) e aplicar regras de segurança de acessos.
2. [Agente Cientista de Dados]: Responsável pela criação de todos os modelos estatísticos (baseline contrafactual), modelos de machine learning (previsão e simulação de demanda) e modelos de otimização matemática (geração da grade prescritiva).
3. [Agente Engenheiro de Machine Learning]: Responsável por "empacotar" a solução, garantindo performance, escalabilidade, monitoramentos de Data/Model Drift e segurança dos modelos e APIs.
</agentic_framework>

<tech_stack>
- Linguagem: Python 3.10+
- Paradigma: Programação Orientada a Objetos (OOP), código modular e limpo (Clean Code).
- API/Deploy: FastAPI.
- Sugestão de Bibliotecas: Pandas/Polars (manipulação), DoWhy/EconML (Inferência Causal), Scikit-Learn/Prophet/XGBoost (Previsão), PuLP ou SciPy (Otimização Matemática), SHAP (Explicabilidade), Evidently/Alibi (Monitoramento de Drift).
</tech_stack>

<instructions>
Por favor, desenvolva este projeto passo a passo, fornecendo a estrutura de diretórios e os scripts em Python. Indique claramente qual [Agente] está atuando em cada módulo:

1. <module_data_and_quality> [Agente Engenheiro de Dados]
- Crie um script para gerar um banco de dados mockado realista com transações históricas, campanhas cadastradas e métricas de uplift (margem, receita, volume).
- Implemente rotinas de Data Quality (validação de schemas, nulos, anomalias) e simule um controle de segurança de acesso aos dados.
</module_data_and_quality>

2. <module_modeling_and_simulation> [Agente Cientista de Dados]
- Baseline Causal: Modelo contrafactual para estimar o que teria acontecido com as vendas/margem nos meses de promoção se a campanha não tivesse ocorrido.
- Demand Forecasting: Modelo preditivo projetando o incremento esperado para cada produto/mês simulando descontos de 10%, 20%, 30% e 40%.
</module_modeling_and_simulation>

3. <module_optimization_and_xai> [Agente Cientista de Dados]
- Otimizador Matemático: Maximize a margem global com base nas simulações, gerando a grade de campanhas sugerida. Inclua restrições lógicas (verba, sobreposição).
- Explicabilidade: Gere um módulo indicando por que a recomendação foi feita e um "Score de Confiança" de acurácia.
</module_optimization_and_xai>

4. <module_mlops_and_scalability> [Agente Engenheiro de Machine Learning]
- Arquitetura de Produção: Crie a interface FastAPI integrando os pipelines dos agentes anteriores.
- Monitoramento e Segurança: Implemente processos para monitoramento contínuo de Drift (dados e modelo), garantindo a escalabilidade, performance da API e segurança da inferência.
</module_mlops_and_scalability>
</instructions>

<output_format>
- Inicie apresentando a arquitetura multi-agentes proposta no Agnos AI e a estrutura de pastas do projeto (tree).
- Em seguida, forneça os blocos de código separados por arquivo/módulo e por Agente, garantindo type hints e docstrings.
- Como o código será extenso, atue primeiro como o [Agente Engenheiro de Dados] e entregue o Módulo 1. Ao final, pare e me pergunte se pode invocar o [Agente Cientista de Dados] para prosseguir com os módulos 2 e 3.
</output_format>

## Repository

- Remote: https://github.com/Luizgs7/rgm-pipeline.git
- Branch: main

---

## Architecture

Multi-agent system for RGM (Revenue Growth Management) — generates an optimized promotional campaign grid for retail/industry.

```
rgm-pipeline/
├── agents/
│   ├── data_engineer/        # Módulo 1: mock, DQ, ACL
│   │   ├── mock_generator.py   # MockDataGenerator — gera DB SQLite com produtos, lojas, campanhas, transações, uplift
│   │   ├── data_quality.py     # DataQualityRunner — schema, nulos, IQR, regras de negócio
│   │   └── access_control.py   # AccessControlService — RBAC com mascaramento e auditoria
│   ├── data_scientist/       # Módulos 2+3: causal baseline, demand forecast, otimização, XAI
│   │   ├── causal_baseline.py    # CausalBaselineEstimator — DiD + regressão contrafactual
│   │   ├── demand_forecasting.py # DemandForecaster — XGBoost + simulação de cenários de desconto
│   │   ├── optimizer.py          # CampaignOptimizer — ILP (PuLP) maximiza margem com restrições
│   │   └── explainability.py     # CampaignExplainer — SHAP + Score de Confiança composto
│   └── ml_engineer/          # Módulo 4: FastAPI, drift monitoring, segurança
│       ├── api.py                # FastAPI app — 6 endpoints REST
│       ├── schemas.py            # Pydantic request/response models
│       ├── security.py           # API Key auth + rate limiting + request tracing
│       └── drift_monitor.py      # DriftMonitor — Evidently + fallback KS/chi²
├── config/settings.py        # Caminhos, parâmetros mock, roles ACL, limiares DQ
├── data/
│   ├── raw/rgm_database.db       # SQLite gerado pelo MockDataGenerator
│   └── processed/                # Parquets: causal_baseline, demand_simulation, campaign_grid, campaign_explanations
├── run_module1.py            # Entry point do Módulo 1
├── run_module2_3.py          # Entry point dos Módulos 2 e 3
├── run_module4.py            # Entry point do Módulo 4 (servidor API)
└── requirements.txt
```

## Commands

```bash
# Instalar dependências
pip install -r requirements.txt

# Módulo 1 — gera DB SQLite, valida DQ, demonstra ACL (executar primeiro)
python run_module1.py

# Módulos 2+3 — baseline causal, previsão de demanda, otimização, XAI
python run_module2_3.py

# Módulo 4 — inicia servidor FastAPI (dev)
python run_module4.py --reload

# Módulo 4 — produção (multi-worker)
python run_module4.py --host 0.0.0.0 --port 8000 --workers 4

# Dashboard Streamlit
streamlit run dashboard.py          # → http://localhost:8501

# Testes (69 testes, ~6s)
pip install pyarrow pytest pytest-asyncio httpx
python -m pytest

# Docker (pipeline setup + API)
docker-compose up
```

## API Endpoints

| Método | Rota | Auth | Descrição |
|--------|------|------|-----------|
| GET | `/health` | — | Status dos componentes |
| POST | `/pipeline/run` | admin | Executa pipeline completo |
| GET | `/campaigns/grid` | analyst+ | Grade de campanhas otimizada |
| POST | `/predict/demand` | analyst+ | Predição de demanda por cenário |
| POST | `/monitor/drift` | admin | Verificação de drift |
| GET | `/monitor/drift/latest` | admin | Último relatório de drift |

Docs interativas: `http://localhost:8000/docs`

## Segurança

- Autenticação via header `X-API-Key` (gerado no startup, impresso no log)
- Rate limit: 10 req/s por IP, burst até 60
- Roles: `admin` (acesso total) e `analyst` (leitura e predição)

## Pipeline de Dados (Módulos 2+3)

1. **CausalBaselineEstimator** — constrói painel produto-loja-mês, aplica estimador DiD e treina regressão linear com FE para gerar contrafactuais granulares → `causal_baseline.parquet`
2. **DemandForecaster** — feature engineering (lags, MA, calendário, discount_pct), treino XGBoost com TimeSeriesSplit, simula 4 níveis de desconto → `demand_simulation.parquet`
3. **CampaignOptimizer** — ILP binário (PuLP/CBC): maximiza `Σ net_margin * x_i` com restrição de verba total e sobreposição por produto → `campaign_grid.parquet`
4. **CampaignExplainer** — SHAP TreeExplainer para importância de features, Score de Confiança composto (accuracy + stability + coverage), explicações textuais → `campaign_explanations.parquet`

## Data Model

**SQLite tables** (geradas por `MockDataGenerator`):
- `products` — SKU, categoria, custo, preço base
- `stores` — loja, região, porte
- `campaigns` — campanha, produto, loja, desconto%, verba, período
- `transactions` — transações diárias com efeito promocional embutido (sazonalidade + uplift)
- `uplift_metrics` — incrementais de volume/receita/margem e ROI por campanha

## Access Control Roles

| Role | Tabelas | Operações | Mascaramento |
|------|---------|-----------|--------------|
| admin | todas | read/write/delete | não |
| analyst | todas | read | não |
| viewer | campaigns, uplift_metrics | read | sim (budget, margin, ROI) |
