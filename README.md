# RGM Pipeline — Sistema Multi-Agentes para Revenue Growth Management

> **Produto de Dados ponta a ponta** para geração de grades de campanhas promocionais otimizadas,
> orquestrado pelo framework **Agnos AI** com três agentes especializados.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura da Solução](#arquitetura-da-solução)
- [Agentes e Módulos](#agentes-e-módulos)
  - [Agente 1 — Engenheiro de Dados](#agente-1--engenheiro-de-dados)
  - [Agente 2 — Cientista de Dados](#agente-2--cientista-de-dados)
  - [Agente 3 — Engenheiro de ML](#agente-3--engenheiro-de-ml)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Instalação e Configuração](#instalação-e-configuração)
- [Como Executar](#como-executar)
- [API — Endpoints](#api--endpoints)
- [Experimento: XGBoost vs Foundation Model (LLM)](#experimento-xgboost-vs-foundation-model-llm)
- [Stack Tecnológico](#stack-tecnológico)

---

## Visão Geral

O **RGM Pipeline** automatiza o ciclo completo de Revenue Growth Management para varejo/indústria:

```
Dados Históricos → Baseline Causal → Previsão de Demanda → Otimização → Grade de Campanhas
```

O sistema é capaz de:

| Capacidade | Detalhe |
|-----------|---------|
| **Estimar baseline causal** | O que teria vendido sem a promoção (contrafactual) |
| **Prever demanda** | Incremento esperado para descontos de 10%, 20%, 30% e 40% |
| **Otimizar campanhas** | Maximiza margem global com restrições de verba e sobreposição |
| **Explicar decisões** | SHAP values + Score de Confiança + linguagem natural (LLM) |
| **Monitorar em produção** | Data Drift e Model Drift contínuos via Evidently |

---

## Arquitetura da Solução

O arquivo de arquitetura interativo está em [`docs/rgm_pipeline_architecture.drawio`](docs/rgm_pipeline_architecture.drawio).
Abra em [app.diagrams.net](https://app.diagrams.net) → *File → Open from device*.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Orquestrador: Agnos AI Framework — Python 3.10+                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  AGENTE 1 — Engenheiro de Dados                                              │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────────┐  │
│  │ MockDataGenerator│  │ DataQualityRunner│  │  AccessControlService     │  │
│  │                  │  │                  │  │                           │  │
│  │ 20 produtos      │  │ Schema · Nulos   │  │  RBAC: admin · analyst    │  │
│  │ 10 lojas         │  │ IQR · Negócio    │  │  viewer · mascaramento    │  │
│  │ 80 campanhas     │  │                  │  │  · auditoria              │  │
│  └────────┬─────────┘  └────────┬─────────┘  └───────────────────────────┘  │
└───────────┼─────────────────────┼────────────────────────────────────────────┘
            │                     │ valida
            ▼                     ▼
   ┌─────────────────────────────────────────────────────┐
   │  SQLite Database  (data/raw/rgm_database.db)        │
   │  products · stores · campaigns · transactions       │
   │  uplift_metrics                                     │
   └──────────────────────────┬──────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────────────┐
│  AGENTE 2 — Cientista de Dados                                               │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ CausalBaseline   │  │ DemandForecaster │  │ CampaignOptimizer        │   │
│  │ Estimator        │  │                  │  │                          │   │
│  │ DiD + Regressão  │  │ XGBoost          │  │ ILP Binário (PuLP/CBC)   │   │
│  │ Linear com FE    │  │ TimeSeriesSplit  │  │ Max margem + restrições  │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────┐                                    │
│  │ CampaignExplainer                    │                                    │
│  │ SHAP TreeExplainer · Score Confiança │                                    │
│  └──────────────────────────────────────┘                                    │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┬──────────────────────┐
         ▼                    ▼                    ▼                      ▼
  causal_baseline      demand_simulation    campaign_grid    campaign_explanations
    .parquet              .parquet            .parquet             .parquet
         └────────────────────┼────────────────────┴──────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────────────┐
│  AGENTE 3 — Engenheiro de ML                                                 │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ FastAPI App      │  │ Security         │  │ DriftMonitor             │   │
│  │ 6 endpoints REST │  │ API Key · Rate   │  │ Evidently + KS/chi²      │   │
│  │ :8000            │  │ Limit · Roles    │  │ Data + Model Drift       │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
            │                     │                        │
            ▼                     ▼                        ▼
     Swagger UI /docs    Streamlit Dashboard        REST Clients
```

---

## Agentes e Módulos

### Agente 1 — Engenheiro de Dados

**Localização:** `src/rgm_pipeline/agents/data_engineer/`

**Responsabilidade:** Criar e validar a base de dados, aplicar regras de qualidade e controlar acesso.

| Arquivo | Classe | Função |
|---------|--------|--------|
| `mock_generator.py` | `MockDataGenerator` | Gera banco SQLite com dados históricos realistas usando geração vetorizada (`pd.MultiIndex.from_product`). Inclui efeito promocional de uplift nas transações. |
| `data_quality.py` | `DataQualityRunner` | Valida schemas, detecta nulos (limite 5%), identifica anomalias via IQR (×3) e aplica regras de negócio RGM (margem mínima, desconto máximo). |
| `access_control.py` | `AccessControlService` | RBAC com três perfis: `admin` (leitura/escrita/delete), `analyst` (leitura), `viewer` (leitura com mascaramento de campos sensíveis como budget e ROI). Gera log de auditoria. |

**Tabelas geradas no SQLite:**

| Tabela | Descrição | Volume |
|--------|-----------|--------|
| `products` | SKU, categoria, custo, preço base | 20 produtos |
| `stores` | Loja, região, porte | 10 lojas |
| `campaigns` | Desconto%, verba, período | 80 campanhas |
| `transactions` | Transações diárias com efeito promocional | ~146.000 registros |
| `uplift_metrics` | Incrementais de volume/receita/margem e ROI | 1 por campanha |

---

### Agente 2 — Cientista de Dados

**Localização:** `src/rgm_pipeline/agents/data_scientist/`

**Responsabilidade:** Modelagem estatística, previsão de demanda, otimização matemática e explicabilidade.

#### `CausalBaselineEstimator` — `causal_baseline.py`

Estima o contrafactual: *"o que teria acontecido se a campanha não tivesse ocorrido?"*

- Constrói painel produto-loja-mês
- Aplica estimador **Difference-in-Differences (DiD)**
- Treina regressão linear com **Fixed Effects** para gerar contrafactuais granulares
- Saída: `data/processed/causal_baseline.parquet`

#### `DemandForecaster` — `demand_forecasting.py`

Prevê incremento de volume para cada cenário de desconto.

- Algoritmo: **XGBoost** (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`)
- Validação: `TimeSeriesSplit(n_splits=3)` — preserva ordem cronológica
- **Features engineered:**

| Grupo | Features |
|-------|----------|
| Identidade | `product_enc`, `store_enc` (LabelEncoder) |
| Calendário | `month`, `dayofweek`, `quarter`, `is_weekend`, `day_of_year` |
| Lags (dias) | `lag_7d`, `lag_30d`, `lag_90d` |
| Médias móveis | `ma_7d`, `ma_30d` |
| Promoção | `discount_pct` |

- Simula 4 cenários: 10%, 20%, 30% e 40% de desconto
- Saída: `data/processed/demand_simulation.parquet`

#### `CampaignOptimizer` — `optimizer.py`

Gera a grade de campanhas prescritiva maximizando margem global.

- **Programação Linear Inteira (ILP)** via PuLP/CBC
- Variável de decisão: binária `x_i ∈ {0, 1}` por campanha
- Função objetivo: `max Σ net_margin_i × x_i`
- Restrições: verba total disponível, sobreposição de campanhas no mesmo produto
- Saída: `data/processed/campaign_grid.parquet`

#### `CampaignExplainer` — `explainability.py`

Explica por que cada campanha foi recomendada.

- **SHAP TreeExplainer** para importância de features
- **Score de Confiança composto:** `accuracy × stability × coverage`
- Gera explicações textuais para cada recomendação
- Saída: `data/processed/campaign_explanations.parquet`

---

### Agente 3 — Engenheiro de ML

**Localização:** `src/rgm_pipeline/agents/ml_engineer/`

**Responsabilidade:** Empacotamento em API de produção, segurança e monitoramento contínuo.

#### `api.py` — FastAPI Application

| Método | Rota | Autenticação | Descrição |
|--------|------|:---:|-----------|
| `GET` | `/health` | — | Status de todos os componentes do sistema |
| `POST` | `/pipeline/run` | `admin` | Executa o pipeline completo (Módulos 1→2→3) |
| `GET` | `/campaigns/grid` | `analyst+` | Retorna a grade de campanhas otimizada |
| `POST` | `/predict/demand` | `analyst+` | Predição de demanda para um cenário de desconto |
| `POST` | `/monitor/drift` | `admin` | Dispara verificação de drift nos dados e modelo |
| `GET` | `/monitor/drift/latest` | `admin` | Retorna o último relatório de drift |

#### `security.py` — Camada de Segurança

- Autenticação via header `X-API-Key` (gerado no startup, impresso no log)
- Rate limiting: **10 req/s por IP**, burst até 60
- Roles: `admin` (acesso total) e `analyst` (leitura e predição)
- Request tracing com UUID por requisição

#### `drift_monitor.py` — Monitoramento de Drift

- **Evidently** como motor principal de detecção
- Fallback estatístico: **KS-test** (features contínuas) e **chi²** (categorias)
- Monitora: Data Drift (distribuição de features) e Model Drift (degradação de métricas)
- Relatórios estruturados em JSON

#### `schemas.py` — Contratos de API

Modelos Pydantic para validação de tipos em todos os endpoints:

| Schema | Tipo | Uso |
|--------|------|-----|
| `PipelineRunRequest` / `PipelineRunResponse` | Request / Response | `POST /pipeline/run` |
| `DemandPredictRequest` / `DemandPredictResponse` | Request / Response | `POST /predict/demand` |
| `DriftCheckRequest` / `DriftReport` | Request / Response | `POST /monitor/drift` |
| `CampaignGridResponse` / `CampaignItem` | Response | `GET /campaigns/grid` |
| `HealthResponse` | Response | `GET /health` |

---

## Estrutura de Diretórios

```
rgm-pipeline/
├── src/rgm_pipeline/
│   ├── agents/
│   │   ├── data_engineer/
│   │   │   ├── mock_generator.py      # MockDataGenerator
│   │   │   ├── data_quality.py        # DataQualityRunner
│   │   │   └── access_control.py      # AccessControlService
│   │   ├── data_scientist/
│   │   │   ├── causal_baseline.py     # CausalBaselineEstimator
│   │   │   ├── demand_forecasting.py  # DemandForecaster (XGBoost)
│   │   │   ├── optimizer.py           # CampaignOptimizer (ILP)
│   │   │   └── explainability.py      # CampaignExplainer (SHAP)
│   │   └── ml_engineer/
│   │       ├── api.py                 # FastAPI — 6 endpoints
│   │       ├── security.py            # API Key + Rate Limit
│   │       ├── drift_monitor.py       # Evidently + KS/chi²
│   │       └── schemas.py             # Pydantic models
│   └── config/settings.py             # Paths, parâmetros globais
├── scripts/
│   ├── run_module1.py                 # Entry point: Engenheiro de Dados
│   ├── run_module2_3.py               # Entry point: Cientista de Dados
│   └── run_module4.py                 # Entry point: API FastAPI
├── notebooks/
│   └── experiment_xgboost_vs_llm_forecaster.ipynb  # Experimento comparativo
├── app/
│   └── dashboard.py                   # Streamlit Dashboard
├── docs/
│   └── rgm_pipeline_architecture.drawio  # Diagrama de arquitetura
├── data/
│   ├── raw/rgm_database.db            # SQLite gerado pelo Módulo 1
│   └── processed/                     # Parquets dos Módulos 2+3
├── tests/
│   ├── unit/                          # Testes unitários (69 testes)
│   └── integration/                   # Testes de integração da API
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Instalação e Configuração

### Pré-requisitos

- Python 3.10+
- `pip` ou `conda`

### 1. Clonar o repositório

```bash
git clone https://github.com/Luizgs7/rgm-pipeline.git
cd rgm-pipeline
```

### 2. Criar ambiente virtual e instalar dependências

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install -e .                   # instala o pacote src/ em modo editável
```

### 3. Configurar variáveis de ambiente (opcional)

Crie um arquivo `.env` na raiz do projeto:

```dotenv
# Necessário apenas para o módulo experimental (LLM real)
ANTHROPIC_API_KEY=sk-ant-...
```

> Sem a chave, o notebook do experimento roda automaticamente em **modo mock** (sem chamadas à API).

---

## Como Executar

Os módulos devem ser executados **em ordem**: Módulo 1 → Módulos 2+3 → Módulo 4.

### Módulo 1 — Engenheiro de Dados

Gera o banco SQLite, valida a qualidade dos dados e demonstra o controle de acesso.

```bash
python scripts/run_module1.py
```

**Saídas:** `data/raw/rgm_database.db` com 5 tabelas e ~73.000 transações.

---

### Módulos 2+3 — Cientista de Dados

Executa baseline causal, previsão de demanda, otimização e explicabilidade.

```bash
python scripts/run_module2_3.py
```

**Saídas** em `data/processed/`:
- `causal_baseline.parquet` — contrafactuais por produto-loja-mês
- `demand_simulation.parquet` — simulações para 4 níveis de desconto
- `campaign_grid.parquet` — grade de campanhas otimizada
- `campaign_explanations.parquet` — explicações e scores de confiança

---

### Módulo 4 — API FastAPI

#### Desenvolvimento (com reload automático)

```bash
python scripts/run_module4.py --reload
```

#### Produção (multi-worker)

```bash
python scripts/run_module4.py --host 0.0.0.0 --port 8000 --workers 4
```

**Acesso:** `http://localhost:8000/docs` (Swagger UI interativo)

---

### Dashboard Streamlit

```bash
streamlit run app/dashboard.py
```

**Acesso:** `http://localhost:8501`

---

### Docker (pipeline completo)

```bash
docker-compose -f docker/docker-compose.yml up
```

---

### Testes

```bash
pip install pytest pytest-asyncio httpx
python -m pytest                   # 69 testes, ~6s
python -m pytest -v --tb=short    # verbose
```

---

### Notebook — Experimento Comparativo

```bash
pip install jupyter
jupyter lab notebooks/experiment_xgboost_vs_llm_forecaster.ipynb
```

Ou executar diretamente (gera outputs inline):

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/experiment_xgboost_vs_llm_forecaster.ipynb
```

---

## API — Endpoints

### Autenticação

Todas as rotas protegidas exigem o header:

```http
X-API-Key: <chave gerada no startup>
```

A chave é impressa no log ao iniciar o servidor:

```
INFO | API Key gerada: rgm-xxxxxxxxxxxxxxxxxxxx
```

### Exemplos de uso

#### Verificar saúde do sistema

```bash
curl http://localhost:8000/health
```

#### Executar pipeline completo

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "X-API-Key: <sua-chave>" \
  -H "Content-Type: application/json" \
  -d '{"force_regenerate": false}'
```

#### Predição de demanda

```bash
curl -X POST http://localhost:8000/predict/demand \
  -H "X-API-Key: <sua-chave>" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "SKU-0001",
    "store_id": "LJ-001",
    "discount_pct": 0.20,
    "year_month": "2024-06"
  }'
```

#### Obter grade de campanhas

```bash
curl http://localhost:8000/campaigns/grid \
  -H "X-API-Key: <sua-chave>"
```

---

## Experimento: XGBoost vs Foundation Model (LLM)

**Notebook:** [`notebooks/experiment_xgboost_vs_llm_forecaster.ipynb`](notebooks/experiment_xgboost_vs_llm_forecaster.ipynb)

Este experimento compara o modelo de previsão de demanda atual (XGBoost) contra um modelo fundacional baseado em LLM (Claude Sonnet 4.6) em termos de acurácia, bias e explicabilidade.

### Configuração do Experimento

| Parâmetro | Valor |
|-----------|-------|
| Período de treino | Jan/2023 – Dez/2023 (2.400 registros mensais) |
| Período de teste | Jan/2024 – Dez/2024 (2.400 registros mensais) |
| Amostra para LLM | 600 registros (50 pares produto-loja × 12 meses) |
| Unidade de análise | Produto × Loja × Mês |
| Modelo LLM | Claude Sonnet 4.6 (zero-shot prompting) |

### Resultados — Métricas Globais

| Métrica | XGBoost | LLM (Claude) | Vencedor |
|---------|:-------:|:------------:|:--------:|
| **WMAPE** ↓ | **5,85%** | 11,23% | XGBoost |
| **Bias** ↓\|·\| | **-1,09%** | +1,37% | XGBoost |
| **MAE** ↓ | **203,9** | 391,2 | XGBoost |
| **RMSE** ↓ | **494,8** | 552,5 | XGBoost |
| **MAPE** ↓ | **7,57%** | 11,35% | XGBoost |
| **R²** ↑ | **0,9363** | 0,9206 | XGBoost |
| **Over-forecast %** | 48,3% | 54,0% | XGBoost |
| **Pearson r** ↑ | **0,9680** | 0,9616 | XGBoost |

> ↓ = menor é melhor · ↑ = maior é melhor

### Resultados — Validação Cruzada do XGBoost

| Fold | MAPE |
|------|------|
| Fold 1 | 20,51% |
| Fold 2 | 15,85% |
| Fold 3 | 17,39% |
| **Média ± DP** | **17,91% ± 1,94%** |

### Calibração de Confiança do LLM

O LLM auto-reporta um score de confiança (0–1) por predição. A análise mostra correlação negativa entre confiança e erro:

| Faixa de Confiança | n | WMAPE LLM | WMAPE XGBoost |
|:------------------:|:-:|:---------:|:-------------:|
| 65–75% | 59 | 12,2% | 6,6% |
| 75–85% | 374 | 11,2% | 5,4% |
| >85% | 167 | **10,9%** | 6,5% |

> O WMAPE do LLM decresce com o aumento da confiança, indicando **calibração razoável**.

### Scorecard Final — 10 Dimensões

| Dimensão | XGBoost | LLM (Claude) | Vencedor |
|----------|:-------:|:------------:|:--------:|
| WMAPE (acurácia geral) | 5,85% | 11,23% | ✅ **XGBoost** |
| Bias (neutralidade) | -1,09% | +1,37% | ✅ **XGBoost** |
| R² (poder explicativo) | 0,9363 | 0,9206 | ✅ **XGBoost** |
| RMSE (erros grandes) | 494,8 | 552,5 | ✅ **XGBoost** |
| MAE | 203,9 | 391,2 | ✅ **XGBoost** |
| Calibração de confiança | N/A | Score 0–1 | ✅ **LLM** |
| Explicabilidade humana | SHAP técnico | NL narrativo | ✅ **LLM** |
| Custo computacional | ~ms/pred | ~1–2s/pred | ✅ **XGBoost** |
| Escalabilidade | >>1.000/s | ~0,5–1/s | ✅ **XGBoost** |
| Cold-start (sem histórico) | Ruim | Bom | ✅ **LLM** |

**XGBoost: 5 dimensões · LLM: 5 dimensões**

### Análise de Explicabilidade

#### XGBoost — Importância SHAP (top features)

As features mais impactantes para a predição de volume foram, em ordem:
1. `lag_1m` — volume do mês anterior (maior preditor individual)
2. `ma_3m` / `ma_6m` — médias móveis recentes
3. `discount_pct` — nível de desconto aplicado
4. `lag_12m` — sazonalidade anual
5. `product_enc` / `store_enc` — identidade do par produto-loja

> O SHAP dependence plot de `discount_pct` evidencia elasticidade positiva não-linear: o efeito do desconto é amplificado quando a média móvel histórica é alta.

#### LLM — Chain-of-Thought

O LLM decompõe cada predição em fatores atribuíveis:

```
Base histórica  →  +Sazonalidade  →  +Efeito desconto  →  +Tendência  →  Volume Predito
```

Exemplo de reasoning (melhor predição — erro 0,0%):
> *"Produto apresenta tendência crescente no histórico recente. Elasticidade-preço estimada
> em 0,00 para o nível de desconto 0%. Sazonalidade do período impacta +3,2%."*

### Conclusões e Recomendação

**O XGBoost supera o LLM em todas as métricas quantitativas** quando há histórico disponível (≥6 meses). O LLM opera com priors gerais sobre elasticidade, sem aprender os padrões idiossincráticos de cada par produto-loja.

**Recomendação: Arquitetura Híbrida**

```
Requisição de previsão
        │
        ▼
┌───────────────────┐
│  Roteador         │  ← verifica histórico disponível
│  Histórico ≥ 6m?  │
└───────┬───────────┘
        │
   Sim  │  Não (cold-start)
   ─────┼───────────────────────────────────────────────┐
        ▼                                               ▼
┌──────────────┐                           ┌─────────────────────┐
│   XGBoost    │                           │  LLM (Claude)       │
│  (predição)  │                           │  (pred + reasoning) │
└──────┬───────┘                           └──────────┬──────────┘
       │                                              │
       ▼                                              ▼
┌──────────────┐                           ┌─────────────────────┐
│  SHAP values │                           │  Análise profunda   │
│  (explain)   │                           │  (chain-of-thought) │
└──────┬───────┘                           └──────────┬──────────┘
       └─────────────────┬─────────────────────────────┘
                         ▼
             ┌───────────────────────┐
             │  Resposta unificada   │
             │  pred + confiança     │
             │  + reasoning narrativo│
             └───────────────────────┘
```

| Cenário | Modelo Recomendado |
|---------|-------------------|
| Produto com histórico ≥ 6 meses | **XGBoost** |
| Cold-start (produto/loja nova) | **LLM** |
| Escala (>1.000 predições/s) | **XGBoost** |
| Explicação para gestores não-técnicos | **LLM** |
| Eventos externos (greve, promo concorrente) | **LLM** |
| Auditoria regulatória (feature attribution) | **SHAP + XGBoost** |

---

## Stack Tecnológico

| Categoria | Bibliotecas |
|-----------|-------------|
| **Core** | Python 3.10+, Pandas, NumPy, Polars |
| **Orquestração** | Agnos AI |
| **ML / Forecasting** | XGBoost, Scikit-Learn, Prophet |
| **Inferência Causal** | DoWhy, EconML |
| **Otimização** | PuLP (CBC solver), SciPy |
| **Explicabilidade** | SHAP |
| **LLM** | Anthropic Claude (claude-sonnet-4-6) |
| **Monitoramento** | Evidently, KS-test, chi² |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Banco de Dados** | SQLite (mock), Parquet (processado) |
| **Visualização** | Streamlit, Matplotlib, Seaborn |
| **Testes** | pytest, pytest-asyncio, httpx |
| **Containerização** | Docker, docker-compose |

---

## Controle de Acesso — Perfis

| Perfil | Tabelas | Operações | Mascaramento |
|--------|---------|-----------|:---:|
| `admin` | Todas | read / write / delete | Não |
| `analyst` | Todas | read | Não |
| `viewer` | campaigns, uplift_metrics | read | Sim (budget, margin, ROI) |

---

*Desenvolvido com [Claude Code](https://claude.ai/code) · Agnos AI Framework*
