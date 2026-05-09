# Backlog de Ideias — RGM Pipeline

> Ideias de evolução baseadas no estado da arte global de Revenue Growth Management (2024–2025).
> Fontes: BCG, McKinsey, Nielsen IQ, Kantar, NielsenIQ/NIQ e publicações acadêmicas recentes.

---

## Sumário

- [1. Modelagem Promocional Avançada](#1-modelagem-promocional-avançada)
- [2. Inteligência de Preços e Portfólio](#2-inteligência-de-preços-e-portfólio)
- [3. Dados Externos e Contexto de Mercado](#3-dados-externos-e-contexto-de-mercado)
- [4. Causal AI de Nova Geração](#4-causal-ai-de-nova-geração)
- [5. Otimização de Nova Geração](#5-otimização-de-nova-geração)
- [6. MLOps e Produção](#6-mlops-e-produção)
- [7. Interface e Experiência do Usuário](#7-interface-e-experiência-do-usuário)
- [8. Tendências Estratégicas do Mercado Global](#8-tendências-estratégicas-do-mercado-global)
- [Priorização](#priorização)

---

## 1. Modelagem Promocional Avançada

### 1.1 Halo & Cannibalization
- **O problema:** quando um SKU entra em promoção, vendas de SKUs complementares sobem (halo) e vendas de substitutos caem (canibalização). Estima-se que **30–40% do volume incremental aparente é antecipação ou desvio de categoria**.
- **Abordagem:** grafo de relacionamento entre SKUs (co-compra, complementaridade, substituição) + modelo de efeitos cruzados via **Graph Neural Networks (GNNs)**.
- **Referência de mercado:** Kraft Heinz e Unilever reportaram ganhos de 8–12% no ROI de verba trade ao incorporar esses efeitos.
- **Status:** `[ ] Não iniciado`

### 1.2 Post-Promotion Dip (Demanda Emprestada)
- **O problema:** após o fim de uma promoção, vendas caem abaixo do baseline por 2–4 semanas porque o consumidor se abasteceu. O otimizador atual pode escolher campanhas com ROI aparente alto, mas ROI líquido negativo.
- **Abordagem:** adicionar janela de *payback* na função objetivo do ILP, estimada via análise de séries temporais pós-campanha.
- **Status:** `[ ] Não iniciado`

### 1.3 Curvas de Elasticidade com Incerteza (Conformal Prediction)
- **O problema:** o XGBoost retorna previsões pontuais. O mercado migrou para **intervalos de predição calibrados** que distinguem incerteza epistêmica (falta de dados) de aleatória (variabilidade real).
- **Abordagem:** *Conformal Prediction* — sem premissas distribucionais, encaixa sobre qualquer modelo existente. Biblioteca: `MAPIE`.
- **Status:** `[ ] Não iniciado`

---

## 2. Inteligência de Preços e Portfólio

### 2.1 Price Pack Architecture (PPA)
- **O problema:** RGM moderno vai além de "quanto descontar" — inclui qual embalagem, em qual canal, a qual preço de prateleira. Um dos pilares do **RGM 3.0** segundo a BCG.
- **Abordagem:** análise de PPA para identificar tamanhos e formatos ideais por segmento de consumidor e canal (hipermercado vs. conveniência vs. e-commerce).
- **Status:** `[ ] Não iniciado`

### 2.2 Price-to-Value (PtV) Scoring
- **O problema:** o modelo de elasticidade não diferencia produtos percebidos como premium de produtos de entrada. A resposta ao desconto é estruturalmente diferente entre eles.
- **Abordagem:** calibrar elasticidade com dados de percepção de valor (Nielsen HomeScan, Kantar Worldpanel).
- **Status:** `[ ] Não iniciado`

### 2.3 Competitive Price Intelligence
- **O problema:** o modelo prevê demanda em função do desconto próprio, ignorando o concorrente. Um desconto de 20% com concorrente a 30% gera resposta muito diferente de um desconto isolado.
- **Abordagem:** dados de preço de prateleira via **Dunnhumby, Profitero ou web scraping estruturado** como feature no DemandForecaster.
- **Status:** `[ ] Não iniciado`

---

## 3. Dados Externos e Contexto de Mercado

### 3.1 Variáveis Macroeconômicas e de Consumidor

| Fonte | Variável | Impacto esperado |
|-------|----------|-----------------|
| IBGE / FGV | IPCA, renda disponível, confiança do consumidor | Elasticidade por categoria |
| Google Trends | Volume de buscas por produto/categoria | Leading indicator de demanda |
| INMET / OpenMeteo | Temperatura, precipitação | Bebidas, higiene, produtos sazonais |
| Calendário comercial | Datas sazonais, regionais, feriados locais | Picos não capturados pelo campo `month` |
| Redes sociais | Sentimento por marca/produto | Demanda de curto prazo |

- **Status:** `[ ] Não iniciado`

### 3.2 Scan Data de Mercado (Market Share)
- **O problema:** sem dados de mercado, não é possível distinguir crescimento de categoria de ganho de share. O uplift pode ser superestimado em mercados em expansão.
- **Abordagem:** integrar dados de sell-out de mercado (Nielsen RMS ou IQVIA) para modelar **elasticidade relativa ao mercado**.
- **Status:** `[ ] Não iniciado`

---

## 4. Causal AI de Nova Geração

### 4.1 Double/Debiased ML (DoubleML)
- **O problema:** o DiD atual assume paralelismo de tendências e pode produzir estimativas viesadas na presença de confundidores de alta dimensionalidade.
- **Abordagem:** separar estimação do efeito de tratamento dos confundidores em dois estágios. Produz intervalos de confiança válidos mesmo com muitas features. Biblioteca: `econml` (já no `requirements.txt`).
- **Status:** `[ ] Não iniciado`

### 4.2 Synthetic Control Method
- **O problema:** para campanhas com poucos produtos tratados, o grupo de controle do DiD pode não ser comparável.
- **Abordagem:** criar um "controle sintético" combinando produtos não-tratados similares com pesos otimizados.
- **Status:** `[ ] Não iniciado`

### 4.3 Uplift Modeling — CATE (Efeito Individual de Tratamento)
- **O problema:** o modelo atual estima um efeito médio. Mas em qual par produto-loja a promoção *realmente* funciona? Concentrar verba onde o retorno marginal é alto pode aumentar o ROI total em 15–25%.
- **Abordagem:** **X-Learner ou R-Learner** via `econml` para estimar o CATE (Conditional Average Treatment Effect) por produto-loja.
- **Referência de mercado:** padrão em empresas tier-1 de CPG desde 2022.
- **Status:** `[ ] Não iniciado`

---

## 5. Otimização de Nova Geração

### 5.1 Otimização Multi-Período com Restrições Dinâmicas
- **O problema:** o ILP atual é estático (uma grade por rodada). Não considera post-promotion dip, budget rolling nem dependências temporais entre campanhas.
- **Abordagem:** programação inteira mista multi-período com restrições de sell-in/sell-out, verba rolling e janelas de exclusão pós-promoção.
- **Status:** `[ ] Não iniciado`

### 5.2 Reinforcement Learning — Contextual Bandits
- **O problema:** o modelo de otimização não aprende com o resultado das campanhas passadas de forma online.
- **Abordagem:** **Contextual Bandits** — o modelo aprende qual campanha funciona melhor em qual contexto sem precisar de simulador completo. Mais prático que RL full para ciclos semanais/mensais.
- **Referência de mercado:** Amazon, Magazine Luiza e Mercado Libre usam variantes desta abordagem para pricing dinâmico.
- **Status:** `[ ] Não iniciado`

### 5.3 Otimização Multi-Echelon (Indústria + Varejo)
- **O problema:** otimizar apenas no nível varejo pode ser subótimo para a indústria. A cadeia completa (fabricante → distribuidor → varejo) tem objetivos parcialmente conflitantes.
- **Abordagem:** programação bi-nível (Stackelberg game) ou otimização colaborativa via troca de dados agregados anonimizados.
- **Status:** `[ ] Não iniciado`

---

## 6. MLOps e Produção

### 6.1 Feature Store
- **O problema:** features são recalculadas a cada treino, sem garantia de consistência entre treino e inferência (*training-serving skew*).
- **Abordagem:** Feature Store centralizada (Feast ou Hopsworks) com versionamento, reutilização entre modelos e point-in-time correct joins.
- **Status:** `[ ] Não iniciado`

### 6.2 Retraining Automático com Trigger por Drift
- **O problema:** o `DriftMonitor` detecta drift, mas o retraining ainda é manual.
- **Abordagem:** pipeline automatizado — detectou drift → dispara retraining → avalia métricas em holdout → promove ou reverte o modelo via feature flag. Stack sugerido: MLflow + Prefect ou Airflow.
- **Status:** `[ ] Não iniciado`

### 6.3 Model Registry com A/B Testing
- **O problema:** não há como comparar modelos em produção com tráfego real.
- **Abordagem:** versionar modelos no MLflow e suportar shadow deployment ou A/B (ex.: 10% do tráfego no modelo candidato, 90% no modelo de produção) com coleta de métricas online.
- **Status:** `[ ] Não iniciado`

### 6.4 Batch Inference Escalável
- **O problema:** para bases com milhares de SKUs × centenas de lojas, o processamento Pandas pode levar horas.
- **Abordagem:** migrar batch inference para **DuckDB** (drop-in, sem infra extra) ou **Spark** para escala enterprise. A lógica dos modelos permanece idêntica.
- **Status:** `[ ] Não iniciado`

---

## 7. Interface e Experiência do Usuário

### 7.1 Agente Conversacional para RGM (NL Interface)
- **O problema:** gestores de trade marketing não lêem dashboards de BI — eles fazem perguntas. Ex.: *"Qual produto da categoria Bebidas tem melhor ROI esperado com 20% de desconto em São Paulo no próximo trimestre?"*
- **Abordagem:** LLM com **tool calling** sobre os endpoints FastAPI existentes. O agente consulta dados, executa simulações e responde com contexto narrativo. O `RealLLMForecaster` do notebook é o embrião desta funcionalidade.
- **Status:** `[ ] Não iniciado`

### 7.2 What-If Simulator Interativo
- **O problema:** a grade de campanhas é gerada em batch. Não há como o usuário explorar cenários interativamente.
- **Abordagem:** interface Streamlit (extensão do `dashboard.py` atual) com sliders de desconto, verba e período conectados ao endpoint `POST /predict/demand` em tempo real.
- **Status:** `[ ] Não iniciado`

### 7.3 Alertas Proativos
- **O problema:** o monitoramento de drift é reativo (chamado por API). Campanhas que desviam da previsão não geram alertas automáticos.
- **Abordagem:** job agendado (cron ou Prefect) que compara sell-out real com previsto e envia alertas (Slack, e-mail, Teams) quando o desvio supera um threshold configurável.
- **Status:** `[ ] Não iniciado`

---

## 8. Tendências Estratégicas do Mercado Global

### 8.1 Omnichannel RGM
- **O problema:** modelos separados para físico vs. digital não capturam efeitos cruzados entre canais. Uma promoção no app pode canibalizar a loja física.
- **Referência de mercado:** P&G e Heineken reportaram 15–20% de melhora no ROI ao unificar modelos de promoção cross-channel.
- **Status:** `[ ] Não iniciado`

### 8.2 ESG / Sustentabilidade integrada ao RGM
- **O problema:** campanhas que geram alto volume incremental de produtos com grande pegada de carbono não são penalizadas na otimização atual.
- **Abordagem:** adicionar "custo sombra" de carbono por tonelada incremental vendida na função objetivo do ILP. Regulação europeia (CSRD) deve tornar isso mandatório para grandes empresas até 2026.
- **Status:** `[ ] Não iniciado`

### 8.3 Federated Learning (Privacy-Preserving RGM)
- **O problema:** com o fim dos cookies third-party e maior restrição de dados (LGPD/GDPR), centralizar dados de múltiplos varejistas é inviável legalmente.
- **Abordagem:** modelo federado onde cada varejista treina localmente e apenas os gradientes agregados são compartilhados — sem expor dados proprietários.
- **Status:** `[ ] Não iniciado`

### 8.4 Demand Sensing em Tempo Real
- **O problema:** RGM tradicional opera em ciclos semanais/mensais. O mercado de FMCG está migrando para janelas de 24–72 horas usando dados de PDV em tempo real.
- **Abordagem:** ingestão contínua via EDI ou API de varejistas parceiros + modelo de forecasting online (atualização incremental, não retraining completo).
- **Status:** `[ ] Não iniciado`

---

## Priorização

> Ordenado por relação impacto × complexidade.

| # | Ideia | Impacto no ROI | Complexidade | Prazo estimado |
|---|-------|:--------------:|:------------:|:--------------:|
| 1 | Post-Promotion Dip no otimizador | Alto | Baixa | 2–3 semanas |
| 2 | Conformal Prediction (intervalos de incerteza) | Médio | Baixa | 1–2 semanas |
| 3 | Variáveis externas (clima, macro, calendário) | Alto | Média | 3–4 semanas |
| 4 | Double ML / Uplift Modeling (CATE) | Alto | Média | 4–6 semanas |
| 5 | What-If Simulator Interativo | Médio | Média | 3–4 semanas |
| 6 | Alertas Proativos de Desvio | Médio | Baixa | 1–2 semanas |
| 7 | Agente Conversacional NL (LLM + tool calling) | Médio | Média | 3–5 semanas |
| 8 | Retraining Automático por Drift | Médio | Alta | 2–3 meses |
| 9 | Competitive Price Intelligence | Alto | Alta | 2–3 meses |
| 10 | Halo & Cannibalization (GNN) | Muito Alto | Alta | 2–3 meses |
| 11 | Feature Store (Feast / Hopsworks) | Médio | Alta | 2–3 meses |
| 12 | Batch Inference Escalável (DuckDB / Spark) | Médio | Média | 3–4 semanas |
| 13 | Synthetic Control / DoubleML | Alto | Média | 4–6 semanas |
| 14 | Price Pack Architecture (PPA) | Alto | Alta | 2–3 meses |
| 15 | Omnichannel RGM | Muito Alto | Muito Alta | 4–6 meses |
| 16 | Contextual Bandits (RL) | Alto | Muito Alta | 3–6 meses |
| 17 | Otimização Multi-Período | Alto | Alta | 2–3 meses |
| 18 | Demand Sensing em Tempo Real | Alto | Muito Alta | 4–6 meses |
| 19 | ESG / Custo Sombra de Carbono | Médio | Média | 3–4 semanas |
| 20 | Federated Learning | Alto | Muito Alta | 6+ meses |

---

*Documento gerado em 2026-04-24. Atualizar à medida que itens forem iniciados ou concluídos.*
