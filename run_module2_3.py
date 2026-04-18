"""
Script de execução dos Módulos 2 e 3 — [Agente Cientista de Dados]

Orquestra:
  1. Baseline Causal (DiD + regressão contrafactual)
  2. Demand Forecasting (XGBoost + simulação de cenários de desconto)
  3. Otimização Matemática (ILP via PuLP — maximiza margem)
  4. Explicabilidade (SHAP + Score de Confiança)

Pré-requisito: run_module1.py deve ter sido executado (banco SQLite disponível).
"""

from loguru import logger

from agents.data_scientist.causal_baseline import CausalBaselineEstimator
from agents.data_scientist.demand_forecasting import DemandForecaster
from agents.data_scientist.optimizer import CampaignOptimizer
from agents.data_scientist.explainability import CampaignExplainer
from config.settings import DB_PATH


def main() -> None:
    logger.info("=" * 60)
    logger.info("RGM Pipeline — Módulos 2 e 3: Agente Cientista de Dados")
    logger.info("=" * 60)

    if not DB_PATH.exists():
        logger.error(
            f"Banco de dados não encontrado em {DB_PATH}. "
            "Execute run_module1.py primeiro."
        )
        return

    # ----------------------------------------------------------------
    # Módulo 2a: Baseline Causal
    # ----------------------------------------------------------------
    baseline_estimator = CausalBaselineEstimator()
    did_result = baseline_estimator.run()

    logger.info(
        f"\n[DiD Summary]\n"
        f"  ATT Volume  : {did_result.att:+.1f} unidades/mês\n"
        f"  ATT Receita : R$ {did_result.att_revenue:+,.2f}\n"
        f"  ATT Margem  : R$ {did_result.att_margin:+,.2f}\n"
        f"  Produtos tratados : {did_result.n_treated}\n"
        f"  Produtos controle : {did_result.n_control}"
    )

    # ----------------------------------------------------------------
    # Módulo 2b: Demand Forecasting + Simulação
    # ----------------------------------------------------------------
    forecaster = DemandForecaster()
    forecast_result = forecaster.run()

    logger.info(
        f"\n[Forecast Summary]\n"
        f"  MAPE médio   : {forecast_result.mape:.2%}\n"
        f"  Cenários     : {len(forecast_result.simulation_df):,}\n"
        f"  Top features :\n"
        f"{forecast_result.feature_importance.head(5).to_string(index=False)}"
    )

    # ----------------------------------------------------------------
    # Módulo 3a: Otimização
    # ----------------------------------------------------------------
    optimizer = CampaignOptimizer(
        total_budget=5_000_000.0,
        max_campaigns_per_product=2,
        campaign_cost_pct=0.05,
    )
    opt_result = optimizer.run(simulation_df=forecast_result.simulation_df)

    logger.info(
        f"\n[Optimizer Summary]\n"
        f"  Status        : {opt_result.solver_status}\n"
        f"  Campanhas     : {opt_result.n_campaigns}\n"
        f"  Margem total  : R$ {opt_result.total_margin:,.2f}\n"
        f"  Verba usada   : R$ {opt_result.total_budget_used:,.2f}"
    )

    if opt_result.campaign_grid.empty:
        logger.warning("Grade de campanhas vazia — encerrando.")
        return

    print("\n[Grade de Campanhas — Top 10]")
    print(
        opt_result.campaign_grid[[
            "product_id", "store_id", "discount_pct",
            "predicted_volume", "net_margin", "campaign_cost",
        ]].head(10).to_string(index=False)
    )

    # ----------------------------------------------------------------
    # Módulo 3b: Explicabilidade
    # ----------------------------------------------------------------
    # Precisa do modelo treinado e do dataset com features
    import sqlite3
    import pandas as pd
    with sqlite3.connect(DB_PATH) as conn:
        full_raw = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])

    full_engineered = forecaster._engineer_features(full_raw, fit=False)

    explainer = CampaignExplainer(
        model=forecaster._model,
        feature_cols=forecaster._feature_cols,
    )
    xai_result = explainer.run(
        grid=opt_result.campaign_grid,
        full_data=full_engineered,
        model_mape=forecast_result.mape,
    )

    logger.info(
        f"\n[XAI Summary]\n"
        f"  Campanhas explicadas : {len(xai_result.campaign_explanations)}\n"
        f"  Distribuição de confiança:\n"
        f"{xai_result.confidence_summary['confidence_label'].value_counts().to_string()}"
    )

    print("\n[Exemplos de Explicação]")
    for _, row in xai_result.campaign_explanations.head(3).iterrows():
        print(f"\n  → {row['explanation']}")

    logger.success("\n[Módulos 2 e 3] Concluídos com sucesso.")
    logger.info(
        "Artefatos gerados em data/processed/:\n"
        "  causal_baseline.parquet\n"
        "  demand_simulation.parquet\n"
        "  campaign_grid.parquet\n"
        "  campaign_explanations.parquet"
    )


if __name__ == "__main__":
    main()
