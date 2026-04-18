"""
[Agente Engenheiro de ML] — Módulo 4: FastAPI Application

Integra os pipelines dos três agentes anteriores em uma API REST:

  GET  /health                    — Verificação de saúde do sistema
  POST /pipeline/run              — Executa o pipeline completo (admin)
  GET  /campaigns/grid            — Retorna a grade de campanhas otimizada
  POST /predict/demand            — Prediz demanda para cenário de desconto
  POST /monitor/drift             — Verifica drift de dados e modelo (admin)
  GET  /monitor/drift/latest      — Último relatório de drift (admin)
"""

import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from agents.ml_engineer.schemas import (
    CampaignGridResponse,
    CampaignItem,
    DemandPredictRequest,
    DemandPredictResponse,
    DriftCheckRequest,
    DriftReport,
    HealthResponse,
    PipelineRunRequest,
    PipelineRunResponse,
)
from agents.ml_engineer.security import (
    DEFAULT_KEYS,
    RateLimitMiddleware,
    RequestTracingMiddleware,
    require_admin,
    require_api_key,
)
from agents.ml_engineer.drift_monitor import DriftMonitor
from config.settings import DB_PATH, PROCESSED_DIR


# ---------------------------------------------------------------------------
# Estado global da aplicação (carregado no startup)
# ---------------------------------------------------------------------------

class AppState:
    """Mantém artefatos carregados em memória para servir requisições."""

    def __init__(self) -> None:
        self.startup_time: float = time.monotonic()
        self.campaign_grid: pd.DataFrame = pd.DataFrame()
        self.explanations: pd.DataFrame = pd.DataFrame()
        self.forecaster: Any = None       # DemandForecaster (carregado sob demanda)
        self._grid_loaded_at: datetime | None = None

    def load_artifacts(self) -> None:
        """Carrega artefatos persistidos pelos módulos anteriores."""
        grid_path = PROCESSED_DIR / "campaign_grid.parquet"
        exp_path = PROCESSED_DIR / "campaign_explanations.parquet"

        if grid_path.exists():
            self.campaign_grid = pd.read_parquet(grid_path)
            self._grid_loaded_at = datetime.utcnow()
            logger.info(f"[API] Grade carregada: {len(self.campaign_grid)} campanhas")
        else:
            logger.warning("[API] campaign_grid.parquet não encontrado. Execute o pipeline.")

        if exp_path.exists():
            self.explanations = pd.read_parquet(exp_path)
            logger.info(f"[API] Explicações carregadas: {len(self.explanations)} registros")

    @property
    def uptime(self) -> float:
        return time.monotonic() - self.startup_time


_state = AppState()
_drift_monitor = DriftMonitor()
_latest_drift_report: DriftReport | None = None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API] Iniciando RGM Pipeline API...")
    logger.info(f"[API] API Keys de desenvolvimento:\n"
                f"  admin   → {DEFAULT_KEYS['admin']}\n"
                f"  analyst → {DEFAULT_KEYS['analyst']}")
    _state.load_artifacts()
    yield
    logger.info("[API] Encerrando RGM Pipeline API.")


# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RGM Pipeline API",
    description=(
        "API do produto de dados RGM (Revenue Growth Management). "
        "Orquestra geração de grade de campanhas promocionais otimizadas "
        "via sistema multi-agentes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Middlewares
app.add_middleware(RequestTracingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Infra"])
async def health_check():
    """Verifica disponibilidade dos componentes críticos."""
    return HealthResponse(
        status="healthy" if DB_PATH.exists() else "degraded",
        db_available=DB_PATH.exists(),
        model_loaded=_state.forecaster is not None,
        grid_available=not _state.campaign_grid.empty,
        uptime_seconds=round(_state.uptime, 1),
    )


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@app.post(
    "/pipeline/run",
    response_model=PipelineRunResponse,
    tags=["Pipeline"],
    summary="Executa pipeline completo (admin only)",
)
async def run_pipeline(
    body: PipelineRunRequest,
    caller: dict = Depends(require_admin),
):
    """
    Executa sequencialmente todos os módulos:
      1. Baseline Causal
      2. Demand Forecasting
      3. Otimização da Grade
      4. Explicabilidade

    Requer role **admin**.
    """
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail="Banco de dados não encontrado. Execute run_module1.py primeiro.",
        )

    start = time.monotonic()
    logger.info(f"[API] Pipeline iniciado por '{caller['owner']}'")

    try:
        # Importações locais para não bloquear o startup da API
        from agents.data_scientist.causal_baseline import CausalBaselineEstimator
        from agents.data_scientist.demand_forecasting import DemandForecaster
        from agents.data_scientist.optimizer import CampaignOptimizer
        from agents.data_scientist.explainability import CampaignExplainer
        import sqlite3

        # Módulo 2a: Causal
        did_result = CausalBaselineEstimator().run()

        # Módulo 2b: Forecast
        forecaster = DemandForecaster()
        forecast_result = forecaster.run()
        _state.forecaster = forecaster

        # Módulo 3a: Otimização
        opt_result = CampaignOptimizer(
            total_budget=body.total_budget,
            max_campaigns_per_product=body.max_campaigns_per_product,
            campaign_cost_pct=body.campaign_cost_pct,
        ).run(simulation_df=forecast_result.simulation_df)

        # Módulo 3b: XAI
        with sqlite3.connect(DB_PATH) as conn:
            raw_txn = pd.read_sql(
                "SELECT * FROM transactions", conn, parse_dates=["date"]
            )
        full_eng = forecaster._engineer_features(raw_txn, fit=False)

        CampaignExplainer(
            model=forecaster._model,
            feature_cols=forecaster._feature_cols,
        ).run(
            grid=opt_result.campaign_grid,
            full_data=full_eng,
            model_mape=forecast_result.mape,
        )

        # Atualiza estado em memória
        _state.load_artifacts()

        duration = time.monotonic() - start
        logger.success(f"[API] Pipeline concluído em {duration:.1f}s")

        return PipelineRunResponse(
            status="success",
            message="Pipeline executado com sucesso.",
            did_att_volume=round(did_result.att, 2),
            did_att_margin=round(did_result.att_margin, 2),
            forecast_mape=round(forecast_result.mape, 4),
            n_campaigns=opt_result.n_campaigns,
            total_margin=round(opt_result.total_margin, 2),
            duration_seconds=round(duration, 2),
        )

    except Exception as exc:
        logger.exception(f"[API] Falha no pipeline: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no pipeline: {str(exc)}",
        )


# ------------------------------------------------------------------
# Grade de Campanhas
# ------------------------------------------------------------------

@app.get(
    "/campaigns/grid",
    response_model=CampaignGridResponse,
    tags=["Campanhas"],
)
async def get_campaign_grid(
    top_n: int = 50,
    min_confidence: float = 0.0,
    caller: dict = Depends(require_api_key),
):
    """
    Retorna a grade de campanhas otimizada.

    Parâmetros:
      - **top_n**: número máximo de campanhas retornadas (ordenadas por margem)
      - **min_confidence**: filtra por score de confiança mínimo (0–1)
    """
    if _state.campaign_grid.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Grade não disponível. Execute POST /pipeline/run.",
        )

    grid = _state.campaign_grid.copy()

    # Junta com explicações se disponível
    if not _state.explanations.empty:
        merge_cols = ["product_id", "store_id", "discount_pct"]
        exp_cols = merge_cols + ["confidence_score", "confidence_label", "explanation"]
        available = [c for c in exp_cols if c in _state.explanations.columns]
        grid = grid.merge(
            _state.explanations[available],
            on=merge_cols,
            how="left",
        )

    if min_confidence > 0 and "confidence_score" in grid.columns:
        grid = grid[grid["confidence_score"] >= min_confidence]

    grid = grid.head(top_n)

    campaigns = [
        CampaignItem(
            product_id=row["product_id"],
            store_id=row["store_id"],
            discount_pct=row["discount_pct"],
            predicted_volume=int(row.get("predicted_volume", 0)),
            net_margin=round(float(row["net_margin"]), 2),
            campaign_cost=round(float(row["campaign_cost"]), 2),
            confidence_score=row.get("confidence_score"),
            confidence_label=row.get("confidence_label"),
            explanation=row.get("explanation"),
        )
        for _, row in grid.iterrows()
    ]

    # Totais do grid filtrado (não do estado global)
    total_margin = float(grid["net_margin"].sum())
    total_budget = float(grid["campaign_cost"].sum())

    return CampaignGridResponse(
        total_campaigns=len(campaigns),
        total_margin=round(total_margin, 2),
        total_budget_used=round(total_budget, 2),
        solver_status="Optimal",
        generated_at=datetime.utcnow(),
        campaigns=campaigns,
    )


# ------------------------------------------------------------------
# Predição de Demanda
# ------------------------------------------------------------------

@app.post(
    "/predict/demand",
    response_model=DemandPredictResponse,
    tags=["Predição"],
)
async def predict_demand(
    body: DemandPredictRequest,
    caller: dict = Depends(require_api_key),
):
    """
    Prediz volume e margem para um produto-loja com cenário de desconto.

    Requer que o modelo tenha sido treinado via POST /pipeline/run.
    """
    if _state.forecaster is None:
        # Tenta carregar simulação em disco como fallback
        sim_path = PROCESSED_DIR / "demand_simulation.parquet"
        if not sim_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo não carregado. Execute POST /pipeline/run.",
            )
        sim = pd.read_parquet(sim_path)
        row = sim[
            (sim["product_id"] == body.product_id) &
            (sim["store_id"] == body.store_id) &
            (sim["discount_pct"].round(2) == round(body.discount_pct, 2))
        ]
        if row.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Produto '{body.product_id}' / loja '{body.store_id}' não encontrado na simulação.",
            )
        r = row.iloc[0]
        return DemandPredictResponse(
            product_id=body.product_id,
            store_id=body.store_id,
            discount_pct=body.discount_pct,
            predicted_volume=int(r["predicted_volume"]),
            predicted_revenue=float(r["predicted_revenue"]),
            predicted_margin=float(r["predicted_margin"]),
            predicted_margin_pct=float(r["predicted_margin_pct"]),
            confidence_score=0.70,
            confidence_label="MÉDIA",
        )

    # Usa modelo em memória
    forecaster = _state.forecaster
    with sqlite3.connect(DB_PATH) as conn:
        sample = pd.read_sql(
            "SELECT * FROM transactions WHERE product_id = ? AND store_id = ? LIMIT 200",
            conn,
            params=(body.product_id, body.store_id),
            parse_dates=["date"],
        )

    if sample.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhum histórico para produto '{body.product_id}' / loja '{body.store_id}'.",
        )

    # fit=False: usa encoders já ajustados no treino, sem re-fittar
    eng = forecaster._engineer_features(sample, fit=False)
    X = eng[forecaster._feature_cols].tail(1).copy()
    X["discount_pct"] = body.discount_pct

    pred_log = forecaster._model.predict(X)
    pred_vol = max(0, int(np.round(np.expm1(pred_log[0]))))

    base_price = float(sample["unit_price"].iloc[-1])
    sim_price = round(base_price * (1 - body.discount_pct), 2)
    pred_revenue = round(pred_vol * sim_price, 2)
    base_margin_pct = float(sample["margin_pct"].mean())
    pred_margin_pct = round(base_margin_pct * (1 - 1.5 * body.discount_pct), 4)
    pred_margin = round(pred_revenue * pred_margin_pct, 2)

    return DemandPredictResponse(
        product_id=body.product_id,
        store_id=body.store_id,
        discount_pct=body.discount_pct,
        predicted_volume=pred_vol,
        predicted_revenue=pred_revenue,
        predicted_margin=pred_margin,
        predicted_margin_pct=pred_margin_pct,
        confidence_score=0.75,
        confidence_label="ALTA" if pred_vol > 0 else "BAIXA",
    )


# ------------------------------------------------------------------
# Drift Monitoring
# ------------------------------------------------------------------

@app.post(
    "/monitor/drift",
    response_model=DriftReport,
    tags=["Monitoramento"],
    summary="Executa verificação de drift (admin only)",
)
async def check_drift(
    body: DriftCheckRequest,
    caller: dict = Depends(require_admin),
):
    """Detecta data drift e model drift. Requer role **admin**."""
    global _latest_drift_report

    if not DB_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail="Banco de dados não disponível.",
        )

    result = _drift_monitor.run(
        check_type=body.check_type,
        n_reference_days=body.n_reference_days,
        n_current_days=body.n_current_days,
    )

    report = DriftReport(
        check_type=result.check_type,
        data_drift_detected=result.data_drift_detected,
        model_drift_detected=result.model_drift_detected,
        drifted_features=result.drifted_features,
        drift_score=result.drift_score,
        recommendation=result.recommendation,
        checked_at=result.checked_at,
    )
    _latest_drift_report = report
    return report


@app.get(
    "/monitor/drift/latest",
    response_model=DriftReport,
    tags=["Monitoramento"],
)
async def get_latest_drift(caller: dict = Depends(require_admin)):
    """Retorna o último relatório de drift disponível."""
    if _latest_drift_report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhum relatório de drift disponível. Execute POST /monitor/drift.",
        )
    return _latest_drift_report
