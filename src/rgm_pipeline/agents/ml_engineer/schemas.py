"""
[Agente Engenheiro de ML] — Módulo 4: Schemas Pydantic da API

Define contratos de entrada e saída de todos os endpoints FastAPI,
garantindo validação automática e documentação OpenAPI gerada.
"""

from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Requisições
# ---------------------------------------------------------------------------

class DemandPredictRequest(BaseModel):
    """Payload para predição de demanda sob cenário de desconto."""

    product_id: str = Field(..., examples=["SKU-0001"])
    store_id: str = Field(..., examples=["LJ-001"])
    discount_pct: float = Field(
        ..., ge=0.0, le=0.60, description="Desconto entre 0% e 60%"
    )
    target_date: Optional[str] = Field(
        None,
        description="Data alvo no formato YYYY-MM (default: próximo mês)",
        examples=["2025-03"],
    )

    @field_validator("discount_pct")
    @classmethod
    def round_discount(cls, v: float) -> float:
        return round(v, 4)


class PipelineRunRequest(BaseModel):
    """Parâmetros para execução do pipeline completo."""

    total_budget: float = Field(
        default=5_000_000.0,
        gt=0,
        description="Verba máxima para geração da grade de campanhas (R$)",
    )
    max_campaigns_per_product: int = Field(
        default=2, ge=1, le=10
    )
    campaign_cost_pct: float = Field(
        default=0.05, ge=0.0, le=0.50,
        description="Percentual da receita como custo operacional da campanha",
    )


class DriftCheckRequest(BaseModel):
    """Payload para verificação de drift de dados ou modelo."""

    check_type: Literal["data", "model", "both"] = "both"
    n_reference_days: int = Field(
        default=90, ge=30, le=365,
        description="Janela de dias para o conjunto de referência",
    )
    n_current_days: int = Field(
        default=30, ge=7, le=90,
        description="Janela de dias recentes para comparação",
    )


# ---------------------------------------------------------------------------
# Respostas
# ---------------------------------------------------------------------------

class DemandPredictResponse(BaseModel):
    product_id: str
    store_id: str
    discount_pct: float
    predicted_volume: int
    predicted_revenue: float
    predicted_margin: float
    predicted_margin_pct: float
    confidence_score: float
    confidence_label: Literal["ALTA", "MÉDIA", "BAIXA"]


class CampaignItem(BaseModel):
    product_id: str
    store_id: str
    discount_pct: float
    predicted_volume: int
    net_margin: float
    campaign_cost: float
    confidence_score: Optional[float] = None
    confidence_label: Optional[str] = None
    explanation: Optional[str] = None


class CampaignGridResponse(BaseModel):
    total_campaigns: int
    total_margin: float
    total_budget_used: float
    solver_status: str
    generated_at: datetime
    campaigns: list[CampaignItem]


class PipelineRunResponse(BaseModel):
    status: Literal["success", "error"]
    message: str
    did_att_volume: Optional[float] = None
    did_att_margin: Optional[float] = None
    forecast_mape: Optional[float] = None
    n_campaigns: Optional[int] = None
    total_margin: Optional[float] = None
    duration_seconds: Optional[float] = None


class DriftReport(BaseModel):
    check_type: str
    data_drift_detected: Optional[bool] = None
    model_drift_detected: Optional[bool] = None
    drifted_features: list[str] = []
    drift_score: Optional[float] = None
    recommendation: str
    checked_at: datetime


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    db_available: bool
    model_loaded: bool
    grid_available: bool
    uptime_seconds: float
    version: str = "1.0.0"
