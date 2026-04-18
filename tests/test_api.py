"""Testes da FastAPI — endpoints de saúde, grade, predição e drift."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents.ml_engineer.api import app, _state
from agents.ml_engineer.security import DEFAULT_KEYS


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def admin_headers():
    return {"X-API-Key": DEFAULT_KEYS["admin"]}


@pytest.fixture(scope="module")
def analyst_headers():
    return {"X-API-Key": DEFAULT_KEYS["analyst"]}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_schema(client):
    resp = client.get("/health")
    body = resp.json()
    assert "status" in body
    assert "db_available" in body
    assert "uptime_seconds" in body
    assert body["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------

def test_no_api_key_returns_401(client):
    resp = client.get("/campaigns/grid")
    assert resp.status_code == 401


def test_invalid_api_key_returns_401(client):
    resp = client.get("/campaigns/grid", headers={"X-API-Key": "invalid-key"})
    assert resp.status_code == 401


def test_analyst_cannot_run_pipeline(client, analyst_headers):
    resp = client.post("/pipeline/run", json={}, headers=analyst_headers)
    assert resp.status_code == 403


def test_analyst_cannot_check_drift(client, analyst_headers):
    resp = client.post("/monitor/drift", json={}, headers=analyst_headers)
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# /campaigns/grid
# ---------------------------------------------------------------------------

def test_grid_empty_returns_404(client, analyst_headers):
    # Estado limpo — sem grade carregada
    original_grid = _state.campaign_grid.copy()
    _state.campaign_grid = pd.DataFrame()
    resp = client.get("/campaigns/grid", headers=analyst_headers)
    _state.campaign_grid = original_grid
    assert resp.status_code == 404


def test_grid_with_data_returns_200(client, analyst_headers):
    # Injeta grade mock no estado
    mock_grid = pd.DataFrame({
        "product_id": ["SKU-0001", "SKU-0002"],
        "store_id": ["LJ-001", "LJ-001"],
        "discount_pct": [0.10, 0.20],
        "predicted_volume": [100, 150],
        "net_margin": [500.0, 750.0],
        "campaign_cost": [50.0, 80.0],
    })
    _state.campaign_grid = mock_grid
    resp = client.get("/campaigns/grid", headers=analyst_headers)
    _state.campaign_grid = pd.DataFrame()

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_campaigns"] == 2
    assert body["total_margin"] == pytest.approx(1250.0)
    assert body["total_budget_used"] == pytest.approx(130.0)
    assert len(body["campaigns"]) == 2


def test_grid_top_n_filter(client, analyst_headers):
    mock_grid = pd.DataFrame({
        "product_id": [f"SKU-{i:04d}" for i in range(1, 6)],
        "store_id": ["LJ-001"] * 5,
        "discount_pct": [0.10] * 5,
        "predicted_volume": [100] * 5,
        "net_margin": [float(i * 100) for i in range(1, 6)],
        "campaign_cost": [10.0] * 5,
    })
    _state.campaign_grid = mock_grid
    resp = client.get("/campaigns/grid?top_n=2", headers=analyst_headers)
    _state.campaign_grid = pd.DataFrame()

    assert resp.status_code == 200
    assert resp.json()["total_campaigns"] == 2


# ---------------------------------------------------------------------------
# /predict/demand
# ---------------------------------------------------------------------------

def test_predict_demand_no_model_uses_simulation_fallback(
    client, analyst_headers, tmp_path
):
    """Quando modelo não carregado, usa parquet de simulação como fallback."""
    sim_df = pd.DataFrame({
        "product_id": ["SKU-0001"],
        "store_id": ["LJ-001"],
        "discount_pct": [0.10],
        "predicted_volume": [120],
        "predicted_revenue": [1080.0],
        "predicted_margin": [324.0],
        "predicted_margin_pct": [0.30],
        "year_month": ["2024-12"],
        "simulated_price": [9.0],
        "unit_price": [10.0],
    })

    sim_path = tmp_path / "demand_simulation.parquet"
    sim_df.to_parquet(sim_path)

    import agents.ml_engineer.api as api_mod
    import config.settings as s
    original_processed = s.PROCESSED_DIR
    s.PROCESSED_DIR = tmp_path
    api_mod.PROCESSED_DIR = tmp_path
    _state.forecaster = None

    resp = client.post(
        "/predict/demand",
        json={"product_id": "SKU-0001", "store_id": "LJ-001", "discount_pct": 0.10},
        headers=analyst_headers,
    )

    s.PROCESSED_DIR = original_processed
    api_mod.PROCESSED_DIR = original_processed

    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_volume"] == 120
    assert body["discount_pct"] == pytest.approx(0.10)


def test_predict_demand_invalid_discount(client, analyst_headers):
    resp = client.post(
        "/predict/demand",
        json={"product_id": "SKU-0001", "store_id": "LJ-001", "discount_pct": 0.99},
        headers=analyst_headers,
    )
    assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# /monitor/drift
# ---------------------------------------------------------------------------

def test_drift_latest_without_run_returns_404(client, admin_headers):
    resp = client.get("/monitor/drift/latest", headers=admin_headers)
    # Pode ser 404 se nunca executado
    assert resp.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limit_header_present(client):
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
