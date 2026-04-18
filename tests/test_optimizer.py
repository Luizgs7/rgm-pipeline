"""Testes do CampaignOptimizer — ILP, restrições e grade prescritiva."""

import numpy as np
import pandas as pd
import pytest

from agents.data_scientist.optimizer import CampaignOptimizer, OptimizationResult


@pytest.fixture()
def simulation_df(sample_transactions) -> pd.DataFrame:
    """Simulação mínima com 3 produtos × 2 lojas × 2 descontos."""
    rng = np.random.default_rng(0)
    products = ["SKU-0001", "SKU-0002", "SKU-0003"]
    stores = ["LJ-001", "LJ-002"]
    discounts = [0.10, 0.20]
    records = []
    for p in products:
        for s in stores:
            for d in discounts:
                vol = int(rng.integers(50, 200))
                price = round(rng.uniform(10, 50), 2)
                sim_price = round(price * (1 - d), 2)
                revenue = round(vol * sim_price, 2)
                margin_pct = max(0.05, 0.35 - 0.5 * d)
                margin = round(revenue * margin_pct, 2)
                records.append({
                    "product_id": p,
                    "store_id": s,
                    "year_month": "2024-12",
                    "discount_pct": d,
                    "predicted_volume": vol,
                    "unit_price": price,
                    "simulated_price": sim_price,
                    "predicted_margin_pct": margin_pct,
                    "predicted_revenue": revenue,
                    "predicted_margin": margin,
                })
    return pd.DataFrame(records)


@pytest.fixture(autouse=True)
def patch_processed_dir(tmp_path, monkeypatch):
    """Redireciona PROCESSED_DIR para diretório temporário nos testes."""
    import agents.data_scientist.optimizer as opt_mod
    monkeypatch.setattr(opt_mod, "PROCESSED_DIR", tmp_path)


@pytest.fixture()
def optimizer() -> CampaignOptimizer:
    return CampaignOptimizer(
        total_budget=500_000.0,
        max_campaigns_per_product=1,
        campaign_cost_pct=0.05,
    )


# ---------------------------------------------------------------------------
# Preparação de candidatos
# ---------------------------------------------------------------------------

def test_prepare_candidates_filters_negative_margin(simulation_df, optimizer):
    # Força algumas margens negativas
    df = simulation_df.copy()
    df.loc[df["product_id"] == "SKU-0001", "predicted_margin"] = -100.0
    cands = optimizer._prepare_candidates(df)
    assert not (cands["net_margin"] <= 0).any()


def test_prepare_candidates_computes_cost(simulation_df, optimizer):
    cands = optimizer._prepare_candidates(simulation_df)
    expected_cost = (cands["predicted_revenue"] * optimizer.campaign_cost_pct).round(2)
    assert (abs(cands["campaign_cost"] - expected_cost) < 0.01).all()


# ---------------------------------------------------------------------------
# Solver ILP
# ---------------------------------------------------------------------------

def test_solver_returns_optimal(simulation_df, optimizer):
    result = optimizer.run(simulation_df=simulation_df)
    assert result.solver_status in ("Optimal", "Not Solved", "Infeasible")
    # Com budget grande e candidatos viáveis, esperamos ótimo
    assert result.solver_status == "Optimal"


def test_grid_not_empty(simulation_df, optimizer):
    result = optimizer.run(simulation_df=simulation_df)
    assert result.n_campaigns > 0
    assert not result.campaign_grid.empty


def test_budget_constraint_respected(simulation_df, optimizer):
    result = optimizer.run(simulation_df=simulation_df)
    assert result.total_budget_used <= optimizer.total_budget + 0.01


def test_max_campaigns_per_product_respected(simulation_df, optimizer):
    result = optimizer.run(simulation_df=simulation_df)
    per_product = result.campaign_grid.groupby("product_id").size()
    assert (per_product <= optimizer.max_campaigns_per_product).all()


def test_total_margin_positive(simulation_df, optimizer):
    result = optimizer.run(simulation_df=simulation_df)
    assert result.total_margin > 0


def test_grid_columns_present(simulation_df, optimizer):
    result = optimizer.run(simulation_df=simulation_df)
    required = {"product_id", "store_id", "discount_pct", "net_margin", "campaign_cost"}
    assert required.issubset(result.campaign_grid.columns)


def test_zero_budget_returns_empty_grid():
    opt = CampaignOptimizer(total_budget=0.0)
    df = pd.DataFrame({
        "product_id": ["SKU-0001"],
        "store_id": ["LJ-001"],
        "year_month": ["2024-12"],
        "discount_pct": [0.10],
        "predicted_volume": [100],
        "unit_price": [10.0],
        "simulated_price": [9.0],
        "predicted_margin_pct": [0.30],
        "predicted_revenue": [900.0],
        "predicted_margin": [270.0],
    })
    result = opt.run(simulation_df=df)
    # Com budget 0, nenhuma campanha deve ser selecionada
    assert result.n_campaigns == 0 or result.total_budget_used == 0.0
