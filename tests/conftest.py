"""Fixtures compartilhadas entre todos os testes do RGM Pipeline."""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.data_engineer.mock_generator import MockDataGenerator


# ---------------------------------------------------------------------------
# Fixtures de dados pequenos (em memória, independentes do SQLite)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def small_generator() -> MockDataGenerator:
    """Gerador com seed fixo e tamanho reduzido."""
    import config.settings as s
    # Sobrescreve config temporariamente para testes rápidos
    original = s.MOCK_CONFIG.copy()
    s.MOCK_CONFIG.update({"n_products": 5, "n_stores": 3, "n_campaigns": 10})
    gen = MockDataGenerator(seed=0)
    s.MOCK_CONFIG.update(original)
    return gen


@pytest.fixture(scope="session")
def sample_products() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 5
    cost = np.round(rng.uniform(5.0, 30.0, n), 2)
    return pd.DataFrame({
        "product_id": [f"SKU-{i:04d}" for i in range(1, n + 1)],
        "product_name": [f"Produto {i}" for i in range(1, n + 1)],
        "category": rng.choice(["Bebidas", "Snacks"], size=n),
        "unit_cost": cost,
        "base_price": np.round(cost * 1.5, 2),
    })


@pytest.fixture(scope="session")
def sample_stores() -> pd.DataFrame:
    return pd.DataFrame({
        "store_id": ["LJ-001", "LJ-002", "LJ-003"],
        "store_name": ["Loja 001", "Loja 002", "Loja 003"],
        "region": ["Sudeste", "Sul", "Nordeste"],
        "store_size": ["Médio", "Pequeno", "Grande"],
    })


@pytest.fixture(scope="session")
def sample_campaigns(sample_products, sample_stores) -> pd.DataFrame:
    return pd.DataFrame({
        "campaign_id": ["CAMP-0001", "CAMP-0002", "CAMP-0003"],
        "product_id": ["SKU-0001", "SKU-0002", "SKU-0003"],
        "store_id": ["LJ-001", "LJ-002", "LJ-001"],
        "discount_pct": [0.10, 0.20, 0.30],
        "start_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]).date,
        "end_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]).date,
        "budget": [10_000.0, 20_000.0, 15_000.0],
        "status": ["encerrada", "encerrada", "encerrada"],
    })


@pytest.fixture(scope="session")
def sample_transactions(sample_products, sample_stores) -> pd.DataFrame:
    """Transações sintéticas simples para testes de DQ, forecasting e optimizer."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
    records = []
    for pid in sample_products["product_id"]:
        for sid in sample_stores["store_id"]:
            base_price = float(
                sample_products.loc[sample_products["product_id"] == pid, "base_price"].iloc[0]
            )
            unit_cost = float(
                sample_products.loc[sample_products["product_id"] == pid, "unit_cost"].iloc[0]
            )
            for i, date in enumerate(dates):
                volume = int(rng.integers(10, 100))
                discount = 0.10 if (i % 30 < 7) else 0.0
                unit_price = round(base_price * (1 - discount), 2)
                revenue = round(volume * unit_price, 2)
                cost = round(volume * unit_cost, 2)
                margin = round(revenue - cost, 2)
                records.append({
                    "transaction_id": f"TXN-{len(records):08d}",
                    "date": date,
                    "product_id": pid,
                    "store_id": sid,
                    "campaign_id": "CAMP-0001" if discount > 0 else None,
                    "discount_pct": discount,
                    "volume": volume,
                    "unit_price": unit_price,
                    "revenue": revenue,
                    "cost": cost,
                    "margin": margin,
                    "margin_pct": round(margin / revenue if revenue else 0.0, 4),
                })
    return pd.DataFrame(records)


@pytest.fixture(scope="session")
def sample_uplift_metrics() -> pd.DataFrame:
    return pd.DataFrame({
        "uplift_id": ["UPL-CAMP-0001", "UPL-CAMP-0002"],
        "campaign_id": ["CAMP-0001", "CAMP-0002"],
        "incremental_volume": [500, 300],
        "incremental_revenue": [5000.0, 2500.0],
        "incremental_margin": [1500.0, 700.0],
        "roi": [0.15, 0.07],
        "confidence_score": [0.85, 0.72],
    })


# ---------------------------------------------------------------------------
# Fixture de banco SQLite temporário
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def temp_db(
    sample_products,
    sample_stores,
    sample_campaigns,
    sample_transactions,
    sample_uplift_metrics,
    tmp_path_factory,
) -> Path:
    """Cria SQLite temporário populado com as fixtures de sessão."""
    db_path = tmp_path_factory.mktemp("data") / "test_rgm.db"
    with sqlite3.connect(db_path) as conn:
        sample_products.to_sql("products", conn, if_exists="replace", index=False)
        sample_stores.to_sql("stores", conn, if_exists="replace", index=False)
        sample_campaigns.to_sql("campaigns", conn, if_exists="replace", index=False)
        sample_transactions.to_sql("transactions", conn, if_exists="replace", index=False)
        sample_uplift_metrics.to_sql("uplift_metrics", conn, if_exists="replace", index=False)
    return db_path


@pytest.fixture()
def patch_db_path(temp_db, monkeypatch):
    """Redireciona DB_PATH global para o banco de teste."""
    import config.settings as s
    monkeypatch.setattr(s, "DB_PATH", temp_db)
    # Também patcha nos módulos que já importaram DB_PATH
    import agents.data_scientist.causal_baseline as cb
    import agents.data_scientist.demand_forecasting as df_mod
    import agents.ml_engineer.drift_monitor as dm
    monkeypatch.setattr(cb, "DB_PATH", temp_db)
    monkeypatch.setattr(df_mod, "DB_PATH", temp_db)
    monkeypatch.setattr(dm, "DB_PATH", temp_db)
    return temp_db
