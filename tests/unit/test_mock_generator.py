"""Testes do MockDataGenerator — geração de dados mock."""

import numpy as np
import pandas as pd
import pytest

import rgm_pipeline.config.settings as s
from rgm_pipeline.agents.data_engineer.mock_generator import MockDataGenerator


@pytest.fixture(scope="module")
def gen_data(tmp_path_factory):
    """Gera dataset completo pequeno uma única vez por módulo de teste."""
    db = tmp_path_factory.mktemp("mock") / "test.db"
    # Configura tamanho mínimo para o teste
    orig = s.MOCK_CONFIG.copy()
    s.MOCK_CONFIG.update({"n_products": 4, "n_stores": 3, "n_campaigns": 8})
    s.RAW_DIR.mkdir(parents=True, exist_ok=True)

    orig_db = s.DB_PATH
    import rgm_pipeline.config.settings as cfg
    cfg.DB_PATH = db

    gen = MockDataGenerator(seed=7)
    data = gen.run()

    s.MOCK_CONFIG.update(orig)
    cfg.DB_PATH = orig_db
    return data


def test_products_shape(gen_data):
    df = gen_data["products"]
    assert len(df) == 4
    assert {"product_id", "category", "unit_cost", "base_price"}.issubset(df.columns)


def test_products_price_above_cost(gen_data):
    df = gen_data["products"]
    assert (df["base_price"] > df["unit_cost"]).all()


def test_stores_shape(gen_data):
    df = gen_data["stores"]
    assert len(df) == 3
    assert df["store_size"].isin(["Pequeno", "Médio", "Grande"]).all()


def test_campaigns_shape(gen_data):
    df = gen_data["campaigns"]
    assert len(df) == 8
    assert (df["discount_pct"] > 0).all()
    assert (df["budget"] > 0).all()


def test_campaign_dates_order(gen_data):
    df = gen_data["campaigns"].copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    assert (df["start_date"] < df["end_date"]).all()


def test_transactions_volume_positive(gen_data):
    assert (gen_data["transactions"]["volume"] > 0).all()


def test_transactions_revenue_consistency(gen_data):
    txn = gen_data["transactions"].copy()
    expected = (txn["volume"] * txn["unit_price"]).round(2)
    diff = (txn["revenue"] - expected).abs()
    assert (diff <= 0.02).all(), "Receita inconsistente com volume * preço"


def test_transactions_margin_consistency(gen_data):
    txn = gen_data["transactions"].copy()
    expected_margin = (txn["revenue"] - txn["cost"]).round(2)
    diff = (txn["margin"] - expected_margin).abs()
    assert (diff <= 0.02).all()


def test_campaign_effect_increases_volume(gen_data):
    """Transações em campanhas devem ter volume médio maior do que sem campanha."""
    txn = gen_data["transactions"]
    with_camp = txn[txn["campaign_id"].notna()]["volume"].mean()
    without_camp = txn[txn["campaign_id"].isna()]["volume"].mean()
    assert with_camp > without_camp, "Campanhas devem aumentar volume médio"


def test_transactions_no_negative_revenue(gen_data):
    assert (gen_data["transactions"]["revenue"] >= 0).all()


def test_uplift_metrics_campaign_ids_match(gen_data):
    camp_ids = set(gen_data["campaigns"]["campaign_id"])
    uplift_ids = set(gen_data["uplift_metrics"]["campaign_id"])
    assert uplift_ids.issubset(camp_ids)


def test_reproducibility():
    """Duas execuções com mesmo seed devem gerar dados idênticos."""
    orig = s.MOCK_CONFIG.copy()
    s.MOCK_CONFIG.update({"n_products": 3, "n_stores": 2, "n_campaigns": 4})

    g1 = MockDataGenerator(seed=99)
    g2 = MockDataGenerator(seed=99)
    p1 = g1.generate_products()
    p2 = g2.generate_products()

    s.MOCK_CONFIG.update(orig)
    pd.testing.assert_frame_equal(p1, p2)
