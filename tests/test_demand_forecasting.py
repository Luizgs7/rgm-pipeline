"""Testes do DemandForecaster — feature engineering, treino e simulação."""

import numpy as np
import pandas as pd
import pytest

from agents.data_scientist.demand_forecasting import DemandForecaster


@pytest.fixture(scope="session")
def trained_forecaster(sample_transactions, tmp_path_factory):
    """Treina o forecaster uma vez na sessão usando as transações de fixture."""
    processed = tmp_path_factory.mktemp("forecast_processed")
    import agents.data_scientist.demand_forecasting as mod
    original = mod.PROCESSED_DIR
    mod.PROCESSED_DIR = processed

    f = DemandForecaster()
    df = f._engineer_features(sample_transactions, fit=True)
    f.train(df)

    mod.PROCESSED_DIR = original
    return f, df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def test_engineer_features_adds_lag_columns(sample_transactions):
    f = DemandForecaster()
    df = f._engineer_features(sample_transactions, fit=True)
    for col in ["lag_7d", "lag_30d", "ma_7d", "ma_30d"]:
        assert col in df.columns, f"Coluna {col} ausente"


def test_engineer_features_no_nan_in_feature_cols(sample_transactions):
    f = DemandForecaster()
    df = f._engineer_features(sample_transactions, fit=True)
    feature_cols = [
        "product_enc", "store_enc", "month", "dayofweek",
        "discount_pct", "lag_7d", "lag_30d", "ma_7d",
    ]
    for col in feature_cols:
        assert df[col].notna().all(), f"NaN encontrado em {col}"


def test_encode_inference_does_not_refit(sample_transactions):
    """Encoders não devem ser refitados na inferência."""
    f = DemandForecaster()
    f._engineer_features(sample_transactions, fit=True)
    original_classes = list(f._le_product.classes_)

    # Inferência com subconjunto de produtos
    subset = sample_transactions[sample_transactions["product_id"] == "SKU-0001"].copy()
    f._engineer_features(subset, fit=False)

    assert list(f._le_product.classes_) == original_classes


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def test_model_trained(trained_forecaster):
    f, _ = trained_forecaster
    assert f._model is not None
    assert f._fitted is True


def test_mape_reasonable(trained_forecaster):
    f, _ = trained_forecaster
    assert 0.0 < f._avg_mape < 2.0, f"MAPE fora do esperado: {f._avg_mape}"


def test_feature_importance_returns_dataframe(trained_forecaster):
    f, _ = trained_forecaster
    imp = f.get_feature_importance()
    assert isinstance(imp, pd.DataFrame)
    assert "feature" in imp.columns
    assert "importance_gain" in imp.columns
    assert len(imp) > 0


# ---------------------------------------------------------------------------
# Simulação de cenários
# ---------------------------------------------------------------------------

def test_simulation_covers_all_discount_scenarios(trained_forecaster):
    f, df = trained_forecaster
    sim = f.simulate_scenarios(df)
    expected = set(f.DISCOUNT_SCENARIOS)
    actual = set(sim["discount_pct"].unique())
    assert expected == actual


def test_simulation_predicted_volume_non_negative(trained_forecaster):
    f, df = trained_forecaster
    sim = f.simulate_scenarios(df)
    assert (sim["predicted_volume"] >= 0).all()


def test_simulation_higher_discount_lower_margin(trained_forecaster):
    """Descontos maiores devem resultar em margem percentual menor."""
    f, df = trained_forecaster
    sim = f.simulate_scenarios(df)
    avg_margin = sim.groupby("discount_pct")["predicted_margin_pct"].mean().sort_index()
    # Margem deve decrescer conforme desconto aumenta
    assert avg_margin.is_monotonic_decreasing, (
        "Margem média não decresce com o desconto"
    )


def test_simulation_columns_present(trained_forecaster):
    f, df = trained_forecaster
    sim = f.simulate_scenarios(df)
    required = {
        "product_id", "store_id", "discount_pct",
        "predicted_volume", "predicted_revenue",
        "predicted_margin", "predicted_margin_pct",
    }
    assert required.issubset(sim.columns)
