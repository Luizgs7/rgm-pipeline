"""Testes do DataQualityRunner — validação de schemas, nulos e regras de negócio."""

import numpy as np
import pandas as pd
import pytest

from rgm_pipeline.agents.data_engineer.data_quality import DataQualityRunner, DQReport


@pytest.fixture()
def runner() -> DataQualityRunner:
    return DataQualityRunner()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_pass(runner, sample_transactions):
    report = DQReport(table="transactions")
    from rgm_pipeline.agents.data_engineer.data_quality import _SchemaChecker
    _SchemaChecker().check(sample_transactions, "transactions", report)
    errors = [i for i in report.issues if i.severity == "error"]
    assert not errors


def test_schema_missing_column(runner):
    df = pd.DataFrame({"product_id": ["A"], "volume": [10]})  # faltam colunas
    report = DQReport(table="transactions")
    from rgm_pipeline.agents.data_engineer.data_quality import _SchemaChecker
    _SchemaChecker().check(df, "transactions", report)
    assert any(i.check == "schema_completeness" for i in report.issues)


# ---------------------------------------------------------------------------
# Nulos
# ---------------------------------------------------------------------------

def test_null_check_pass(runner, sample_transactions):
    report = DQReport(table="transactions")
    from rgm_pipeline.agents.data_engineer.data_quality import _NullChecker
    _NullChecker().check(sample_transactions, "transactions", report)
    errors = [i for i in report.issues if i.severity == "error"]
    assert not errors


def test_null_check_fails_above_threshold():
    df = pd.DataFrame({
        "volume": [None] * 100 + [1] * 10,
        "revenue": range(110),
    })
    report = DQReport(table="transactions")
    from rgm_pipeline.agents.data_engineer.data_quality import _NullChecker
    _NullChecker().check(df, "transactions", report)
    assert any(i.check == "null_rate" for i in report.issues)


# ---------------------------------------------------------------------------
# Anomalias (IQR)
# ---------------------------------------------------------------------------

def test_anomaly_detects_outlier():
    base = [10.0] * 200
    base.append(100_000.0)  # outlier extremo
    df = pd.DataFrame({"volume": base})
    report = DQReport(table="transactions")
    from rgm_pipeline.agents.data_engineer.data_quality import _AnomalyChecker
    _AnomalyChecker().check(df, "transactions", report)
    assert any(i.check == "iqr_anomaly" for i in report.issues)


# ---------------------------------------------------------------------------
# Regras de negócio
# ---------------------------------------------------------------------------

def test_business_rules_negative_volume():
    df = pd.DataFrame({
        "volume": [-1, 10, 5],
        "unit_price": [5.0, 5.0, 5.0],
        "revenue": [-5.0, 50.0, 25.0],
        "cost": [3.0, 30.0, 15.0],
        "margin": [-8.0, 20.0, 10.0],
        "margin_pct": [-1.6, 0.4, 0.4],
    })
    report = DQReport(table="transactions")
    from rgm_pipeline.agents.data_engineer.data_quality import _BusinessRulesChecker
    _BusinessRulesChecker().check(df, "transactions", report)
    assert any(i.check == "positive_volume" for i in report.issues)


def test_business_rules_revenue_consistency():
    df = pd.DataFrame({
        "volume": [10],
        "unit_price": [5.0],
        "revenue": [999.99],   # errado: deveria ser 50.0
        "cost": [30.0],
        "margin": [969.99],
        "margin_pct": [0.97],
    })
    report = DQReport(table="transactions")
    from rgm_pipeline.agents.data_engineer.data_quality import _BusinessRulesChecker
    _BusinessRulesChecker().check(df, "transactions", report)
    assert any(i.check == "revenue_consistency" for i in report.issues)


def test_business_rules_campaign_max_discount():
    df = pd.DataFrame({
        "campaign_id": ["C1"],
        "product_id": ["P1"],
        "store_id": ["S1"],
        "discount_pct": [0.99],   # acima do limite
        "start_date": ["2024-01-01"],
        "end_date": ["2024-01-31"],
        "budget": [1000.0],
        "status": ["encerrada"],
    })
    report = DQReport(table="campaigns")
    from rgm_pipeline.agents.data_engineer.data_quality import _BusinessRulesChecker
    _BusinessRulesChecker().check(df, "campaigns", report)
    assert any(i.check == "max_discount_rule" for i in report.issues)


def test_business_rules_campaign_date_order():
    df = pd.DataFrame({
        "campaign_id": ["C1"],
        "product_id": ["P1"],
        "store_id": ["S1"],
        "discount_pct": [0.10],
        "start_date": ["2024-02-01"],
        "end_date": ["2024-01-01"],   # fim antes do início
        "budget": [1000.0],
        "status": ["encerrada"],
    })
    report = DQReport(table="campaigns")
    from rgm_pipeline.agents.data_engineer.data_quality import _BusinessRulesChecker
    _BusinessRulesChecker().check(df, "campaigns", report)
    assert any(i.check == "date_order" for i in report.issues)


# ---------------------------------------------------------------------------
# Runner completo
# ---------------------------------------------------------------------------

def test_full_runner_returns_report_per_table(
    runner, sample_transactions, sample_campaigns
):
    datasets = {"transactions": sample_transactions, "campaigns": sample_campaigns}
    reports = runner.validate(datasets)
    assert set(reports.keys()) == {"transactions", "campaigns"}
    for name, report in reports.items():
        assert isinstance(report, DQReport)
        assert report.table == name
