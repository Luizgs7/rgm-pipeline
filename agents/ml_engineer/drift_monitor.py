"""
[Agente Engenheiro de ML] — Módulo 4: Monitoramento de Drift

Detecta drift de dados e de modelo usando Evidently:
  - Data Drift: compara distribuição de features entre janela de referência
    e janela corrente usando testes estatísticos (KS, chi-quadrado).
  - Model Drift: monitora a distribuição das predições de volume ao longo do tempo.

Gera relatórios HTML para visualização e emite alertas via log.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import DB_PATH, PROCESSED_DIR

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, RegressionPreset
    from evidently.report import Report
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False
    logger.warning("[Drift] Evidently não instalado. Usando fallback estatístico.")


# ---------------------------------------------------------------------------
# Estrutura de resultado
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Resultado do monitoramento de drift."""
    check_type: str
    data_drift_detected: Optional[bool] = None
    model_drift_detected: Optional[bool] = None
    drifted_features: list[str] = field(default_factory=list)
    drift_score: Optional[float] = None
    recommendation: str = ""
    report_path: Optional[Path] = None
    checked_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Monitor principal
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Monitora data drift e model drift para o RGM Pipeline.

    Usa Evidently quando disponível; fallback para testes KS/chi² via scipy.
    """

    NUMERIC_FEATURES = ["volume", "revenue", "margin", "margin_pct", "discount_pct"]
    REPORT_DIR = PROCESSED_DIR / "drift_reports"

    def __init__(self) -> None:
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Carregamento de janelas de dados
    # ------------------------------------------------------------------

    def _load_windows(
        self,
        n_reference_days: int,
        n_current_days: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carrega janela de referência (histórico) e janela corrente (recente)
        a partir das transações do SQLite.
        """
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(
                "SELECT * FROM transactions", conn, parse_dates=["date"]
            )

        max_date = df["date"].max()
        current_start = max_date - timedelta(days=n_current_days)
        reference_end = current_start - timedelta(days=1)
        reference_start = reference_end - timedelta(days=n_reference_days)

        reference = df[
            (df["date"] >= reference_start) & (df["date"] <= reference_end)
        ].copy()

        current = df[df["date"] > current_start].copy()

        logger.info(
            f"[Drift] Referência: {len(reference):,} registros "
            f"({reference_start.date()} → {reference_end.date()})"
        )
        logger.info(
            f"[Drift] Corrente:   {len(current):,} registros "
            f"({current_start.date()} → {max_date.date()})"
        )

        return reference, current

    # ------------------------------------------------------------------
    # Data Drift
    # ------------------------------------------------------------------

    def check_data_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> tuple[bool, list[str], float]:
        """
        Verifica data drift nas features numéricas.

        Returns:
            (drift_detected, drifted_features, overall_drift_score)
        """
        if _EVIDENTLY_AVAILABLE:
            return self._data_drift_evidently(reference, current)
        return self._data_drift_fallback(reference, current)

    def _data_drift_evidently(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> tuple[bool, list[str], float]:
        """Drift via Evidently DataDriftPreset."""
        cols = [c for c in self.NUMERIC_FEATURES if c in reference.columns]

        report = Report(metrics=[DataDriftPreset(columns=cols)])
        report.run(reference_data=reference[cols], current_data=current[cols])

        result = report.as_dict()
        drift_by_col = result["metrics"][0]["result"]["drift_by_columns"]

        drifted = [
            col for col, info in drift_by_col.items()
            if info.get("drift_detected", False)
        ]
        n_drifted = len(drifted)
        score = round(n_drifted / max(len(cols), 1), 4)
        drift_detected = score > 0.30  # >30% das features com drift

        # Salva relatório HTML
        report_path = self.REPORT_DIR / f"data_drift_{datetime.utcnow():%Y%m%d_%H%M}.html"
        report.save_html(str(report_path))

        return drift_detected, drifted, score

    def _data_drift_fallback(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> tuple[bool, list[str], float]:
        """Fallback: teste KS por feature quando Evidently não está disponível."""
        from scipy import stats

        drifted: list[str] = []
        p_values: list[float] = []

        for col in self.NUMERIC_FEATURES:
            if col not in reference.columns or col not in current.columns:
                continue
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            _, p_value = stats.ks_2samp(ref_vals, cur_vals)
            p_values.append(p_value)

            if p_value < 0.05:
                drifted.append(col)
                logger.warning(f"[Drift] DATA DRIFT detectado em '{col}' (p={p_value:.4f})")

        score = round(len(drifted) / max(len(p_values), 1), 4)
        drift_detected = score > 0.30

        return drift_detected, drifted, score

    # ------------------------------------------------------------------
    # Model Drift (distribuição de predições)
    # ------------------------------------------------------------------

    def check_model_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> tuple[bool, float]:
        """
        Verifica model drift comparando a distribuição de 'volume' (proxy
        das predições) entre referência e corrente via KS test.

        Em produção, substituir 'volume' pela coluna de predições do modelo
        persistido em produção.
        """
        from scipy import stats

        ref_preds = reference["volume"].dropna().values
        cur_preds = current["volume"].dropna().values

        if len(ref_preds) < 10 or len(cur_preds) < 10:
            logger.warning("[Drift] Dados insuficientes para model drift check.")
            return False, 0.0

        stat, p_value = stats.ks_2samp(ref_preds, cur_preds)

        # Normaliza KS statistic (0 = idêntico, 1 = completamente diferente)
        drift_score = round(float(stat), 4)
        drift_detected = p_value < 0.05

        if drift_detected:
            logger.warning(
                f"[Drift] MODEL DRIFT detectado! KS stat={stat:.4f}, p={p_value:.4f}"
            )
        else:
            logger.info(f"[Drift] Model drift OK. KS stat={stat:.4f}, p={p_value:.4f}")

        return drift_detected, drift_score

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def run(
        self,
        check_type: str = "both",
        n_reference_days: int = 90,
        n_current_days: int = 30,
    ) -> DriftResult:
        """
        Executa verificação completa de drift.

        Args:
            check_type: "data", "model" ou "both"
            n_reference_days: tamanho da janela de referência em dias
            n_current_days: tamanho da janela corrente em dias
        """
        logger.info(f"=== [Drift Monitor] Verificando drift ({check_type}) ===")

        reference, current = self._load_windows(n_reference_days, n_current_days)

        if reference.empty or current.empty:
            return DriftResult(
                check_type=check_type,
                recommendation="Dados insuficientes para análise de drift.",
            )

        result = DriftResult(check_type=check_type)

        if check_type in ("data", "both"):
            data_drift, drifted_features, data_score = self.check_data_drift(
                reference, current
            )
            result.data_drift_detected = data_drift
            result.drifted_features = drifted_features
            result.drift_score = data_score

            if data_drift:
                logger.warning(
                    f"[Drift] DATA DRIFT em {len(drifted_features)} feature(s): "
                    f"{drifted_features}"
                )

        if check_type in ("model", "both"):
            model_drift, model_score = self.check_model_drift(reference, current)
            result.model_drift_detected = model_drift
            if result.drift_score is None:
                result.drift_score = model_score

        result.recommendation = self._build_recommendation(result)
        logger.info(f"[Drift] Recomendação: {result.recommendation}")

        return result

    @staticmethod
    def _build_recommendation(result: DriftResult) -> str:
        issues = []

        if result.data_drift_detected:
            issues.append(
                f"data drift em {len(result.drifted_features)} feature(s) "
                f"({', '.join(result.drifted_features[:3])})"
            )
        if result.model_drift_detected:
            issues.append("model drift na distribuição de predições")

        if not issues:
            return "Nenhum drift significativo detectado. Monitoramento normal."

        return (
            f"AÇÃO REQUERIDA: detectado {' e '.join(issues)}. "
            "Recomendam-se: (1) re-treino do modelo com dados recentes, "
            "(2) revisão das features com maior drift, "
            "(3) validação da grade de campanhas antes do próximo ciclo."
        )
