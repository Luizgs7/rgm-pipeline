"""
[Agente Engenheiro de Dados] — Módulo 1: Data Quality

Valida schemas, detecta nulos, identifica anomalias estatísticas e
aplica regras de negócio sobre os datasets do RGM Pipeline.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from loguru import logger

from rgm_pipeline.config.settings import DQ_CONFIG


# ---------------------------------------------------------------------------
# Estruturas de resultado
# ---------------------------------------------------------------------------

@dataclass
class DQIssue:
    """Representa um problema encontrado durante a validação."""
    severity: str          # "error" | "warning"
    table: str
    column: str
    check: str
    detail: str
    affected_rows: int = 0


@dataclass
class DQReport:
    """Relatório consolidado de qualidade de dados."""
    table: str
    passed: bool = True
    issues: list[DQIssue] = field(default_factory=list)

    def add_issue(self, issue: DQIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "error":
            self.passed = False

    def summary(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "passed": self.passed,
            "errors": sum(1 for i in self.issues if i.severity == "error"),
            "warnings": sum(1 for i in self.issues if i.severity == "warning"),
            "issues": [vars(i) for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Schemas esperados por tabela
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, dict[str, type]] = {
    "products": {
        "product_id": str,
        "product_name": str,
        "category": str,
        "unit_cost": float,
        "base_price": float,
    },
    "stores": {
        "store_id": str,
        "store_name": str,
        "region": str,
        "store_size": str,
    },
    "campaigns": {
        "campaign_id": str,
        "product_id": str,
        "store_id": str,
        "discount_pct": float,
        "start_date": object,
        "end_date": object,
        "budget": float,
        "status": str,
    },
    "transactions": {
        "transaction_id": str,
        "date": object,
        "product_id": str,
        "store_id": str,
        "volume": int,
        "unit_price": float,
        "revenue": float,
        "cost": float,
        "margin": float,
        "margin_pct": float,
    },
    "uplift_metrics": {
        "uplift_id": str,
        "campaign_id": str,
        "incremental_volume": int,
        "incremental_revenue": float,
        "incremental_margin": float,
        "roi": float,
        "confidence_score": float,
    },
}


# ---------------------------------------------------------------------------
# Verificadores individuais
# ---------------------------------------------------------------------------

class _SchemaChecker:
    """Valida se as colunas obrigatórias estão presentes."""

    def check(self, df: pd.DataFrame, table: str, report: DQReport) -> None:
        expected = set(SCHEMAS.get(table, {}).keys())
        actual = set(df.columns)
        missing = expected - actual

        if missing:
            report.add_issue(DQIssue(
                severity="error",
                table=table,
                column=str(missing),
                check="schema_completeness",
                detail=f"Colunas ausentes: {missing}",
            ))
        else:
            logger.debug(f"[{table}] Schema OK — {len(actual)} colunas presentes.")


# Colunas que são nullable por design (ex: FK opcional)
_NULLABLE_COLUMNS: dict[str, set[str]] = {
    "transactions": {"campaign_id"},
}


class _NullChecker:
    """Detecta colunas com percentual de nulos acima do limiar, exceto opcionais."""

    def check(self, df: pd.DataFrame, table: str, report: DQReport) -> None:
        threshold = DQ_CONFIG["max_null_pct"]
        nullable = _NULLABLE_COLUMNS.get(table, set())

        for col in df.columns:
            if col in nullable:
                continue
            null_pct = df[col].isna().mean()
            if null_pct > threshold:
                report.add_issue(DQIssue(
                    severity="error",
                    table=table,
                    column=col,
                    check="null_rate",
                    detail=f"{null_pct:.1%} de nulos (limiar: {threshold:.1%})",
                    affected_rows=int(df[col].isna().sum()),
                ))


class _AnomalyChecker:
    """Detecta outliers numéricos via IQR."""

    def check(self, df: pd.DataFrame, table: str, report: DQReport) -> None:
        mult = DQ_CONFIG["iqr_multiplier"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue

            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - mult * iqr, q3 + mult * iqr

            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if not outliers.empty:
                report.add_issue(DQIssue(
                    severity="warning",
                    table=table,
                    column=col,
                    check="iqr_anomaly",
                    detail=f"{len(outliers)} outliers fora de [{lower:.2f}, {upper:.2f}]",
                    affected_rows=len(outliers),
                ))


class _BusinessRulesChecker:
    """Valida regras de negócio específicas do domínio RGM."""

    def check(self, df: pd.DataFrame, table: str, report: DQReport) -> None:
        if table == "transactions":
            self._check_transactions(df, report)
        elif table == "campaigns":
            self._check_campaigns(df, report)
        elif table == "uplift_metrics":
            self._check_uplift(df, report)

    @staticmethod
    def _check_transactions(df: pd.DataFrame, report: DQReport) -> None:
        # Margem não pode ser inferior ao limite mínimo
        min_margin = DQ_CONFIG["min_margin_pct"]
        bad = df[df["margin_pct"] < min_margin]
        if not bad.empty:
            report.add_issue(DQIssue(
                severity="warning",
                table="transactions",
                column="margin_pct",
                check="min_margin_rule",
                detail=f"{len(bad)} transações com margem abaixo de {min_margin:.0%}",
                affected_rows=len(bad),
            ))

        # Volume deve ser positivo
        neg_vol = df[df["volume"] <= 0]
        if not neg_vol.empty:
            report.add_issue(DQIssue(
                severity="error",
                table="transactions",
                column="volume",
                check="positive_volume",
                detail=f"{len(neg_vol)} transações com volume não positivo",
                affected_rows=len(neg_vol),
            ))

        # Receita = volume * unit_price (tolerância 1 centavo)
        df = df.copy()
        df["expected_revenue"] = df["volume"] * df["unit_price"]
        revenue_mismatch = df[abs(df["revenue"] - df["expected_revenue"]) > 0.02]
        if not revenue_mismatch.empty:
            report.add_issue(DQIssue(
                severity="error",
                table="transactions",
                column="revenue",
                check="revenue_consistency",
                detail=f"{len(revenue_mismatch)} registros com receita inconsistente",
                affected_rows=len(revenue_mismatch),
            ))

    @staticmethod
    def _check_campaigns(df: pd.DataFrame, report: DQReport) -> None:
        max_discount = DQ_CONFIG["max_discount_pct"]
        over_discount = df[df["discount_pct"] > max_discount]
        if not over_discount.empty:
            report.add_issue(DQIssue(
                severity="error",
                table="campaigns",
                column="discount_pct",
                check="max_discount_rule",
                detail=f"{len(over_discount)} campanhas com desconto acima de {max_discount:.0%}",
                affected_rows=len(over_discount),
            ))

        # Data de início deve ser anterior à data de fim
        df = df.copy()
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        bad_dates = df[df["start_date"] >= df["end_date"]]
        if not bad_dates.empty:
            report.add_issue(DQIssue(
                severity="error",
                table="campaigns",
                column="start_date/end_date",
                check="date_order",
                detail=f"{len(bad_dates)} campanhas com datas inválidas (início >= fim)",
                affected_rows=len(bad_dates),
            ))

    @staticmethod
    def _check_uplift(df: pd.DataFrame, report: DQReport) -> None:
        # Score de confiança entre 0 e 1
        bad_score = df[(df["confidence_score"] < 0) | (df["confidence_score"] > 1)]
        if not bad_score.empty:
            report.add_issue(DQIssue(
                severity="error",
                table="uplift_metrics",
                column="confidence_score",
                check="score_range",
                detail=f"{len(bad_score)} registros com confidence_score fora de [0, 1]",
                affected_rows=len(bad_score),
            ))


# ---------------------------------------------------------------------------
# Orquestrador de DQ
# ---------------------------------------------------------------------------

class DataQualityRunner:
    """Executa todas as verificações de qualidade sobre um conjunto de DataFrames."""

    def __init__(self) -> None:
        self._checkers = [
            _SchemaChecker(),
            _NullChecker(),
            _AnomalyChecker(),
            _BusinessRulesChecker(),
        ]

    def validate(self, datasets: dict[str, pd.DataFrame]) -> dict[str, DQReport]:
        """Valida todos os datasets e retorna relatórios por tabela."""
        reports: dict[str, DQReport] = {}

        for table_name, df in datasets.items():
            logger.info(f"[DQ] Validando tabela: {table_name} ({len(df):,} registros)")
            report = DQReport(table=table_name)

            for checker in self._checkers:
                checker.check(df, table_name, report)

            status = "PASSOU" if report.passed else "FALHOU"
            logger.info(f"[DQ] {table_name} → {status} | "
                        f"Erros: {sum(1 for i in report.issues if i.severity == 'error')} | "
                        f"Avisos: {sum(1 for i in report.issues if i.severity == 'warning')}")

            reports[table_name] = report

        return reports

    def print_full_report(self, reports: dict[str, DQReport]) -> None:
        """Imprime relatório detalhado de DQ no console."""
        print("\n" + "=" * 60)
        print("RELATÓRIO DE DATA QUALITY — RGM Pipeline")
        print("=" * 60)

        for table, report in reports.items():
            status_str = "✓ PASSOU" if report.passed else "✗ FALHOU"
            print(f"\n[{table}] {status_str}")

            if not report.issues:
                print("  Nenhum problema encontrado.")
                continue

            for issue in report.issues:
                icon = "ERROR" if issue.severity == "error" else "WARN "
                print(f"  [{icon}] {issue.check} | coluna: {issue.column}")
                print(f"           {issue.detail}")
                if issue.affected_rows:
                    print(f"           Linhas afetadas: {issue.affected_rows:,}")

        print("\n" + "=" * 60)
