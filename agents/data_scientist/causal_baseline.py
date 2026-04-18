"""
[Agente Cientista de Dados] — Módulo 2: Baseline Causal (Contrafactual)

Estima o que teria acontecido com vendas e margem nos períodos de promoção
caso as campanhas não tivessem ocorrido, usando Difference-in-Differences (DiD)
e regressão com variável de tratamento.
"""

import sqlite3
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from loguru import logger

from config.settings import DB_PATH, PROCESSED_DIR


@dataclass
class DiDResult:
    """Resultado do estimador Difference-in-Differences."""
    att: float              # Average Treatment Effect on the Treated
    att_revenue: float      # ATT em receita
    att_margin: float       # ATT em margem
    n_treated: int
    n_control: int
    baseline_df: pd.DataFrame   # Contrafactuais por transação/período


class CausalBaselineEstimator:
    """
    Estima o baseline contrafactual via DiD + regressão linear com FE.

    Estratégia:
      1. Identifica unidades tratadas (produto-loja com campanha) e controles
         (mesmo produto, outra loja sem campanha no mesmo período).
      2. Aplica estimador DiD: Δ = (pós_trat - pré_trat) - (pós_ctrl - pré_ctrl)
      3. Treina modelo de regressão para projetar contrafactual granular.
    """

    def __init__(self) -> None:
        self._le_product = LabelEncoder()
        self._le_store = LabelEncoder()
        self._reg_model = LinearRegression()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Carregamento de dados
    # ------------------------------------------------------------------

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega transações e campanhas do SQLite."""
        logger.info("[Causal] Carregando dados do banco...")
        with sqlite3.connect(DB_PATH) as conn:
            txn = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])
            camp = pd.read_sql("SELECT * FROM campaigns", conn,
                               parse_dates=["start_date", "end_date"])
        logger.info(f"[Causal] Transações: {len(txn):,} | Campanhas: {len(camp):,}")
        return txn, camp

    # ------------------------------------------------------------------
    # Preparação do painel DiD
    # ------------------------------------------------------------------

    def _build_panel(
        self,
        txn: pd.DataFrame,
        camp: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Constrói painel mensal produto-loja com flag de tratamento.

        Cada linha = (produto, loja, ano-mês) com:
          - treated: 1 se houve campanha ativa naquele mês
          - post: 1 se o mês está dentro do período da campanha
        """
        # Agrega transações para nível mensal
        txn = txn.copy()
        txn["year_month"] = txn["date"].dt.to_period("M")

        panel = (
            txn.groupby(["product_id", "store_id", "year_month"])
            .agg(
                volume=("volume", "sum"),
                revenue=("revenue", "sum"),
                margin=("margin", "sum"),
            )
            .reset_index()
        )

        # Conjunto de (produto, loja) tratados
        treated_pairs = set(
            zip(camp["product_id"], camp["store_id"])
        )

        panel["treated"] = panel.apply(
            lambda r: int((r["product_id"], r["store_id"]) in treated_pairs),
            axis=1,
        )

        # Período "post": mês dentro de alguma campanha ativa desse produto-loja
        camp_index = self._build_camp_index(camp)
        panel["post"] = panel.apply(
            lambda r: self._is_post(r, camp_index), axis=1
        )

        panel["did_group"] = panel["treated"].astype(str) + "_" + panel["post"].astype(str)
        return panel

    @staticmethod
    def _build_camp_index(camp: pd.DataFrame) -> dict:
        index: dict = {}
        for row in camp.itertuples(index=False):
            key = (row.product_id, row.store_id)
            index.setdefault(key, []).append(
                (pd.Period(row.start_date, "M"), pd.Period(row.end_date, "M"))
            )
        return index

    @staticmethod
    def _is_post(row: pd.Series, index: dict) -> int:
        key = (row["product_id"], row["store_id"])
        for start, end in index.get(key, []):
            if start <= row["year_month"] <= end:
                return 1
        return 0

    # ------------------------------------------------------------------
    # Estimador DiD
    # ------------------------------------------------------------------

    def _estimate_did(self, panel: pd.DataFrame) -> dict[str, float]:
        """Calcula ATT via estimador DiD 2x2 clássico."""
        means = panel.groupby(["treated", "post"])["volume"].mean()

        # DiD = (treated_post - treated_pre) - (control_post - control_pre)
        try:
            att = (
                (means.loc[(1, 1)] - means.loc[(1, 0)])
                - (means.loc[(0, 1)] - means.loc[(0, 0)])
            )
        except KeyError:
            att = 0.0

        means_rev = panel.groupby(["treated", "post"])["revenue"].mean()
        means_mar = panel.groupby(["treated", "post"])["margin"].mean()

        att_rev = (
            (means_rev.loc[(1, 1)] - means_rev.loc[(1, 0)])
            - (means_rev.loc[(0, 1)] - means_rev.loc[(0, 0)])
        ) if all(k in means_rev.index for k in [(1,1),(1,0),(0,1),(0,0)]) else 0.0

        att_mar = (
            (means_mar.loc[(1, 1)] - means_mar.loc[(1, 0)])
            - (means_mar.loc[(0, 1)] - means_mar.loc[(0, 0)])
        ) if all(k in means_mar.index for k in [(1,1),(1,0),(0,1),(0,0)]) else 0.0

        return {"att_volume": att, "att_revenue": att_rev, "att_margin": att_mar}

    # ------------------------------------------------------------------
    # Modelo de regressão para contrafactual granular
    # ------------------------------------------------------------------

    def _fit_counterfactual_model(self, panel: pd.DataFrame) -> None:
        """
        Treina LinearRegression com FE de produto e loja para projetar
        o volume contrafactual (sem tratamento).
        """
        control = panel[panel["treated"] == 0].copy()

        control["product_enc"] = self._le_product.fit_transform(control["product_id"])
        control["store_enc"] = self._le_store.fit_transform(control["store_id"])
        control["month_num"] = control["year_month"].apply(lambda p: p.month)
        control["year_num"] = control["year_month"].apply(lambda p: p.year)

        features = ["product_enc", "store_enc", "month_num", "year_num"]
        self._reg_model.fit(control[features], control["volume"])
        logger.info("[Causal] Modelo contrafactual treinado no grupo controle.")

    def _predict_counterfactual(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Projeta volume contrafactual para todas as unidades."""
        df = panel.copy()

        df["product_enc"] = df["product_id"].apply(
            lambda x: self._le_product.transform([x])[0]
            if x in self._le_product.classes_ else -1
        )
        df["store_enc"] = df["store_id"].apply(
            lambda x: self._le_store.transform([x])[0]
            if x in self._le_store.classes_ else -1
        )
        df["month_num"] = df["year_month"].apply(lambda p: p.month)
        df["year_num"] = df["year_month"].apply(lambda p: p.year)

        known_mask = (df["product_enc"] >= 0) & (df["store_enc"] >= 0)
        features = ["product_enc", "store_enc", "month_num", "year_num"]

        df["volume_counterfactual"] = np.nan
        df.loc[known_mask, "volume_counterfactual"] = self._reg_model.predict(
            df.loc[known_mask, features]
        )

        # Para unidades de controle fora do encoding, usa volume observado como proxy
        df["volume_counterfactual"] = df["volume_counterfactual"].fillna(df["volume"])
        df["volume_counterfactual"] = df["volume_counterfactual"].clip(lower=0)

        # Uplift incremental estimado
        df["incremental_volume"] = df["volume"] - df["volume_counterfactual"]

        return df

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def run(self) -> DiDResult:
        """Executa o pipeline completo de estimação causal."""
        logger.info("=== [Agente Cientista de Dados] Baseline Causal ===")

        txn, camp = self.load_data()
        panel = self._build_panel(txn, camp)

        did_stats = self._estimate_did(panel)
        logger.info(
            f"[DiD] ATT volume: {did_stats['att_volume']:.1f} unidades/mês | "
            f"ATT receita: R$ {did_stats['att_revenue']:.2f} | "
            f"ATT margem: R$ {did_stats['att_margin']:.2f}"
        )

        self._fit_counterfactual_model(panel)
        baseline_df = self._predict_counterfactual(panel)

        # Salva resultado
        out_path = PROCESSED_DIR / "causal_baseline.parquet"
        baseline_df.to_parquet(out_path, index=False)
        logger.success(f"[Causal] Baseline salvo em {out_path}")

        return DiDResult(
            att=did_stats["att_volume"],
            att_revenue=did_stats["att_revenue"],
            att_margin=did_stats["att_margin"],
            n_treated=int(panel[panel["treated"] == 1]["product_id"].nunique()),
            n_control=int(panel[panel["treated"] == 0]["product_id"].nunique()),
            baseline_df=baseline_df,
        )
