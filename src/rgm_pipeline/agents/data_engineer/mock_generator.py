"""
[Agente Engenheiro de Dados] — Módulo 1: Geração de Banco de Dados Mock

Gera transações históricas realistas, campanhas promocionais e métricas
de uplift (margem, receita, volume) para o domínio de RGM.

Estratégia vetorizada: pd.MultiIndex.from_product cria o produto cartesiano
(produto × loja × data) sem loops Python. Campanhas são aplicadas iterando
apenas sobre os ~80 registros de campanha com operações pandas mascaradas.
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from rgm_pipeline.config.settings import MOCK_CONFIG, RAW_DIR, DB_PATH


class MockDataGenerator:
    """Gera e persiste o banco de dados mock do RGM Pipeline."""

    def __init__(self, seed: int = MOCK_CONFIG["seed"]) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Tabelas de dimensão
    # ------------------------------------------------------------------

    def generate_products(self) -> pd.DataFrame:
        """Gera tabela de produtos com SKU, categoria e custo unitário."""
        n = MOCK_CONFIG["n_products"]
        unit_cost = np.round(self.rng.uniform(2.0, 50.0, size=n), 2)
        markup = self.rng.uniform(1.3, 2.5, size=n)

        return pd.DataFrame({
            "product_id": [f"SKU-{i:04d}" for i in range(1, n + 1)],
            "product_name": [f"Produto {i:04d}" for i in range(1, n + 1)],
            "category": self.rng.choice(MOCK_CONFIG["categories"], size=n),
            "unit_cost": unit_cost,
            "base_price": np.round(unit_cost * markup, 2),
        })

    def generate_stores(self) -> pd.DataFrame:
        """Gera tabela de lojas com região e porte."""
        n = MOCK_CONFIG["n_stores"]
        return pd.DataFrame({
            "store_id": [f"LJ-{i:03d}" for i in range(1, n + 1)],
            "store_name": [f"Loja {i:03d}" for i in range(1, n + 1)],
            "region": self.rng.choice(MOCK_CONFIG["regions"], size=n),
            "store_size": self.rng.choice(
                ["Pequeno", "Médio", "Grande"], size=n, p=[0.3, 0.5, 0.2]
            ),
        })

    # ------------------------------------------------------------------
    # Tabela de campanhas
    # ------------------------------------------------------------------

    def generate_campaigns(
        self, products: pd.DataFrame, stores: pd.DataFrame
    ) -> pd.DataFrame:
        """Gera campanhas promocionais com desconto, verba e período."""
        logger.info("Gerando campanhas promocionais...")
        n = MOCK_CONFIG["n_campaigns"]
        start_ts = pd.Timestamp(MOCK_CONFIG["history_start"])
        end_ts = pd.Timestamp(MOCK_CONFIG["history_end"])
        total_days = (end_ts - start_ts).days

        camp_starts = [
            start_ts + pd.Timedelta(days=int(d))
            for d in self.rng.integers(0, total_days - 30, size=n)
        ]
        durations = self.rng.integers(7, 45, size=n)
        camp_ends = [
            min(s + pd.Timedelta(days=int(d)), end_ts)
            for s, d in zip(camp_starts, durations)
        ]

        return pd.DataFrame({
            "campaign_id": [f"CAMP-{i:04d}" for i in range(1, n + 1)],
            "product_id": self.rng.choice(products["product_id"].values, size=n),
            "store_id": self.rng.choice(stores["store_id"].values, size=n),
            "discount_pct": [
                float(d) for d in self.rng.choice(MOCK_CONFIG["discount_levels"], size=n)
            ],
            "start_date": [s.date() for s in camp_starts],
            "end_date": [e.date() for e in camp_ends],
            "budget": np.round(self.rng.uniform(5_000, 200_000, size=n), 2),
            "status": "encerrada",
        })

    # ------------------------------------------------------------------
    # Tabela de transações históricas — geração vetorizada
    # ------------------------------------------------------------------

    def generate_transactions(
        self,
        products: pd.DataFrame,
        stores: pd.DataFrame,
        campaigns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Gera transações diárias vetorizadas via produto cartesiano
        (produto × loja × data).

        Efeitos promocionais são aplicados iterando sobre campanhas
        com mascaramento pandas — sem loops Python por linha.
        """
        logger.info("Gerando transações históricas (vetorizado)...")
        dates = pd.date_range(MOCK_CONFIG["history_start"], MOCK_CONFIG["history_end"])

        # ── Produto cartesiano ──────────────────────────────────────
        idx = pd.MultiIndex.from_product(
            [products["product_id"], stores["store_id"], dates],
            names=["product_id", "store_id", "date"],
        )
        df = idx.to_frame(index=False)

        # ── Atributos de produto e loja ─────────────────────────────
        df = df.merge(
            products[["product_id", "base_price", "unit_cost"]], on="product_id"
        )
        df = df.merge(stores[["store_id", "store_size"]], on="store_id")

        # ── Volume base por par produto-loja ────────────────────────
        pairs = df[["product_id", "store_id"]].drop_duplicates().copy()
        pairs["base_volume"] = self.rng.integers(10, 200, size=len(pairs))
        df = df.merge(pairs, on=["product_id", "store_id"])

        size_map = {"Pequeno": 0.6, "Médio": 1.0, "Grande": 1.8}
        df["store_factor"] = df["store_size"].map(size_map)

        # ── Sazonalidade vetorizada ─────────────────────────────────
        df["weekday_eff"] = 1.0 + 0.3 * (df["date"].dt.dayofweek >= 4)
        df["month_eff"] = 1.0 + 0.2 * (df["date"].dt.month == 12)
        df["noise"] = self.rng.uniform(0.85, 1.15, size=len(df))
        df["lam"] = (
            df["base_volume"] * df["store_factor"]
            * df["weekday_eff"] * df["month_eff"] * df["noise"]
        ).clip(lower=1.0)

        df["volume"] = self.rng.poisson(df["lam"].values).clip(1)
        df["discount_pct"] = 0.0
        df["campaign_id"] = pd.NA

        # ── Aplica efeito de campanhas ──────────────────────────────
        camps_ts = campaigns.copy()
        camps_ts["start_date"] = pd.to_datetime(camps_ts["start_date"])
        camps_ts["end_date"] = pd.to_datetime(camps_ts["end_date"])

        for camp in camps_ts.itertuples(index=False):
            mask = (
                (df["product_id"] == camp.product_id)
                & (df["store_id"] == camp.store_id)
                & (df["date"] >= camp.start_date)
                & (df["date"] <= camp.end_date)
            )
            if not mask.any():
                continue

            n_hit = mask.sum()
            df.loc[mask, "discount_pct"] = camp.discount_pct
            df.loc[mask, "campaign_id"] = camp.campaign_id

            uplift = 1.0 + camp.discount_pct * self.rng.uniform(1.5, 2.5, size=n_hit)
            df.loc[mask, "volume"] = (
                df.loc[mask, "volume"].values * uplift
            ).astype(int).clip(1)

        # ── Financeiro ──────────────────────────────────────────────
        df["unit_price"] = (df["base_price"] * (1.0 - df["discount_pct"])).round(2)
        df["revenue"] = (df["volume"] * df["unit_price"]).round(2)
        df["cost"] = (df["volume"] * df["unit_cost"]).round(2)
        df["margin"] = (df["revenue"] - df["cost"]).round(2)
        df["margin_pct"] = (
            df["margin"] / df["revenue"].replace(0.0, np.nan)
        ).fillna(0.0).round(4)

        df["transaction_id"] = [f"TXN-{i:08d}" for i in range(1, len(df) + 1)]

        # Remove colunas auxiliares
        df = df.drop(
            columns=["store_factor", "base_volume", "weekday_eff",
                     "month_eff", "noise", "lam"],
        )

        logger.info(f"Transações geradas: {len(df):,}")
        return df

    # ------------------------------------------------------------------
    # Tabela de métricas de uplift
    # ------------------------------------------------------------------

    def generate_uplift_metrics(
        self,
        campaigns: pd.DataFrame,
        transactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Agrega métricas de uplift por campanha comparando períodos com/sem promoção."""
        logger.info("Calculando métricas de uplift por campanha...")

        txn_camp = transactions[transactions["campaign_id"].notna()].copy()
        txn_base = transactions[transactions["campaign_id"].isna()].copy()

        records = []
        for camp in campaigns.itertuples(index=False):
            mask_c = txn_camp["campaign_id"] == camp.campaign_id
            camp_txn = txn_camp[mask_c]
            if camp_txn.empty:
                continue

            mask_b = (
                (txn_base["product_id"] == camp.product_id)
                & (txn_base["store_id"] == camp.store_id)
                & (pd.to_datetime(txn_base["date"]) >= pd.Timestamp(camp.start_date))
                & (pd.to_datetime(txn_base["date"]) <= pd.Timestamp(camp.end_date))
            )
            base_txn = txn_base[mask_b]

            inc_volume = int(camp_txn["volume"].sum() - base_txn["volume"].sum())
            inc_revenue = round(
                float(camp_txn["revenue"].sum() - base_txn["revenue"].sum()), 2
            )
            inc_margin = round(
                float(camp_txn["margin"].sum() - base_txn["margin"].sum()), 2
            )
            roi = round(inc_margin / camp.budget if camp.budget > 0 else 0.0, 4)

            records.append({
                "uplift_id": f"UPL-{camp.campaign_id}",
                "campaign_id": camp.campaign_id,
                "incremental_volume": inc_volume,
                "incremental_revenue": inc_revenue,
                "incremental_margin": inc_margin,
                "roi": roi,
                "confidence_score": round(float(self.rng.uniform(0.6, 0.99)), 2),
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Persistência no SQLite
    # ------------------------------------------------------------------

    def save_to_database(
        self,
        products: pd.DataFrame,
        stores: pd.DataFrame,
        campaigns: pd.DataFrame,
        transactions: pd.DataFrame,
        uplift_metrics: pd.DataFrame,
    ) -> None:
        """Persiste todas as tabelas no banco SQLite."""
        logger.info(f"Salvando banco de dados em: {DB_PATH}")
        with sqlite3.connect(DB_PATH) as conn:
            products.to_sql("products", conn, if_exists="replace", index=False)
            stores.to_sql("stores", conn, if_exists="replace", index=False)
            campaigns.to_sql("campaigns", conn, if_exists="replace", index=False)
            transactions.to_sql("transactions", conn, if_exists="replace", index=False)
            uplift_metrics.to_sql("uplift_metrics", conn, if_exists="replace", index=False)
            self._create_indexes(conn)
        logger.success("Banco de dados criado com sucesso.")

    @staticmethod
    def _create_indexes(conn: sqlite3.Connection) -> None:
        stmts = [
            "CREATE INDEX IF NOT EXISTS idx_txn_date     ON transactions(date)",
            "CREATE INDEX IF NOT EXISTS idx_txn_product  ON transactions(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_txn_store    ON transactions(store_id)",
            "CREATE INDEX IF NOT EXISTS idx_txn_campaign ON transactions(campaign_id)",
            "CREATE INDEX IF NOT EXISTS idx_camp_product ON campaigns(product_id)",
        ]
        for s in stmts:
            conn.execute(s)

    # ------------------------------------------------------------------
    # Ponto de entrada
    # ------------------------------------------------------------------

    def run(self) -> dict[str, pd.DataFrame]:
        """Executa o pipeline completo de geração de dados."""
        logger.info("=== [Agente Engenheiro de Dados] Gerando mock database ===")

        products = self.generate_products()
        stores = self.generate_stores()
        campaigns = self.generate_campaigns(products, stores)
        transactions = self.generate_transactions(products, stores, campaigns)
        uplift_metrics = self.generate_uplift_metrics(campaigns, transactions)

        self.save_to_database(products, stores, campaigns, transactions, uplift_metrics)

        datasets = {
            "products": products,
            "stores": stores,
            "campaigns": campaigns,
            "transactions": transactions,
            "uplift_metrics": uplift_metrics,
        }
        for name, df in datasets.items():
            logger.info(f"  {name}: {len(df):,} registros")

        return datasets
