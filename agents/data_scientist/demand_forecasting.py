"""
[Agente Cientista de Dados] — Módulo 2: Demand Forecasting

Treina modelo XGBoost para prever incremento de volume por produto/mês
e gera simulações para cenários de desconto: 10%, 20%, 30% e 40%.
"""

import sqlite3
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from loguru import logger

from config.settings import DB_PATH, PROCESSED_DIR, MOCK_CONFIG


@dataclass
class ForecastResult:
    """Resultado da simulação de demanda por cenário de desconto."""
    simulation_df: pd.DataFrame
    feature_importance: pd.DataFrame
    mape: float


class DemandForecaster:
    """
    Previsão de demanda com XGBoost + simulação de cenários de desconto.

    Features: produto/loja encoded, calendário, discount_pct, lags e médias móveis.
    Os encoders são ajustados apenas durante o treino; inferência usa .transform().
    """

    DISCOUNT_SCENARIOS = MOCK_CONFIG["discount_levels"]

    def __init__(self) -> None:
        self._le_product = LabelEncoder()
        self._le_store = LabelEncoder()
        self._model: xgb.XGBRegressor | None = None
        self._feature_cols: list[str] = []
        self._fitted = False
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Carrega transações do SQLite (discount_pct já presente na tabela)."""
        logger.info("[Forecast] Carregando dados...")
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(
                "SELECT * FROM transactions", conn, parse_dates=["date"]
            )
        df["discount_pct"] = df["discount_pct"].fillna(0.0)
        logger.info(f"[Forecast] Dataset: {len(df):,} registros")
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encoda product_id e store_id. fit=True durante treino, False na inferência."""
        df = df.copy()
        if fit:
            df["product_enc"] = self._le_product.fit_transform(df["product_id"])
            df["store_enc"] = self._le_store.fit_transform(df["store_id"])
            self._fitted = True
        else:
            known_products = set(self._le_product.classes_)
            known_stores = set(self._le_store.classes_)
            df["product_enc"] = df["product_id"].apply(
                lambda x: int(self._le_product.transform([x])[0])
                if x in known_products else -1
            )
            df["store_enc"] = df["store_id"].apply(
                lambda x: int(self._le_store.transform([x])[0])
                if x in known_stores else -1
            )
        return df

    def _engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Gera features de calendário, lag e desconto.

        Args:
            fit: True ao treinar (fit_transform nos encoders), False na inferência.
        """
        df = df.sort_values(["product_id", "store_id", "date"]).copy()

        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        df["quarter"] = df["date"].dt.quarter
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["day_of_year"] = df["date"].dt.dayofyear

        df = self._encode_categoricals(df, fit=fit)

        grp = df.groupby(["product_id", "store_id"])["volume"]
        df["lag_7d"] = grp.shift(7).fillna(0)
        df["lag_30d"] = grp.shift(30).fillna(0)
        df["lag_90d"] = grp.shift(90).fillna(0)
        df["ma_7d"] = grp.transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        ).fillna(0)
        df["ma_30d"] = grp.transform(
            lambda x: x.shift(1).rolling(30, min_periods=1).mean()
        ).fillna(0)

        df["discount_pct"] = df["discount_pct"].fillna(0.0)
        df["log_volume"] = np.log1p(df["volume"])

        return df

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> None:
        """Treina XGBoost com validação temporal (TimeSeriesSplit)."""
        logger.info("[Forecast] Iniciando treinamento do modelo...")

        self._feature_cols = [
            "product_enc", "store_enc",
            "month", "dayofweek", "quarter", "is_weekend", "day_of_year",
            "discount_pct",
            "lag_7d", "lag_30d", "lag_90d",
            "ma_7d", "ma_30d",
        ]

        df_model = df.dropna(subset=self._feature_cols + ["log_volume"])
        X = df_model[self._feature_cols]
        y = df_model["log_volume"]

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores: list[float] = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            m = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=42,
                n_jobs=-1, verbosity=0,
            )
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
            y_pred = np.expm1(m.predict(X.iloc[val_idx]))
            y_true = np.expm1(y.iloc[val_idx])
            mape_scores.append(mean_absolute_percentage_error(y_true + 1, y_pred + 1))
            logger.info(f"  Fold {fold + 1}: MAPE = {mape_scores[-1]:.2%}")

        # Treino final
        self._model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42,
            n_jobs=-1, verbosity=0,
        )
        self._model.fit(X, y)

        self._avg_mape = float(np.mean(mape_scores))
        logger.success(f"[Forecast] Treinamento concluído. MAPE médio: {self._avg_mape:.2%}")

    # ------------------------------------------------------------------
    # Simulação de cenários
    # ------------------------------------------------------------------

    def simulate_scenarios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Projeta volume e margem para cada (produto, loja) nos 4 cenários de desconto.
        Usa o último período disponível como base de features.
        """
        logger.info("[Forecast] Simulando cenários de desconto...")
        df = df.copy()
        df["year_month"] = df["date"].dt.to_period("M")
        last_period = df["year_month"].max()
        base = df[df["year_month"] == last_period].copy()

        records = []
        for discount in self.DISCOUNT_SCENARIOS:
            sim_features = base[self._feature_cols].copy()
            sim_features["discount_pct"] = discount

            pred_log = self._model.predict(sim_features)
            pred_vol = np.expm1(pred_log).clip(0)

            base_margin_pct = base["margin_pct"].values
            pred_margin_pct = np.clip(base_margin_pct * (1.0 - 1.5 * discount), -0.5, 0.8)

            chunk = base[["product_id", "store_id", "unit_price"]].copy()
            chunk["year_month"] = str(last_period)
            chunk["discount_pct"] = discount
            chunk["predicted_volume"] = np.round(pred_vol).astype(int).clip(0)
            chunk["simulated_price"] = np.round(
                chunk["unit_price"] * (1.0 - discount), 2
            )
            chunk["predicted_margin_pct"] = np.round(pred_margin_pct, 4)
            chunk["predicted_revenue"] = np.round(
                chunk["predicted_volume"] * chunk["simulated_price"], 2
            )
            chunk["predicted_margin"] = np.round(
                chunk["predicted_revenue"] * chunk["predicted_margin_pct"], 2
            )
            records.append(chunk)

        simulation_df = pd.concat(records, ignore_index=True)
        logger.info(f"[Forecast] Simulação: {len(simulation_df):,} cenários")
        return simulation_df

    def get_feature_importance(self) -> pd.DataFrame:
        importance = self._model.get_booster().get_score(importance_type="gain")
        return (
            pd.DataFrame(
                [(k, v) for k, v in importance.items()],
                columns=["feature", "importance_gain"],
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def run(self) -> ForecastResult:
        """Carga → features → treino → simulação."""
        logger.info("=== [Agente Cientista de Dados] Demand Forecasting ===")

        raw = self.load_data()
        df = self._engineer_features(raw, fit=True)

        self.train(df)

        simulation_df = self.simulate_scenarios(df)
        feature_importance = self.get_feature_importance()

        sim_path = PROCESSED_DIR / "demand_simulation.parquet"
        simulation_df.to_parquet(sim_path, index=False)
        logger.success(f"[Forecast] Simulação salva em {sim_path}")

        return ForecastResult(
            simulation_df=simulation_df,
            feature_importance=feature_importance,
            mape=self._avg_mape,
        )
