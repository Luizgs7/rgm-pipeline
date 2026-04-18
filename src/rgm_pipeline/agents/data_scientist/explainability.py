"""
[Agente Cientista de Dados] — Módulo 3: Explicabilidade (XAI)

Explica as recomendações da grade de campanhas usando SHAP para o modelo
de previsão de demanda e gera um Score de Confiança composto por:
  - Acurácia do modelo (MAPE)
  - Estabilidade da predição (std via bootstrap)
  - Cobertura histórica do produto-loja
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger

from rgm_pipeline.config.settings import PROCESSED_DIR


@dataclass
class ExplainabilityReport:
    """Relatório de explicabilidade para a grade de campanhas."""
    campaign_explanations: pd.DataFrame     # Explicações por campanha recomendada
    global_shap_summary: pd.DataFrame       # Importância global das features
    confidence_summary: pd.DataFrame        # Score de confiança por campanha


class CampaignExplainer:
    """
    Gera explicações locais (SHAP) e Score de Confiança para cada campanha
    recomendada pelo otimizador.

    O Score de Confiança (0–1) combina:
      1. Accuracy score: 1 - MAPE do modelo de previsão
      2. Stability score: 1 - coeficiente de variação das predições bootstrap
      3. Coverage score: cobertura histórica do par produto-loja
    """

    def __init__(self, model: xgb.XGBRegressor, feature_cols: list[str]) -> None:
        self._model = model
        self._feature_cols = feature_cols
        self._explainer: shap.TreeExplainer | None = None

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def _init_explainer(self, background_data: pd.DataFrame) -> None:
        """Inicializa o TreeExplainer do SHAP com amostra de background."""
        sample = background_data[self._feature_cols].sample(
            min(200, len(background_data)), random_state=42
        )
        self._explainer = shap.TreeExplainer(self._model, sample)
        logger.info("[XAI] SHAP TreeExplainer inicializado.")

    def compute_shap_values(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Calcula valores SHAP para o conjunto X.

        Returns:
            shap_values: array de shape (n_samples, n_features)
            shap_df: DataFrame com colunas = features e linhas = amostras
        """
        shap_values = self._explainer.shap_values(X[self._feature_cols])
        shap_df = pd.DataFrame(shap_values, columns=self._feature_cols, index=X.index)
        return shap_values, shap_df

    def global_feature_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Importância global = média dos |SHAP values| por feature."""
        importance = np.abs(shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self._feature_cols, "mean_abs_shap": importance})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Score de Confiança
    # ------------------------------------------------------------------

    def compute_confidence_score(
        self,
        grid: pd.DataFrame,
        full_data: pd.DataFrame,
        model_mape: float,
        n_bootstrap: int = 50,
    ) -> pd.DataFrame:
        """
        Calcula Score de Confiança composto para cada campanha na grade.

        Args:
            grid: Grade de campanhas (output do otimizador).
            full_data: Dataset completo com features (para bootstrap e cobertura).
            model_mape: MAPE global do modelo de previsão.
            n_bootstrap: Iterações de bootstrap para estimar estabilidade.
        """
        logger.info("[XAI] Calculando Score de Confiança...")

        accuracy_score = max(0.0, 1.0 - model_mape)
        results = []

        for _, row in grid.iterrows():
            # Coverage: quantos dias históricos existem para este produto-loja
            mask = (
                (full_data["product_id"] == row["product_id"]) &
                (full_data["store_id"] == row["store_id"])
            )
            n_obs = mask.sum()
            coverage_score = min(1.0, n_obs / 365.0)

            # Stability: bootstrap na amostra do produto-loja
            local_data = full_data[mask][self._feature_cols].copy()
            if len(local_data) >= 10:
                preds = []
                for _ in range(n_bootstrap):
                    sample = local_data.sample(
                        min(30, len(local_data)), replace=True, random_state=None
                    )
                    sample = sample.copy()
                    sample["discount_pct"] = row["discount_pct"]
                    preds.append(np.expm1(self._model.predict(sample)).mean())
                cv = float(np.std(preds) / (np.mean(preds) + 1e-9))
                stability_score = max(0.0, 1.0 - cv)
            else:
                stability_score = 0.5

            # Score composto (pesos calibráveis)
            confidence = round(
                0.40 * accuracy_score +
                0.35 * stability_score +
                0.25 * coverage_score,
                4,
            )

            results.append({
                "product_id": row["product_id"],
                "store_id": row["store_id"],
                "discount_pct": row["discount_pct"],
                "accuracy_score": round(accuracy_score, 4),
                "stability_score": round(stability_score, 4),
                "coverage_score": round(coverage_score, 4),
                "confidence_score": confidence,
                "confidence_label": self._label(confidence),
            })

        return pd.DataFrame(results)

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.80:
            return "ALTA"
        elif score >= 0.60:
            return "MÉDIA"
        else:
            return "BAIXA"

    # ------------------------------------------------------------------
    # Geração de explicações textuais por campanha
    # ------------------------------------------------------------------

    def explain_campaigns(
        self,
        grid: pd.DataFrame,
        full_data: pd.DataFrame,
        confidence_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Gera explicação textual do motivo de cada recomendação,
        baseada nos top drivers SHAP para aquela observação.
        """
        logger.info("[XAI] Gerando explicações textuais...")

        rows = []
        for _, camp_row in grid.iterrows():
            mask = (
                (full_data["product_id"] == camp_row["product_id"]) &
                (full_data["store_id"] == camp_row["store_id"])
            )
            local = full_data[mask].copy()
            if local.empty:
                continue

            # Usa última observação disponível
            sample = local.iloc[[-1]][self._feature_cols].copy()
            sample["discount_pct"] = camp_row["discount_pct"]

            shap_row = self._explainer.shap_values(sample)[0]
            top_features = sorted(
                zip(self._feature_cols, shap_row),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:3]

            drivers = ", ".join(
                f"{feat} ({'+' if val > 0 else ''}{val:.2f})"
                for feat, val in top_features
            )

            conf_row = confidence_df[
                (confidence_df["product_id"] == camp_row["product_id"]) &
                (confidence_df["store_id"] == camp_row["store_id"]) &
                (confidence_df["discount_pct"] == camp_row["discount_pct"])
            ]
            conf_score = conf_row["confidence_score"].values[0] if not conf_row.empty else 0.0
            conf_label = conf_row["confidence_label"].values[0] if not conf_row.empty else "N/A"

            explanation = (
                f"Campanha recomendada com desconto de {camp_row['discount_pct']:.0%}. "
                f"Margem líquida esperada: R$ {camp_row['net_margin']:,.2f}. "
                f"Principais drivers da previsão: [{drivers}]. "
                f"Confiança: {conf_label} ({conf_score:.0%})."
            )

            rows.append({
                "product_id": camp_row["product_id"],
                "store_id": camp_row["store_id"],
                "discount_pct": camp_row["discount_pct"],
                "net_margin": camp_row["net_margin"],
                "confidence_score": conf_score,
                "confidence_label": conf_label,
                "explanation": explanation,
                "top_shap_drivers": drivers,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def run(
        self,
        grid: pd.DataFrame,
        full_data: pd.DataFrame,
        model_mape: float,
    ) -> ExplainabilityReport:
        """Executa pipeline completo de explicabilidade."""
        logger.info("=== [Agente Cientista de Dados] Explicabilidade (XAI) ===")

        self._init_explainer(full_data)

        # SHAP global (amostra)
        sample_data = full_data.sample(min(500, len(full_data)), random_state=42)
        shap_values, _ = self.compute_shap_values(sample_data)
        global_importance = self.global_feature_importance(shap_values)

        logger.info("[XAI] Top 5 features por SHAP global:")
        for _, row_imp in global_importance.head(5).iterrows():
            logger.info(f"  {row_imp['feature']}: {row_imp['mean_abs_shap']:.4f}")

        # Score de confiança por campanha
        confidence_df = self.compute_confidence_score(grid, full_data, model_mape)

        # Explicações textuais
        explanations_df = self.explain_campaigns(grid, full_data, confidence_df)

        # Persiste
        exp_path = PROCESSED_DIR / "campaign_explanations.parquet"
        explanations_df.to_parquet(exp_path, index=False)
        logger.success(f"[XAI] Explicações salvas em {exp_path}")

        return ExplainabilityReport(
            campaign_explanations=explanations_df,
            global_shap_summary=global_importance,
            confidence_summary=confidence_df,
        )
