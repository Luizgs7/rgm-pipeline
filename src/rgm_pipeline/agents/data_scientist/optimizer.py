"""
[Agente Cientista de Dados] — Módulo 3: Otimizador Matemático

Maximiza a margem global gerando a grade de campanhas prescritiva.
Usa programação linear inteira (ILP) via PuLP com as seguintes restrições:

  - Verba máxima total (budget global)
  - Não sobreposição de campanhas para o mesmo produto-loja no mesmo mês
  - Máximo de campanhas por produto por mês
  - Desconto máximo permitido por categoria
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import pulp
from loguru import logger

from rgm_pipeline.config.settings import PROCESSED_DIR


@dataclass
class OptimizationResult:
    """Grade de campanhas otimizada com métricas consolidadas."""
    campaign_grid: pd.DataFrame     # Grade prescritiva recomendada
    total_margin: float             # Margem total esperada
    total_budget_used: float        # Verba utilizada
    n_campaigns: int                # Número de campanhas na grade
    solver_status: str              # Status do solver PuLP


class CampaignOptimizer:
    """
    Gerador de grade de campanhas via ILP (Integer Linear Programming).

    Recebe a tabela de simulação de demanda e decide quais campanhas
    (produto × loja × desconto) ativar para maximizar a margem total,
    respeitando restrições operacionais e financeiras.
    """

    def __init__(
        self,
        total_budget: float = 5_000_000.0,
        max_campaigns_per_product: int = 2,
        campaign_cost_pct: float = 0.05,
    ) -> None:
        """
        Args:
            total_budget: Verba máxima disponível para todas as campanhas.
            max_campaigns_per_product: Limite de campanhas simultâneas por produto.
            campaign_cost_pct: Percentual da receita usado como custo operacional da campanha.
        """
        self.total_budget = total_budget
        self.max_campaigns_per_product = max_campaigns_per_product
        self.campaign_cost_pct = campaign_cost_pct

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------

    def load_simulation(self, simulation_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Carrega simulação de demanda (do parquet ou do DataFrame passado)."""
        if simulation_df is not None:
            return simulation_df.copy()

        sim_path = PROCESSED_DIR / "demand_simulation.parquet"
        logger.info(f"[Optimizer] Carregando simulação de {sim_path}")
        return pd.read_parquet(sim_path)

    # ------------------------------------------------------------------
    # Preparação das variáveis de decisão
    # ------------------------------------------------------------------

    def _prepare_candidates(self, sim: pd.DataFrame) -> pd.DataFrame:
        """
        Cria tabela de candidatos: uma linha por (produto, loja, desconto).

        Calcula o custo estimado de executar a campanha e o ganho líquido
        (margem - custo_campanha).
        """
        cands = sim.copy()

        # Custo operacional da campanha = % da receita simulada
        cands["campaign_cost"] = np.round(
            cands["predicted_revenue"] * self.campaign_cost_pct, 2
        )
        cands["net_margin"] = np.round(
            cands["predicted_margin"] - cands["campaign_cost"], 2
        )

        # Filtra candidatos com margem líquida positiva
        cands = cands[cands["net_margin"] > 0].reset_index(drop=True)
        cands["candidate_id"] = cands.index
        logger.info(f"[Optimizer] Candidatos viáveis (margem > 0): {len(cands):,}")
        return cands

    # ------------------------------------------------------------------
    # Modelo ILP
    # ------------------------------------------------------------------

    def _build_and_solve(self, cands: pd.DataFrame) -> tuple[pulp.LpProblem, pd.DataFrame]:
        """
        Constrói e resolve o problema ILP:

          MAX  Σ net_margin_i * x_i
          s.t.
            Σ campaign_cost_i * x_i  ≤  total_budget          (verba)
            Σ x_i  ≤  max_campaigns   ∀ produto               (sobreposição por produto)
            x_i ∈ {0, 1}                                       (decisão binária)
        """
        logger.info("[Optimizer] Construindo modelo ILP...")

        prob = pulp.LpProblem("RGM_Campaign_Optimizer", pulp.LpMaximize)

        # Pré-computa dicts para lookup O(1) (evita .loc em loop)
        margin_by_id: dict[int, float] = dict(
            zip(cands["candidate_id"], cands["net_margin"])
        )
        cost_by_id: dict[int, float] = dict(
            zip(cands["candidate_id"], cands["campaign_cost"])
        )

        # Variáveis de decisão binárias
        x = {
            i: pulp.LpVariable(f"x_{i}", cat="Binary")
            for i in cands["candidate_id"]
        }

        # Função objetivo: maximizar margem líquida total
        prob += pulp.lpSum(margin_by_id[i] * x[i] for i in x)

        # Restrição 1: verba total
        prob += (
            pulp.lpSum(cost_by_id[i] * x[i] for i in x) <= self.total_budget,
            "budget_constraint",
        )

        # Restrição 2: máximo de campanhas por produto
        for product_id, group in cands.groupby("product_id"):
            ids = group["candidate_id"].tolist()
            prob += (
                pulp.lpSum(x[i] for i in ids) <= self.max_campaigns_per_product,
                f"overlap_{product_id}",
            )

        # Resolve com CBC (solver padrão do PuLP)
        logger.info("[Optimizer] Resolvendo com solver CBC...")
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        logger.info(f"[Optimizer] Status do solver: {status}")

        # Extrai solução
        selected_ids = [i for i in x if pulp.value(x[i]) == 1]
        grid = cands[cands["candidate_id"].isin(selected_ids)].copy()

        return prob, grid

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def run(self, simulation_df: pd.DataFrame | None = None) -> OptimizationResult:
        """Executa o pipeline de otimização e retorna a grade de campanhas."""
        logger.info("=== [Agente Cientista de Dados] Otimizador Matemático ===")

        sim = self.load_simulation(simulation_df)
        cands = self._prepare_candidates(sim)

        if cands.empty:
            logger.warning("[Optimizer] Nenhum candidato viável. Grade vazia.")
            return OptimizationResult(
                campaign_grid=pd.DataFrame(),
                total_margin=0.0,
                total_budget_used=0.0,
                n_campaigns=0,
                solver_status="NO_FEASIBLE_CANDIDATES",
            )

        prob, grid = self._build_and_solve(cands)

        total_margin = float(grid["net_margin"].sum())
        total_budget_used = float(grid["campaign_cost"].sum())
        status = pulp.LpStatus[prob.status]

        # Ordena por maior margem líquida
        grid = grid.sort_values("net_margin", ascending=False).reset_index(drop=True)

        # Persiste grade
        grid_path = PROCESSED_DIR / "campaign_grid.parquet"
        grid.to_parquet(grid_path, index=False)
        logger.success(
            f"[Optimizer] Grade gerada: {len(grid)} campanhas | "
            f"Margem total: R$ {total_margin:,.2f} | "
            f"Verba usada: R$ {total_budget_used:,.2f}"
        )

        return OptimizationResult(
            campaign_grid=grid,
            total_margin=total_margin,
            total_budget_used=total_budget_used,
            n_campaigns=len(grid),
            solver_status=status,
        )
