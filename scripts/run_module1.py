"""
Script de execução do Módulo 1 — [Agente Engenheiro de Dados]

Orquestra:
  1. Geração do banco de dados mock (MockDataGenerator)
  2. Validação de Data Quality (DataQualityRunner)
  3. Demonstração do controle de acesso (AccessControlService)
"""

from loguru import logger

from rgm_pipeline.agents.data_engineer.mock_generator import MockDataGenerator
from rgm_pipeline.agents.data_engineer.data_quality import DataQualityRunner
from rgm_pipeline.agents.data_engineer.access_control import AccessControlService, User, AuthenticationError


def main() -> None:
    logger.info("=" * 60)
    logger.info("RGM Pipeline — Módulo 1: Agente Engenheiro de Dados")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # 1. Gerar banco de dados mock
    # ----------------------------------------------------------------
    generator = MockDataGenerator()
    datasets = generator.run()

    # ----------------------------------------------------------------
    # 2. Executar Data Quality
    # ----------------------------------------------------------------
    dq_runner = DataQualityRunner()
    # Validamos um subconjunto para não tornar a demo lenta
    sample_datasets = {
        name: df.sample(min(5_000, len(df)), random_state=42)
        for name, df in datasets.items()
    }
    reports = dq_runner.validate(sample_datasets)
    dq_runner.print_full_report(reports)

    # ----------------------------------------------------------------
    # 3. Demonstrar controle de acesso
    # ----------------------------------------------------------------
    logger.info("\n[ACL] Demonstração de Controle de Acesso")
    acl = AccessControlService()

    # Admin acessa tudo sem mascaramento
    admin = acl.authenticate("U001")
    txn_admin = acl.get_data(admin, "transactions", datasets)
    logger.info(f"Admin leu transactions: {len(txn_admin):,} registros | "
                f"colunas: {list(txn_admin.columns)}")

    # Viewer acessa campanhas com mascaramento
    viewer = acl.authenticate("U003")
    camp_viewer = acl.get_data(viewer, "campaigns", datasets)
    logger.info(f"Viewer leu campaigns (mascarado): budget={camp_viewer['budget'].iloc[0]}, "
                f"store_id={camp_viewer['store_id'].iloc[0]}")

    # Viewer tenta acessar transactions (acesso negado)
    try:
        acl.get_data(viewer, "transactions", datasets)
    except AuthenticationError as e:
        logger.warning(str(e))

    # Log de auditoria (somente admin)
    audit_df = acl.get_audit_log(admin)
    logger.info(f"\nLog de auditoria ({len(audit_df)} eventos registrados):\n{audit_df.to_string()}")

    logger.success("\n[Módulo 1] Concluído com sucesso.")
    logger.info("Banco de dados disponível em: data/raw/rgm_database.db")


if __name__ == "__main__":
    main()
