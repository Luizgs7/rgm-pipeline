"""
[Agente Engenheiro de Dados] — Módulo 1: Controle de Acesso

Simula RBAC (Role-Based Access Control) sobre as tabelas do RGM Pipeline.
Garante que usuários só acessem dados permitidos pelo seu perfil e que
campos sensíveis sejam mascarados conforme necessário.
"""

import hashlib
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from loguru import logger

from rgm_pipeline.config.settings import ACCESS_ROLES


# ---------------------------------------------------------------------------
# Estruturas de identidade
# ---------------------------------------------------------------------------

@dataclass
class User:
    """Representa um usuário do sistema RGM."""
    user_id: str
    username: str
    role: str
    email: str

    def __post_init__(self) -> None:
        if self.role not in ACCESS_ROLES:
            raise ValueError(
                f"Role inválida: '{self.role}'. "
                f"Roles disponíveis: {list(ACCESS_ROLES.keys())}"
            )


class AuthenticationError(PermissionError):
    """Acesso negado por falha de autenticação ou autorização."""


# ---------------------------------------------------------------------------
# Serviço de controle de acesso
# ---------------------------------------------------------------------------

class AccessControlService:
    """
    Gerencia autenticação e autorização de acesso às tabelas do RGM.

    Em produção, este serviço seria substituído por integração com
    um IdP (ex.: Keycloak, Azure AD) via OAuth 2.0 / OIDC.
    """

    def __init__(self) -> None:
        # Simulação de repositório de usuários (substituir por DB real)
        self._users: dict[str, User] = {}
        self._audit_log: list[dict] = []
        self._seed_default_users()

    def _seed_default_users(self) -> None:
        """Popula usuários padrão para demonstração."""
        defaults = [
            User("U001", "admin_rgm",    "admin",   "admin@rgm.com"),
            User("U002", "analista_mkt", "analyst", "analista@rgm.com"),
            User("U003", "gerente_pdv",  "viewer",  "gerente@rgm.com"),
        ]
        for user in defaults:
            self._users[user.user_id] = user

    # ------------------------------------------------------------------
    # Autenticação (simulada)
    # ------------------------------------------------------------------

    def authenticate(self, user_id: str) -> User:
        """Autentica um usuário pelo ID (stub — em produção usa JWT/OAuth)."""
        user = self._users.get(user_id)
        if not user:
            raise AuthenticationError(f"Usuário '{user_id}' não encontrado.")
        logger.info(f"[ACL] Usuário autenticado: {user.username} (role: {user.role})")
        return user

    def register_user(self, user: User) -> None:
        """Registra um novo usuário no sistema."""
        if user.user_id in self._users:
            raise ValueError(f"Usuário '{user.user_id}' já existe.")
        self._users[user.user_id] = user
        logger.info(f"[ACL] Novo usuário registrado: {user.username} (role: {user.role})")

    # ------------------------------------------------------------------
    # Autorização
    # ------------------------------------------------------------------

    def authorize(self, user: User, table: str, operation: str) -> None:
        """
        Verifica se o usuário tem permissão para a operação na tabela.

        Raises:
            AuthenticationError: se a permissão for negada.
        """
        role_config = ACCESS_ROLES[user.role]
        allowed_tables = role_config["tables"]
        allowed_ops = role_config["operations"]

        self._audit(user, table, operation, granted=False)

        if table not in allowed_tables:
            raise AuthenticationError(
                f"[ACL] NEGADO — {user.username} ({user.role}) não tem acesso "
                f"à tabela '{table}'."
            )

        if operation not in allowed_ops:
            raise AuthenticationError(
                f"[ACL] NEGADO — {user.username} ({user.role}) não pode "
                f"executar '{operation}' em '{table}'."
            )

        self._audit(user, table, operation, granted=True)
        logger.debug(f"[ACL] PERMITIDO — {user.username}: {operation} em {table}")

    # ------------------------------------------------------------------
    # Acesso seguro aos dados
    # ------------------------------------------------------------------

    def get_data(
        self,
        user: User,
        table: str,
        datasets: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Retorna os dados da tabela aplicando as políticas de acesso do usuário.

        Aplica mascaramento de campos sensíveis para roles restritas.
        """
        self.authorize(user, table, "read")

        df = datasets[table].copy()
        role_config = ACCESS_ROLES[user.role]

        if role_config["mask_pii"]:
            df = self._mask_sensitive_fields(df, table)

        logger.info(
            f"[ACL] {user.username} acessou '{table}' "
            f"({'com mascaramento' if role_config['mask_pii'] else 'acesso completo'})"
        )
        return df

    # ------------------------------------------------------------------
    # Mascaramento de dados sensíveis
    # ------------------------------------------------------------------

    # Campos sensíveis por tabela (nome do campo → estratégia de mascaramento)
    _SENSITIVE_FIELDS: dict[str, dict[str, str]] = {
        "transactions": {
            "store_id": "hash",
            "cost": "suppress",
            "margin": "suppress",
            "margin_pct": "suppress",
        },
        "campaigns": {
            "budget": "suppress",
            "store_id": "hash",
        },
        "uplift_metrics": {
            "roi": "suppress",
            "incremental_margin": "suppress",
        },
    }

    def _mask_sensitive_fields(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """Aplica mascaramento conforme estratégia por campo."""
        sensitive = self._SENSITIVE_FIELDS.get(table, {})

        for col, strategy in sensitive.items():
            if col not in df.columns:
                continue

            if strategy == "hash":
                df[col] = df[col].apply(self._hash_value)
            elif strategy == "suppress":
                df[col] = "***"

        return df

    @staticmethod
    def _hash_value(value: str) -> str:
        """Substitui o valor por seu SHA-256 truncado (primeiros 8 chars)."""
        return hashlib.sha256(str(value).encode()).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Auditoria
    # ------------------------------------------------------------------

    def _audit(self, user: User, table: str, operation: str, granted: bool) -> None:
        """Registra evento de acesso no log de auditoria."""
        import datetime
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "table": table,
            "operation": operation,
            "granted": granted,
        }
        self._audit_log.append(entry)

    def get_audit_log(self, user: User) -> pd.DataFrame:
        """Retorna o log de auditoria (somente admins)."""
        if user.role != "admin":
            raise AuthenticationError(
                f"[ACL] NEGADO — apenas admins podem consultar o log de auditoria."
            )
        return pd.DataFrame(self._audit_log)

    def get_permissions(self, user: User) -> dict:
        """Retorna as permissões do usuário de forma legível."""
        return {
            "user": user.username,
            "role": user.role,
            **ACCESS_ROLES[user.role],
        }
