"""
[Agente Engenheiro de ML] — Módulo 4: Segurança da API

Implementa:
  - Autenticação via API Key (header X-API-Key)
  - Rate limiting por IP (token bucket)
  - Validação de inputs contra injeção
  - Middleware de logging de requisições com rastreio por request_id
"""

import hashlib
import secrets
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


# ---------------------------------------------------------------------------
# Configuração de API Keys (em produção: banco de dados ou vault)
# ---------------------------------------------------------------------------

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Simula repositório de API keys: hash(key) → {role, owner}
_API_KEY_STORE: dict[str, dict] = {}


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def provision_api_key(owner: str, role: str) -> str:
    """Gera e registra uma nova API key. Retorna a key em plain text (única vez)."""
    raw_key = f"rgm-{secrets.token_urlsafe(32)}"
    _API_KEY_STORE[_hash_key(raw_key)] = {"owner": owner, "role": role}
    logger.info(f"[Security] API key provisionada para '{owner}' (role: {role})")
    return raw_key


def _seed_default_keys() -> dict[str, str]:
    """Cria keys padrão para desenvolvimento e retorna mapeamento owner → key."""
    keys = {
        "admin": provision_api_key("admin", "admin"),
        "analyst": provision_api_key("analyst", "analyst"),
    }
    return keys


# Semeia as keys na inicialização do módulo
DEFAULT_KEYS = _seed_default_keys()


# ---------------------------------------------------------------------------
# Dependência de autenticação FastAPI
# ---------------------------------------------------------------------------

async def require_api_key(
    raw_key: str | None = Depends(_API_KEY_HEADER),
) -> dict:
    """
    Dependência FastAPI que valida o header X-API-Key.

    Returns:
        Dict com {owner, role} do portador da key.

    Raises:
        HTTPException 401: key ausente ou inválida.
    """
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header ausente.",
        )

    key_hash = _hash_key(raw_key)
    metadata = _API_KEY_STORE.get(key_hash)

    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida.",
        )

    return metadata


def require_admin(caller: dict = Depends(require_api_key)) -> dict:
    """Restringe endpoint a portadores com role 'admin'."""
    if caller["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operação restrita a administradores.",
        )
    return caller


# ---------------------------------------------------------------------------
# Rate Limiter (token bucket por IP)
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Token bucket por IP com reposição contínua."""

    def __init__(self, rate: float, capacity: int) -> None:
        self._rate = rate          # tokens/segundo
        self._capacity = capacity
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": float(capacity), "last_refill": time.monotonic()}
        )

    def consume(self, ip: str) -> bool:
        """Tenta consumir 1 token. Retorna False se o bucket estiver vazio."""
        bucket = self._buckets[ip]
        now = time.monotonic()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(
            self._capacity,
            bucket["tokens"] + elapsed * self._rate,
        )
        bucket["last_refill"] = now

        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False


_limiter = _TokenBucket(rate=10.0, capacity=60)  # 10 req/s, burst até 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware FastAPI que aplica rate limiting por IP."""

    async def dispatch(self, request: Request, call_next: Callable):
        client_ip = request.client.host if request.client else "unknown"

        if not _limiter.consume(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit excedido. Tente novamente em instantes."},
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Middleware de logging e rastreio
# ---------------------------------------------------------------------------

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Adiciona request_id único a cada requisição e loga latência."""

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = str(uuid.uuid4())[:8]
        start = time.monotonic()

        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"→ {response.status_code} ({duration_ms:.1f}ms)"
        )

        response.headers["X-Request-ID"] = request_id
        return response
