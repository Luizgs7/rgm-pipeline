"""
Script de execução do Módulo 4 — [Agente Engenheiro de Machine Learning]

Inicia o servidor FastAPI via Uvicorn com configurações de produção.

Pré-requisito: run_module1.py deve ter sido executado.
               run_module2_3.py deve ter sido executado para servir predições.

Uso:
    python run_module4.py
    python run_module4.py --host 0.0.0.0 --port 8080 --workers 4
"""

import argparse
import uvicorn
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGM Pipeline API Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true", help="Hot-reload (dev only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(
        f"Iniciando RGM Pipeline API em http://{args.host}:{args.port}\n"
        f"  Docs: http://{args.host}:{args.port}/docs\n"
        f"  ReDoc: http://{args.host}:{args.port}/redoc"
    )

    uvicorn.run(
        "agents.ml_engineer.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info",
        access_log=False,   # substituído pelo RequestTracingMiddleware
    )


if __name__ == "__main__":
    main()
