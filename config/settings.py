"""Configurações globais do projeto RGM Pipeline."""

from pathlib import Path

# Diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Banco de dados mock
DB_PATH = RAW_DIR / "rgm_database.db"

# Parâmetros do mock
MOCK_CONFIG = {
    "seed": 42,
    "n_products": 20,
    "n_stores": 10,
    "n_campaigns": 80,
    "history_start": "2023-01-01",
    "history_end": "2024-12-31",
    "discount_levels": [0.10, 0.20, 0.30, 0.40],
    "categories": ["Bebidas", "Laticínios", "Higiene", "Limpeza", "Snacks"],
    "regions": ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"],
}

# Controle de acesso — perfis e permissões
ACCESS_ROLES = {
    "admin": {
        "tables": ["transactions", "campaigns", "uplift_metrics", "products", "stores"],
        "operations": ["read", "write", "delete"],
        "mask_pii": False,
    },
    "analyst": {
        "tables": ["transactions", "campaigns", "uplift_metrics", "products", "stores"],
        "operations": ["read"],
        "mask_pii": False,
    },
    "viewer": {
        "tables": ["campaigns", "uplift_metrics"],
        "operations": ["read"],
        "mask_pii": True,
    },
}

# Data Quality — limiares
DQ_CONFIG = {
    "max_null_pct": 0.05,       # 5% máximo de nulos por coluna
    "iqr_multiplier": 3.0,       # Fator IQR para detecção de anomalias
    "min_margin_pct": -0.10,     # Margem mínima aceitável (-10%)
    "max_discount_pct": 0.60,    # Desconto máximo permitido (60%)
}
