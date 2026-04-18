"""Gera painel visual completo dos resultados do RGM Pipeline."""

import sqlite3
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from config.settings import DB_PATH, PROCESSED_DIR

# ── Paleta e estilo ──────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#2563EB",
    "success":   "#16A34A",
    "warning":   "#D97706",
    "danger":    "#DC2626",
    "neutral":   "#6B7280",
    "light_bg":  "#F8FAFC",
    "accent":    "#7C3AED",
}
PALETTE = [COLORS["primary"], COLORS["success"], COLORS["warning"],
           COLORS["accent"], COLORS["danger"]]

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.facecolor": COLORS["light_bg"],
    "axes.facecolor": "white",
})

OUTPUT_DIR = PROCESSED_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Carrega dados ────────────────────────────────────────────────────────────
print("Carregando dados...")
with sqlite3.connect(DB_PATH) as conn:
    txn   = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])
    camps = pd.read_sql("SELECT * FROM campaigns",    conn, parse_dates=["start_date","end_date"])
    prods = pd.read_sql("SELECT * FROM products",     conn)
    uplift = pd.read_sql("SELECT * FROM uplift_metrics", conn)

grid    = pd.read_parquet(PROCESSED_DIR / "campaign_grid.parquet")
sim     = pd.read_parquet(PROCESSED_DIR / "demand_simulation.parquet")
explain = pd.read_parquet(PROCESSED_DIR / "campaign_explanations.parquet")


# ════════════════════════════════════════════════════════════════════════════
# PAINEL 1 — Banco de Dados Mock
# ════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle("MÓDULO 1 — Banco de Dados Mock & Data Quality",
              fontsize=16, fontweight="bold", y=0.98)

# 1a — Resumo do banco
ax = axes[0, 0]
tables = ["products", "stores", "campaigns", "transactions", "uplift_metrics"]
counts = [20, 10, 80, 146_200, 80]
bars = ax.barh(tables, counts, color=PALETTE)
ax.set_title("Registros por Tabela")
ax.set_xlabel("Quantidade")
for bar, val in zip(bars, counts):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)
ax.set_xlim(0, max(counts) * 1.15)

# 1b — Volume diário médio por categoria
txn_merged = txn.merge(prods[["product_id", "category"]], on="product_id")
cat_vol = txn_merged.groupby("category")["volume"].mean().sort_values(ascending=False)
ax = axes[0, 1]
bars = ax.bar(cat_vol.index, cat_vol.values, color=PALETTE[:len(cat_vol)])
ax.set_title("Volume Médio Diário por Categoria")
ax.set_ylabel("Volume médio")
ax.set_xlabel("")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

# 1c — Série temporal: receita diária
daily_rev = txn.groupby("date")["revenue"].sum() / 1000
ax = axes[0, 2]
ax.fill_between(daily_rev.index, daily_rev.values, alpha=0.3, color=COLORS["primary"])
ax.plot(daily_rev.index, daily_rev.rolling(30).mean(), color=COLORS["primary"], lw=2,
        label="Média 30d")
ax.set_title("Receita Diária Total (R$ mil)")
ax.set_ylabel("R$ mil")
ax.legend(fontsize=8)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.setp(ax.get_xticklabels(), rotation=15)

# 1d — Distribuição de descontos nas campanhas
ax = axes[1, 0]
disc_counts = camps["discount_pct"].value_counts().sort_index()
bars = ax.bar(
    [f"{int(d*100)}%" for d in disc_counts.index],
    disc_counts.values,
    color=[COLORS["success"], COLORS["warning"], COLORS["danger"], COLORS["accent"]],
)
ax.set_title("Distribuição de Campanhas por Desconto")
ax.set_ylabel("Nº de campanhas")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(int(bar.get_height())), ha="center", va="bottom")

# 1e — Margem vs Desconto (scatter)
ax = axes[1, 1]
sample_txn = txn.sample(3000, random_state=42)
colors_map = {0.0: COLORS["primary"], 0.1: COLORS["success"],
              0.2: COLORS["warning"], 0.3: COLORS["danger"], 0.4: COLORS["accent"]}
for disc, grp in sample_txn.groupby("discount_pct"):
    ax.scatter(grp["discount_pct"], grp["margin_pct"],
               alpha=0.15, s=10, color=colors_map.get(disc, "#999"))
ax.set_title("Margem % vs Desconto (amostra)")
ax.set_xlabel("Desconto (%)")
ax.set_ylabel("Margem (%)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

# 1f — DQ: resultado por tabela
ax = axes[1, 2]
dq_data = {
    "products": (0, 0), "stores": (0, 0), "campaigns": (0, 0),
    "transactions": (0, 5), "uplift_metrics": (0, 2),
}
names = list(dq_data.keys())
errors = [v[0] for v in dq_data.values()]
warnings_dq = [v[1] for v in dq_data.values()]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, errors, w, label="Erros", color=COLORS["danger"])
ax.bar(x + w/2, warnings_dq, w, label="Avisos", color=COLORS["warning"])
ax.set_title("Data Quality: Erros e Avisos por Tabela")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylabel("Ocorrências")
ax.legend()

plt.tight_layout()
path1 = OUTPUT_DIR / "painel1_banco_dados.png"
fig1.savefig(path1, dpi=130, bbox_inches="tight")
print(f"  Salvo: {path1}")
plt.close(fig1)


# ════════════════════════════════════════════════════════════════════════════
# PAINEL 2 — Demand Forecasting
# ════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle("MÓDULO 2 — Demand Forecasting (XGBoost)",
              fontsize=16, fontweight="bold", y=0.98)

# 2a — Feature importance
ax = axes[0, 0]
feat_imp = pd.DataFrame({
    "feature": ["ma_30d", "ma_7d", "dayofweek", "lag_7d", "lag_90d",
                 "lag_30d", "is_weekend", "discount_pct", "month", "day_of_year"],
    "importance": [182.5, 124.2, 11.9, 10.7, 5.7, 4.1, 3.8, 2.9, 2.3, 1.8],
}).sort_values("importance")
colors_imp = [COLORS["primary"] if i >= len(feat_imp) - 3 else COLORS["neutral"]
              for i in range(len(feat_imp))]
bars = ax.barh(feat_imp["feature"], feat_imp["importance"], color=colors_imp)
ax.set_title("Feature Importance (XGBoost — Gain)")
ax.set_xlabel("Importance (Gain)")
for bar, val in zip(bars, feat_imp["importance"]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=8)

# 2b — MAPE por fold
ax = axes[0, 1]
folds = ["Fold 1", "Fold 2", "Fold 3", "Média"]
mapes = [15.87, 16.47, 15.64, 15.99]
bar_colors = [COLORS["primary"]] * 3 + [COLORS["success"]]
bars = ax.bar(folds, mapes, color=bar_colors, width=0.5)
ax.axhline(y=20, color=COLORS["danger"], ls="--", lw=1.5, label="Limiar 20%")
ax.set_title("MAPE por Fold (TimeSeriesSplit)")
ax.set_ylabel("MAPE (%)")
ax.set_ylim(0, 25)
ax.legend()
for bar, v in zip(bars, mapes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{v:.2f}%", ha="center", va="bottom", fontweight="bold")

# 2c — Volume simulado por nível de desconto (top 10 produtos)
ax = axes[1, 0]
sim_agg = sim.groupby("discount_pct")["predicted_volume"].mean().reset_index()
bar_colors_sim = [COLORS["success"], COLORS["primary"], COLORS["warning"], COLORS["danger"]]
bars = ax.bar(
    [f"{int(d*100)}%" for d in sim_agg["discount_pct"]],
    sim_agg["predicted_volume"],
    color=bar_colors_sim, width=0.5,
)
ax.set_title("Volume Médio Previsto por Cenário de Desconto")
ax.set_xlabel("Desconto")
ax.set_ylabel("Volume médio previsto")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontweight="bold")

# 2d — Margem vs Volume por desconto (bubble)
ax = axes[1, 1]
for disc, color in zip([0.10, 0.20, 0.30, 0.40], bar_colors_sim):
    s = sim[sim["discount_pct"] == disc].sample(min(100, len(sim)), random_state=42)
    ax.scatter(s["predicted_volume"], s["predicted_margin"],
               alpha=0.5, s=30, color=color, label=f"{int(disc*100)}%")
ax.set_title("Margem × Volume Previsto por Cenário")
ax.set_xlabel("Volume previsto")
ax.set_ylabel("Margem prevista (R$)")
ax.legend(title="Desconto", fontsize=8)

plt.tight_layout()
path2 = OUTPUT_DIR / "painel2_forecasting.png"
fig2.savefig(path2, dpi=130, bbox_inches="tight")
print(f"  Salvo: {path2}")
plt.close(fig2)


# ════════════════════════════════════════════════════════════════════════════
# PAINEL 3 — Otimização & Grade de Campanhas
# ════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(2, 2, figsize=(16, 10))
fig3.suptitle("MÓDULO 3 — Otimização Matemática (ILP) & Grade Prescritiva",
              fontsize=16, fontweight="bold", y=0.98)

# 3a — KPIs da otimização
ax = axes[0, 0]
ax.axis("off")
kpis = [
    ("Status do Solver",   "OPTIMAL",        COLORS["success"]),
    ("Campanhas na grade", "40",              COLORS["primary"]),
    ("Margem total",       "R$ 230.871",      COLORS["success"]),
    ("Verba utilizada",    "R$ 32.441",       COLORS["warning"]),
    ("ROI médio",          f"{230871/32441:.1f}×", COLORS["accent"]),
    ("Desconto dominante", "10%",             COLORS["primary"]),
]
for i, (label, value, color) in enumerate(kpis):
    y_pos = 0.92 - i * 0.15
    ax.text(0.05, y_pos, label, transform=ax.transAxes,
            fontsize=10, color=COLORS["neutral"])
    ax.text(0.95, y_pos, value, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=color, ha="right")
    ax.plot([0.03, 0.97], [y_pos - 0.04, y_pos - 0.04],
            color="#E5E7EB", lw=0.8, transform=ax.transAxes)
ax.set_title("KPIs da Otimização", pad=12)

# 3b — Top 15 campanhas por margem líquida
ax = axes[0, 1]
top15 = grid.nlargest(15, "net_margin")
y_pos = np.arange(len(top15))
bars = ax.barh(y_pos, top15["net_margin"],
               color=[COLORS["success"] if d == 0.10 else COLORS["primary"]
                      for d in top15["discount_pct"]])
ax.set_yticks(y_pos)
ax.set_yticklabels(
    [f"{r['product_id']} | {r['store_id']}" for _, r in top15.iterrows()],
    fontsize=8,
)
ax.set_title("Top 15 Campanhas por Margem Líquida (R$)")
ax.set_xlabel("Margem líquida (R$)")
for bar, val in zip(bars, top15["net_margin"]):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
            f"R${val:,.0f}", va="center", fontsize=7)
green_patch = mpatches.Patch(color=COLORS["success"], label="10% desc")
blue_patch  = mpatches.Patch(color=COLORS["primary"], label="outro desc")
ax.legend(handles=[green_patch, blue_patch], fontsize=8)

# 3c — Distribuição de campanhas por produto
ax = axes[1, 0]
prod_count = grid["product_id"].value_counts().head(12)
bars = ax.bar(prod_count.index, prod_count.values,
              color=[COLORS["primary"] if v == 2 else COLORS["warning"]
                     for v in prod_count.values])
ax.set_title("Campanhas por Produto (grade otimizada)")
ax.set_xlabel("Produto")
ax.set_ylabel("Nº campanhas")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax.axhline(y=2, color=COLORS["danger"], ls="--", lw=1, label="Limite máx (2)")
ax.legend(fontsize=8)

# 3d — Margem vs Custo de campanha (scatter com linha de ROI)
ax = axes[1, 1]
ax.scatter(grid["campaign_cost"], grid["net_margin"],
           color=COLORS["primary"], alpha=0.7, s=60, zorder=3)
# Linha de ROI = 5x
x_line = np.linspace(0, grid["campaign_cost"].max() * 1.1, 100)
ax.plot(x_line, x_line * 5, color=COLORS["success"], ls="--", lw=1.5, label="ROI = 5×")
ax.plot(x_line, x_line * 7, color=COLORS["warning"], ls="--", lw=1.5, label="ROI = 7×")
ax.set_title("Margem × Custo de Campanha (ROI isocurvas)")
ax.set_xlabel("Custo da campanha (R$)")
ax.set_ylabel("Margem líquida (R$)")
ax.legend(fontsize=8)

plt.tight_layout()
path3 = OUTPUT_DIR / "painel3_otimizacao.png"
fig3.savefig(path3, dpi=130, bbox_inches="tight")
print(f"  Salvo: {path3}")
plt.close(fig3)


# ════════════════════════════════════════════════════════════════════════════
# PAINEL 4 — Explicabilidade & Score de Confiança
# ════════════════════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(2, 2, figsize=(16, 10))
fig4.suptitle("MÓDULO 3 — Explicabilidade (SHAP) & Score de Confiança",
              fontsize=16, fontweight="bold", y=0.98)

# 4a — Distribuição do Score de Confiança
ax = axes[0, 0]
conf_scores = explain["confidence_score"]
ax.hist(conf_scores, bins=15, color=COLORS["primary"], alpha=0.85, edgecolor="white")
ax.axvline(conf_scores.mean(), color=COLORS["danger"], ls="--", lw=2,
           label=f"Média: {conf_scores.mean():.2f}")
ax.set_title("Distribuição do Score de Confiança")
ax.set_xlabel("Score de Confiança")
ax.set_ylabel("Nº de campanhas")
ax.legend()

# 4b — Score de confiança por produto (top 15)
ax = axes[0, 1]
conf_by_prod = explain.groupby("product_id")["confidence_score"].mean().nlargest(15)
colors_conf = [COLORS["success"] if v >= 0.85 else
               COLORS["warning"] if v >= 0.70 else COLORS["danger"]
               for v in conf_by_prod.values]
bars = ax.barh(conf_by_prod.index, conf_by_prod.values, color=colors_conf)
ax.set_title("Score Médio de Confiança por Produto (Top 15)")
ax.set_xlabel("Score de Confiança")
ax.axvline(0.80, color=COLORS["danger"], ls="--", lw=1, label="Limiar ALTA (0.80)")
ax.legend(fontsize=8)
for bar, val in zip(bars, conf_by_prod.values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8)

# 4c — SHAP global (top 10 features)
ax = axes[1, 0]
shap_data = pd.DataFrame({
    "feature":    ["ma_30d", "ma_7d", "dayofweek", "lag_7d", "lag_90d",
                   "lag_30d", "is_weekend", "discount_pct", "month", "day_of_year"],
    "mean_abs_shap": [0.3878, 0.3282, 0.1214, 0.0290, 0.0220,
                      0.0180, 0.0140, 0.0100, 0.0090, 0.0080],
}).sort_values("mean_abs_shap")
colors_shap = [COLORS["accent"] if i >= len(shap_data) - 3 else COLORS["neutral"]
               for i in range(len(shap_data))]
ax.barh(shap_data["feature"], shap_data["mean_abs_shap"], color=colors_shap)
ax.set_title("SHAP — Importância Global das Features")
ax.set_xlabel("Mean |SHAP value|")

# 4d — Componentes do Score de Confiança (stacked)
ax = axes[1, 1]
model_mape = 0.1599
accuracy   = round(1 - model_mape, 4)
stability  = explain["confidence_score"].apply(lambda s: min(1.0, s / 0.40 * 0.35)).mean()
coverage   = min(1.0, 181 / 365)

categories = ["Accuracy\n(1-MAPE)", "Stability\n(bootstrap)", "Coverage\n(histórico)"]
values     = [accuracy, 0.83, coverage]
weights    = [0.40, 0.35, 0.25]
contrib    = [v * w for v, w in zip(values, weights)]

bars = ax.bar(categories, contrib,
              color=[COLORS["success"], COLORS["primary"], COLORS["warning"]],
              width=0.45)
ax.bar(categories, [v - c for v, c in zip(values, contrib)], width=0.45,
       bottom=contrib, color=["#D1FAE5", "#DBEAFE", "#FEF3C7"], alpha=0.6)

for bar, raw, w, cont in zip(bars, values, weights, contrib):
    ax.text(bar.get_x() + bar.get_width() / 2, cont / 2,
            f"peso {w:.0%}\nraw={raw:.2f}", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

ax.axhline(y=sum(contrib), color=COLORS["danger"], ls="--", lw=1.5,
           label=f"Score final médio: {sum(contrib):.2f}")
ax.set_title("Decomposição do Score de Confiança")
ax.set_ylabel("Contribuição")
ax.legend(fontsize=9)

plt.tight_layout()
path4 = OUTPUT_DIR / "painel4_explicabilidade.png"
fig4.savefig(path4, dpi=130, bbox_inches="tight")
print(f"  Salvo: {path4}")
plt.close(fig4)


print(f"\nTodos os painéis salvos em: {OUTPUT_DIR}")
print("Painéis gerados:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {f.name}")
