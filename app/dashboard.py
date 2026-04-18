"""
RGM Pipeline — Dashboard Streamlit

Visualiza as projeções de demanda e a grade de campanhas prescritiva
geradas pelo sistema multi-agentes.

Execução:
    streamlit run dashboard.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="RGM Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR     = Path(__file__).parent
DB_PATH      = BASE_DIR / "data" / "raw"  / "rgm_database.db"
PROCESSED    = BASE_DIR / "data" / "processed"

PALETTE = {
    "blue":   "#2563EB",
    "green":  "#16A34A",
    "amber":  "#D97706",
    "red":    "#DC2626",
    "purple": "#7C3AED",
    "gray":   "#6B7280",
}

# ── CSS customizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    div[data-testid="metric-container"] label { font-size: .78rem; color: #6B7280; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.5rem; font-weight: 700;
    }
    .section-title {
        font-size: 1rem; font-weight: 600; color: #374151;
        border-left: 4px solid #2563EB;
        padding-left: .6rem; margin: 1rem 0 .5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Carga de dados (cacheada) ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_db_tables():
    if not DB_PATH.exists():
        return {k: pd.DataFrame() for k in
                ["transactions", "campaigns", "products", "stores", "uplift"]}
    with sqlite3.connect(DB_PATH) as conn:
        txn   = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])
        camps = pd.read_sql("SELECT * FROM campaigns",    conn,
                            parse_dates=["start_date", "end_date"])
        prods = pd.read_sql("SELECT * FROM products",     conn)
        stores= pd.read_sql("SELECT * FROM stores",       conn)
        uplift= pd.read_sql("SELECT * FROM uplift_metrics", conn)
    txn = txn.merge(prods[["product_id", "category"]], on="product_id", how="left")
    return {"transactions": txn, "campaigns": camps,
            "products": prods, "stores": stores, "uplift": uplift}


@st.cache_data(ttl=300)
def load_processed():
    out = {}
    for name, fname in [
        ("simulation",   "demand_simulation.parquet"),
        ("grid",         "campaign_grid.parquet"),
        ("explanations", "campaign_explanations.parquet"),
        ("baseline",     "causal_baseline.parquet"),
    ]:
        p = PROCESSED / fname
        out[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return out


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("RGM Pipeline")
    st.caption("Revenue Growth Management")
    st.divider()

    db      = load_db_tables()
    proc    = load_processed()
    sim     = proc["simulation"]
    grid    = proc["grid"]
    explain = proc["explanations"]
    txn     = db["transactions"]
    prods   = db["products"]

    pipeline_ready = not grid.empty and not sim.empty

    if not pipeline_ready:
        st.warning("Execute o pipeline primeiro:\n```\npython run_module1.py\npython run_module2_3.py\n```")

    st.subheader("Filtros")

    all_products = sorted(sim["product_id"].unique()) if not sim.empty else []
    all_stores   = sorted(sim["store_id"].unique())   if not sim.empty else []
    all_discounts= sorted(sim["discount_pct"].unique()) if not sim.empty else []

    sel_products = st.multiselect(
        "Produtos", all_products,
        default=all_products[:5] if all_products else [],
        placeholder="Todos",
    )
    sel_stores = st.multiselect(
        "Lojas", all_stores,
        default=all_stores[:3] if all_stores else [],
        placeholder="Todas",
    )
    sel_discounts = st.multiselect(
        "Cenários de desconto",
        [f"{int(d*100)}%" for d in all_discounts],
        default=[f"{int(d*100)}%" for d in all_discounts],
    )
    disc_map = {f"{int(d*100)}%": d for d in all_discounts}
    sel_disc_vals = [disc_map[d] for d in sel_discounts if d in disc_map]

    top_n = st.slider("Top N campanhas na grade", 5, 40, 15)
    st.divider()
    st.caption("Dados: SQLite local  \nModelo: XGBoost + ILP")


# ── Aplicar filtros ──────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if sel_products and "product_id" in df.columns:
        df = df[df["product_id"].isin(sel_products)]
    if sel_stores and "store_id" in df.columns:
        df = df[df["store_id"].isin(sel_stores)]
    if sel_disc_vals and "discount_pct" in df.columns:
        df = df[df["discount_pct"].isin(sel_disc_vals)]
    return df


sim_f  = apply_filters(sim)
grid_f = apply_filters(grid).nlargest(top_n, "net_margin") if not grid.empty else pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.title("📊 RGM Pipeline — Dashboard")
st.caption("Sistema Multi-Agentes para Revenue Growth Management")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Visão Geral",
    "📈 Previsão de Demanda",
    "🎯 Grade de Campanhas",
    "🔍 Explicabilidade",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — VISÃO GERAL
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    total_txn   = len(txn) if not txn.empty else 0
    total_rev   = txn["revenue"].sum() if not txn.empty else 0
    total_camps = len(grid) if not grid.empty else 0
    total_margin= grid["net_margin"].sum() if not grid.empty else 0
    total_budget= grid["campaign_cost"].sum() if not grid.empty else 0
    roi         = total_margin / total_budget if total_budget > 0 else 0

    c1.metric("Transações históricas",  f"{total_txn:,.0f}")
    c2.metric("Receita total (hist.)",  f"R$ {total_rev/1e6:.1f}M")
    c3.metric("Campanhas na grade",     f"{total_camps}")
    c4.metric("Margem total esperada",  f"R$ {total_margin:,.0f}")
    c5.metric("ROI médio",              f"{roi:.1f}×",
              delta=f"verba R$ {total_budget:,.0f}")

    st.markdown('<div class="section-title">Receita Diária Histórica</div>',
                unsafe_allow_html=True)

    if not txn.empty:
        daily = txn.groupby("date").agg(
            revenue=("revenue", "sum"),
            volume=("volume",  "sum"),
        ).reset_index()
        daily["revenue_ma30"] = daily["revenue"].rolling(30).mean()
        daily["has_campaign"] = txn[txn["campaign_id"].notna()].groupby("date").size().reindex(daily["date"]).notna().values

        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(go.Scatter(
            x=daily["date"], y=daily["revenue"] / 1000,
            name="Receita diária", fill="tozeroy",
            line=dict(color=PALETTE["blue"], width=1),
            fillcolor="rgba(37,99,235,0.08)",
        ), secondary_y=False)
        fig_ts.add_trace(go.Scatter(
            x=daily["date"], y=daily["revenue_ma30"] / 1000,
            name="Média 30d", line=dict(color=PALETTE["blue"], width=2.5, dash="solid"),
        ), secondary_y=False)
        fig_ts.add_trace(go.Bar(
            x=daily["date"], y=daily["volume"],
            name="Volume", marker_color="rgba(124,58,237,0.15)",
            opacity=0.6,
        ), secondary_y=True)
        fig_ts.update_layout(
            height=320, margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
        )
        fig_ts.update_yaxes(title_text="Receita (R$ mil)", secondary_y=False)
        fig_ts.update_yaxes(title_text="Volume (un.)", secondary_y=True)
        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown('<div class="section-title">Performance por Categoria</div>',
                unsafe_allow_html=True)

    if not txn.empty and "category" in txn.columns:
        cat_agg = txn.groupby("category").agg(
            receita=("revenue",    "sum"),
            volume=("volume",      "sum"),
            margem=("margin",      "sum"),
            n_txn= ("transaction_id", "count"),
        ).reset_index()
        cat_agg["margin_pct"] = cat_agg["margem"] / cat_agg["receita"]

        col_a, col_b = st.columns(2)
        with col_a:
            fig_cat = px.bar(
                cat_agg.sort_values("receita", ascending=True),
                x="receita", y="category", orientation="h",
                title="Receita por Categoria (R$)",
                color="margin_pct",
                color_continuous_scale=["#FEF3C7", "#16A34A"],
                labels={"receita": "Receita (R$)", "category": "",
                        "margin_pct": "Margem %"},
                text_auto=".2s",
            )
            fig_cat.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_b:
            fig_treemap = px.treemap(
                cat_agg,
                path=["category"],
                values="volume",
                color="margin_pct",
                color_continuous_scale=["#FEF3C7", "#16A34A"],
                title="Volume por Categoria (tamanho) / Margem % (cor)",
            )
            fig_treemap.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig_treemap, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREVISÃO DE DEMANDA
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if sim_f.empty:
        st.info("Nenhum dado de simulação disponível para os filtros selecionados.")
    else:
        st.markdown('<div class="section-title">Volume Previsto por Cenário de Desconto</div>',
                    unsafe_allow_html=True)

        agg_disc = (
            sim_f.groupby("discount_pct")
            .agg(
                vol_medio=("predicted_volume",  "mean"),
                vol_total=("predicted_volume",  "sum"),
                margem_media=("predicted_margin", "mean"),
                receita_media=("predicted_revenue","mean"),
            )
            .reset_index()
        )
        agg_disc["desconto_label"] = agg_disc["discount_pct"].apply(
            lambda x: f"{int(x*100)}%"
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_vol = px.bar(
                agg_disc,
                x="desconto_label", y="vol_medio",
                title="Volume Médio Previsto por Desconto",
                color="desconto_label",
                color_discrete_sequence=[PALETTE["green"], PALETTE["blue"],
                                          PALETTE["amber"], PALETTE["red"]],
                text_auto=".0f",
                labels={"desconto_label": "Desconto", "vol_medio": "Volume médio"},
            )
            fig_vol.update_layout(height=340, showlegend=False,
                                  margin=dict(t=50, b=10))
            st.plotly_chart(fig_vol, use_container_width=True)

        with col2:
            fig_margin = px.bar(
                agg_disc,
                x="desconto_label", y="margem_media",
                title="Margem Média Prevista por Desconto (R$)",
                color="desconto_label",
                color_discrete_sequence=[PALETTE["green"], PALETTE["blue"],
                                          PALETTE["amber"], PALETTE["red"]],
                text_auto=".0f",
                labels={"desconto_label": "Desconto", "margem_media": "Margem média (R$)"},
            )
            fig_margin.update_layout(height=340, showlegend=False,
                                     margin=dict(t=50, b=10))
            st.plotly_chart(fig_margin, use_container_width=True)

        st.markdown('<div class="section-title">Margem × Volume por Cenário</div>',
                    unsafe_allow_html=True)

        fig_scatter = px.scatter(
            sim_f.sample(min(2000, len(sim_f)), random_state=42),
            x="predicted_volume",
            y="predicted_margin",
            color=sim_f.sample(min(2000, len(sim_f)), random_state=42)["discount_pct"].apply(
                lambda x: f"{int(x*100)}%"
            ),
            size="predicted_revenue",
            size_max=18,
            opacity=0.65,
            title="Margem Prevista × Volume Previsto (por cenário de desconto)",
            labels={
                "predicted_volume":  "Volume previsto (un.)",
                "predicted_margin":  "Margem prevista (R$)",
                "color":             "Desconto",
            },
            color_discrete_sequence=[PALETTE["green"], PALETTE["blue"],
                                      PALETTE["amber"], PALETTE["red"]],
        )
        fig_scatter.update_layout(height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown('<div class="section-title">Top Produtos por Volume Previsto (desconto 10%)</div>',
                    unsafe_allow_html=True)

        sim_10 = sim_f[sim_f["discount_pct"] == 0.10].groupby("product_id").agg(
            vol=("predicted_volume", "sum"),
            margem=("predicted_margin", "sum"),
        ).reset_index().nlargest(15, "vol")

        if not sim_10.empty:
            fig_top_prod = px.bar(
                sim_10.sort_values("vol"),
                x="vol", y="product_id", orientation="h",
                color="margem",
                color_continuous_scale=["#DBEAFE", "#1D4ED8"],
                text_auto=".0f",
                title="Volume Total Previsto — Top 15 Produtos (desconto 10%)",
                labels={"vol": "Volume total", "product_id": "Produto",
                        "margem": "Margem (R$)"},
            )
            fig_top_prod.update_layout(height=420, margin=dict(t=50, b=10))
            st.plotly_chart(fig_top_prod, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRADE DE CAMPANHAS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if grid_f.empty:
        st.info("Grade de campanhas não disponível. Verifique os filtros ou execute o pipeline.")
    else:
        # KPIs da grade filtrada
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Campanhas (filtradas)",  f"{len(grid_f)}")
        c2.metric("Margem total",           f"R$ {grid_f['net_margin'].sum():,.0f}")
        c3.metric("Verba utilizada",        f"R$ {grid_f['campaign_cost'].sum():,.0f}")
        roi_f = (grid_f["net_margin"].sum() / grid_f["campaign_cost"].sum()
                 if grid_f["campaign_cost"].sum() > 0 else 0)
        c4.metric("ROI (filtrado)",         f"{roi_f:.1f}×")

        st.markdown('<div class="section-title">Top Campanhas por Margem Líquida</div>',
                    unsafe_allow_html=True)

        grid_disp = grid_f.copy()
        grid_disp["label"] = grid_disp["product_id"] + " · " + grid_disp["store_id"]
        grid_disp["desconto"] = grid_disp["discount_pct"].apply(lambda x: f"{int(x*100)}%")

        fig_grid = px.bar(
            grid_disp.sort_values("net_margin"),
            x="net_margin", y="label",
            color="desconto",
            orientation="h",
            title=f"Top {top_n} Campanhas — Margem Líquida (R$)",
            labels={"net_margin": "Margem líquida (R$)", "label": "Produto · Loja",
                    "desconto": "Desconto"},
            color_discrete_map={
                "10%": PALETTE["green"], "20%": PALETTE["blue"],
                "30%": PALETTE["amber"], "40%": PALETTE["red"],
            },
            text_auto=".0f",
        )
        fig_grid.update_layout(height=max(350, len(grid_disp) * 28),
                               margin=dict(t=50, b=10))
        st.plotly_chart(fig_grid, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-title">Margem × Custo (isocurvas ROI)</div>',
                        unsafe_allow_html=True)
            fig_roi = px.scatter(
                grid_disp,
                x="campaign_cost", y="net_margin",
                color="desconto",
                size="predicted_volume",
                size_max=20,
                hover_data=["product_id", "store_id"],
                title="Margem Líquida × Custo da Campanha",
                labels={"campaign_cost": "Custo da campanha (R$)",
                        "net_margin": "Margem líquida (R$)", "desconto": "Desconto"},
                color_discrete_map={
                    "10%": PALETTE["green"], "20%": PALETTE["blue"],
                    "30%": PALETTE["amber"], "40%": PALETTE["red"],
                },
            )
            # Isocurvas de ROI
            import numpy as np
            x_range = np.linspace(0, grid_disp["campaign_cost"].max() * 1.1, 50)
            for roi_val, color, dash in [(5, "#16A34A", "dash"), (7, "#D97706", "dot")]:
                fig_roi.add_trace(go.Scatter(
                    x=x_range, y=x_range * roi_val,
                    mode="lines", name=f"ROI {roi_val}×",
                    line=dict(color=color, dash=dash, width=1.5),
                ))
            fig_roi.update_layout(height=340, margin=dict(t=50, b=10))
            st.plotly_chart(fig_roi, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">Distribuição por Desconto</div>',
                        unsafe_allow_html=True)
            disc_dist = grid_disp.groupby("desconto").agg(
                n=("net_margin", "count"),
                margem_total=("net_margin", "sum"),
                verba=("campaign_cost", "sum"),
            ).reset_index()

            fig_pie = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "pie"}]],
                subplot_titles=["Nº de campanhas", "Margem total"],
            )
            colors_pie = [PALETTE["green"], PALETTE["blue"],
                          PALETTE["amber"], PALETTE["red"]]
            fig_pie.add_trace(go.Pie(
                labels=disc_dist["desconto"], values=disc_dist["n"],
                hole=0.45, marker_colors=colors_pie, showlegend=True,
            ), row=1, col=1)
            fig_pie.add_trace(go.Pie(
                labels=disc_dist["desconto"], values=disc_dist["margem_total"],
                hole=0.45, marker_colors=colors_pie, showlegend=False,
            ), row=1, col=2)
            fig_pie.update_layout(height=340, margin=dict(t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-title">Tabela da Grade Prescritiva</div>',
                    unsafe_allow_html=True)

        display_cols = {
            "product_id":       "Produto",
            "store_id":         "Loja",
            "desconto":         "Desconto",
            "predicted_volume": "Volume Prev.",
            "net_margin":       "Margem Líq. (R$)",
            "campaign_cost":    "Custo Camp. (R$)",
        }
        grid_table = grid_disp[[c for c in display_cols if c in grid_disp.columns]].rename(
            columns=display_cols
        )
        st.dataframe(
            grid_table.style
            .format({"Margem Líq. (R$)": "R$ {:,.0f}",
                     "Custo Camp. (R$)": "R$ {:,.0f}",
                     "Volume Prev.": "{:,.0f}"})
            .background_gradient(subset=["Margem Líq. (R$)"],
                                  cmap="Greens"),
            use_container_width=True,
            height=min(450, (len(grid_table) + 1) * 38),
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLICABILIDADE
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Importância Global das Features (SHAP)</div>',
                unsafe_allow_html=True)

    shap_data = pd.DataFrame({
        "feature": ["ma_30d", "ma_7d", "dayofweek", "lag_7d", "lag_90d",
                    "lag_30d", "is_weekend", "discount_pct", "month", "day_of_year"],
        "mean_abs_shap": [0.388, 0.328, 0.121, 0.029, 0.022,
                          0.018, 0.014, 0.010, 0.009, 0.008],
        "tipo": ["Tendência", "Tendência", "Calendário", "Lag", "Lag",
                 "Lag", "Calendário", "Promoção", "Calendário", "Calendário"],
    }).sort_values("mean_abs_shap", ascending=True)

    col_shap, col_conf = st.columns([3, 2])

    with col_shap:
        fig_shap = px.bar(
            shap_data,
            x="mean_abs_shap", y="feature", orientation="h",
            color="tipo",
            title="SHAP — Importância Média das Features",
            labels={"mean_abs_shap": "Mean |SHAP value|",
                    "feature": "", "tipo": "Tipo"},
            color_discrete_map={
                "Tendência":  PALETTE["blue"],
                "Lag":        PALETTE["purple"],
                "Calendário": PALETTE["amber"],
                "Promoção":   PALETTE["green"],
            },
            text=shap_data["mean_abs_shap"].apply(lambda x: f"{x:.3f}"),
        )
        fig_shap.update_layout(height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_conf:
        if not explain.empty:
            fig_conf_hist = px.histogram(
                explain,
                x="confidence_score",
                nbins=20,
                title="Distribuição do Score de Confiança",
                labels={"confidence_score": "Score de Confiança",
                        "count": "Nº de campanhas"},
                color_discrete_sequence=[PALETTE["blue"]],
            )
            mean_conf = explain["confidence_score"].mean()
            fig_conf_hist.add_vline(
                x=mean_conf,
                line_dash="dash", line_color=PALETTE["red"],
                annotation_text=f"Média: {mean_conf:.3f}",
                annotation_position="top right",
            )
            fig_conf_hist.update_layout(height=360, margin=dict(t=50, b=10))
            st.plotly_chart(fig_conf_hist, use_container_width=True)
        else:
            st.info("Dados de explicabilidade não disponíveis.")

    if not explain.empty:
        st.markdown('<div class="section-title">Decomposição do Score de Confiança</div>',
                    unsafe_allow_html=True)

        col_decomp, col_label = st.columns([2, 1])
        with col_decomp:
            decomp = pd.DataFrame({
                "Componente": ["Accuracy (1-MAPE)", "Stability (bootstrap)", "Coverage (histórico)"],
                "Valor Raw":  [0.840, 0.947, 0.496],
                "Peso":       [0.40,  0.35,  0.25],
            })
            decomp["Contribuição"] = decomp["Valor Raw"] * decomp["Peso"]

            fig_decomp = go.Figure()
            colors_decomp = [PALETTE["green"], PALETTE["blue"], PALETTE["amber"]]
            for i, row in decomp.iterrows():
                fig_decomp.add_trace(go.Bar(
                    name=row["Componente"],
                    x=[row["Componente"]],
                    y=[row["Contribuição"]],
                    marker_color=colors_decomp[i],
                    text=f"raw={row['Valor Raw']:.2f}<br>peso={row['Peso']:.0%}",
                    textposition="inside",
                    textfont=dict(color="white", size=11),
                ))
            fig_decomp.add_hline(
                y=decomp["Contribuição"].sum(),
                line_dash="dot", line_color=PALETTE["red"],
                annotation_text=f"Score final: {decomp['Contribuição'].sum():.3f}",
            )
            fig_decomp.update_layout(
                title="Contribuição de Cada Componente",
                height=320, barmode="group",
                showlegend=False, margin=dict(t=50, b=10),
                yaxis_title="Contribuição",
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

        with col_label:
            st.markdown("#### Legenda")
            st.markdown("""
| Label | Faixa |
|-------|-------|
| 🟢 ALTA   | ≥ 0.80 |
| 🟡 MÉDIA  | ≥ 0.60 |
| 🔴 BAIXA  | < 0.60 |
""")
            if "confidence_label" in explain.columns:
                label_counts = explain["confidence_label"].value_counts()
                for label, cnt in label_counts.items():
                    icon = "🟢" if label == "ALTA" else "🟡" if label == "MÉDIA" else "🔴"
                    st.markdown(f"{icon} **{label}**: {cnt} campanhas")

        st.markdown('<div class="section-title">Explicações das Campanhas Recomendadas</div>',
                    unsafe_allow_html=True)

        exp_f = apply_filters(explain)
        if not exp_f.empty:
            for _, row in exp_f.head(5).iterrows():
                conf = row.get("confidence_score", 0)
                label = row.get("confidence_label", "N/A")
                icon  = "🟢" if label == "ALTA" else "🟡" if label == "MÉDIA" else "🔴"
                with st.expander(
                    f"{icon} {row['product_id']} · {row['store_id']} "
                    f"| Desconto {int(row['discount_pct']*100)}% "
                    f"| Margem R$ {row['net_margin']:,.0f} "
                    f"| Confiança {conf:.0%}"
                ):
                    st.write(row.get("explanation", "—"))
                    col_x1, col_x2, col_x3 = st.columns(3)
                    col_x1.metric("Score de Confiança", f"{conf:.3f}")
                    col_x2.metric("Margem líquida",
                                  f"R$ {row['net_margin']:,.0f}")
                    col_x3.metric("SHAP drivers",
                                  row.get("top_shap_drivers", "—")[:30] + "…")
        else:
            st.info("Selecione produtos/lojas na sidebar para ver as explicações.")
