import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

COL_DIAG = "Diagnostico"
COL_DESC = "Descripcion"
COL_CAPITULO = "Capitulo"
COL_TIPO = "Tipo"
COL_CATEGORIA = "Categoria"
COL_TOTAL_EPIS = "Total episodios"
COL_TOTAL_DIAS = "Total dias"
COL_DIAS_GT15 = "Dias IT >15 dias"
COL_DURACION_MEDIA = "Tod durac media"
COL_PCT_EPI_LT15 = "%TotEpis<15dias"
COL_PCT_EPI_GT15 = "%TotEpis>15dias"
COL_PCT_DIAS_LT15 = "%Totdias<15d"
COL_PCT_DIAS_GT15 = "%Totdias>15d"

RENAME_MAP = {
    "Diagnóstico": COL_DIAG,
    "Descripción": COL_DESC,
    "Capítulo": COL_CAPITULO,
    "Total días": COL_TOTAL_DIAS,
    "Días IT >15 días": COL_DIAS_GT15,
    "Tod durac media": COL_DURACION_MEDIA,
    "%TotEpis<15dias": COL_PCT_EPI_LT15,
    "%TotEpis>15dias": COL_PCT_EPI_GT15,
    "%Totdias<15d": COL_PCT_DIAS_LT15,
    "%Totdias>15d": COL_PCT_DIAS_GT15,
}

SIZE_OPTIONS = {
    "Total de episodios": COL_TOTAL_EPIS,
    "Dias totales": COL_TOTAL_DIAS,
    "Dias >15": COL_DIAS_GT15,
}

COLOR_OPTIONS = {
    "Sin color": None,
    "Capitulo": COL_CAPITULO,
    "Tipo": COL_TIPO,
}


def ratio(value: float, total: float) -> float:
    return (value / total * 100.0) if total else 0.0

@st.cache_data
def cargar_datos() -> pd.DataFrame:
    df = pd.read_excel("Dx_consolidadov3.xlsx")
    df = df.rename(columns=RENAME_MAP)
    df = df[df[COL_DIAG].notna()].copy()
    return df

def clasificar(df: pd.DataFrame, epis_min: int, dur_max: float, pct_corto: float) -> pd.DataFrame:
    df = df.copy()
    df[COL_CATEGORIA] = "Incluido"
    df.loc[df[COL_TOTAL_EPIS] <= epis_min, COL_CATEGORIA] = "ExclNum"
    mask = df[COL_CATEGORIA] != "ExclNum"
    df.loc[
        mask
        & (df[COL_DURACION_MEDIA] < dur_max)
        & (df[COL_PCT_EPI_LT15] > pct_corto),
        COL_CATEGORIA,
    ] = "ExclDur"
    return df

def obtener_metricas(df: pd.DataFrame) -> dict:
    metricas = {}
    for categoria in ["Incluido", "ExclNum", "ExclDur"]:
        subset = df[df[COL_CATEGORIA] == categoria]
        metricas[categoria] = {
            "diagnosticos": int(subset.shape[0]),
            "episodios": int(subset[COL_TOTAL_EPIS].sum()),
            "dias": int(subset[COL_TOTAL_DIAS].sum()),
            "dias_mayor_15": int(subset[COL_DIAS_GT15].sum()),
        }
    metricas["Total"] = {
        clave: metricas["Incluido"][clave]
        + metricas["ExclNum"][clave]
        + metricas["ExclDur"][clave]
        for clave in ["diagnosticos", "episodios", "dias", "dias_mayor_15"]
    }
    return metricas


df_base = cargar_datos()

st.set_page_config(page_title="Diagnosticos IBERMUTUA", layout="wide")
st.title("Analisis de Filtrado de Diagnosticos IBERMUTUA")
st.markdown("### Visualizacion del impacto de los criterios de exclusion")

st.sidebar.header("Parametros de filtrado")
max_episodios = int(df_base[COL_TOTAL_EPIS].max())

episodios_min = st.sidebar.slider(
    "Episodios minimos para gestion activa",
    min_value=0,
    max_value=max_episodios,
    value=500,
    step=50,
)

duracion_max = st.sidebar.slider(
    "Duracion media maxima (dias)",
    min_value=0.0,
    max_value=float(df_base[COL_DURACION_MEDIA].max()),
    value=15.0,
    step=0.5,
)

pct_cortos = st.sidebar.slider(
    "% episodios que terminan antes de 15 dias",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=0.5,
) / 100.0

st.sidebar.caption(
    "Los diagnosticos con episodios menores o iguales al umbral pasan a ExclNum. "
    "El resto pasa a ExclDur si su duracion media es inferior al umbral y el porcentaje de episodios cortos supera el limite."
)

df = clasificar(df_base, episodios_min, duracion_max, pct_cortos)
metricas = obtener_metricas(df)

porc_diag_incluidos = ratio(metricas["Incluido"]["diagnosticos"], metricas["Total"]["diagnosticos"])
porc_episodios_incluidos = ratio(metricas["Incluido"]["episodios"], metricas["Total"]["episodios"])
porc_dias_retenidos = ratio(metricas["Incluido"]["dias"], metricas["Total"]["dias"])
porc_dias_mas_15 = ratio(metricas["Incluido"]["dias_mayor_15"], metricas["Total"]["dias_mayor_15"])

st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Diagnosticos", f"{metricas['Total']['diagnosticos']:,}")
with col2:
    st.metric("Diagnosticos Incluidos", f"{metricas['Incluido']['diagnosticos']:,}", delta=f"{porc_diag_incluidos:.1f}% del total")
with col3:
    st.metric("Episodios Retenidos", f"{metricas['Incluido']['episodios']:,}", delta=f"{porc_episodios_incluidos:.1f}% del total")
with col4:
    st.metric("% Dias IT Retenidos", f"{porc_dias_retenidos:.1f}%", delta=f"Con {porc_diag_incluidos:.1f}% de diagnosticos")
with col5:
    st.metric("% Dias >15 Retenidos", f"{porc_dias_mas_15:.1f}%", delta="Impacto economico directo")

st.markdown("---")
st.markdown("### Graficos Funnel - Proceso de Filtrado")

fig_funnel = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Numero de Diagnosticos",
        "Total de Episodios",
        "Total de Dias IT",
        "Dias IT >15 (Coste Mutua)",
    ),
    specs=[[{"type": "funnel"}, {"type": "funnel"}], [{"type": "funnel"}, {"type": "funnel"}]],
    vertical_spacing=0.15,
    horizontal_spacing=0.12,
)

funnel_config = {
    "diagnosticos": (1, 1),
    "episodios": (1, 2),
    "dias": (2, 1),
    "dias_mayor_15": (2, 2),
}

for campo, (row, col) in funnel_config.items():
    total = metricas["Total"][campo]
    excl_num = metricas["ExclNum"][campo]
    excl_dur = metricas["ExclDur"][campo]
    incluido = metricas["Incluido"][campo]
    valores = [total, total - excl_num, total - excl_num - excl_dur, incluido]
    fig_funnel.add_trace(
        go.Funnel(
            y=[
                "Total Inicial",
                f"Tras excluir volumen<br><span style='color:red'>(-{excl_num:,})</span>",
                f"Tras excluir duracion<br><span style='color:red'>(-{excl_dur:,})</span>",
                "INCLUIDOS<br>Gestion Activa",
            ],
            x=valores,
            textposition="inside",
            textinfo="value+percent initial",
            opacity=0.85,
            marker={"color": ["#636EFA", "#FFA15A", "#FECB52", "#00CC96"]},
            connector={"line": {"color": "royalblue", "width": 2}},
            texttemplate="<b>%{value:,.0f}</b><br>%{percentInitial}",
            hovertemplate="%{label}<br>Cantidad: %{value:,.0f}<br>% del total: %{percentInitial}<extra></extra>",
        ),
        row=row,
        col=col,
    )

fig_funnel.update_layout(
    height=800,
    showlegend=False,
    title_text="Proceso de Filtrado: Del Universo Total a la Gestion Activa",
    title_x=0.5,
    title_font_size=18,
)

st.plotly_chart(fig_funnel, use_container_width=True)
st.markdown("---")
st.markdown("### Eficiencia del Filtrado - Principio de Pareto")

fig_eficiencia = go.Figure(
    data=[
        go.Bar(
            x=["Diagnosticos", "Episodios", "Dias Totales", "Dias >15"],
            y=[porc_diag_incluidos, porc_episodios_incluidos, porc_dias_retenidos, porc_dias_mas_15],
            text=[
                f"{porc_diag_incluidos:.1f}%",
                f"{porc_episodios_incluidos:.1f}%",
                f"{porc_dias_retenidos:.1f}%",
                f"{porc_dias_mas_15:.1f}%",
            ],
            textposition="outside",
            marker_color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        )
    ]
)

fig_eficiencia.add_hline(
    y=80,
    line_dash="dash",
    line_color="gray",
    annotation_text="Objetivo 80%",
    annotation_position="right",
)

fig_eficiencia.update_layout(
    title={"text": "Porcentaje Retenido con Diagnosticos Incluidos", "x": 0.5},
    yaxis_title="% Retenido",
    xaxis_title="Metrica",
    height=500,
    showlegend=False,
)

st.plotly_chart(fig_eficiencia, use_container_width=True)
st.markdown("---")
st.markdown("### Matriz de priorizacion de diagnosticos")

st.sidebar.markdown("---")
st.sidebar.subheader("Parametros matriz de priorizacion")

df_cuadrante = pd.DataFrame()
df_incluidos = df[df[COL_CATEGORIA] == "Incluido"].copy()
if df_incluidos.empty:
    st.warning("No hay diagnosticos incluidos con los parametros actuales. Ajusta los filtros iniciales.")
else:
    df_incluidos = df_incluidos.assign(
        share_episodios_pct=df_incluidos[COL_TOTAL_EPIS] / metricas["Incluido"]["episodios"] * 100 if metricas["Incluido"]["episodios"] else 0,
        share_dias15_pct=df_incluidos[COL_DIAS_GT15] / metricas["Incluido"]["dias_mayor_15"] * 100 if metricas["Incluido"]["dias_mayor_15"] else 0,
        duracion_media=df_incluidos[COL_DURACION_MEDIA],
    )

    size_field_key = st.sidebar.selectbox("Tamano burbuja", list(SIZE_OPTIONS.keys()), index=0)
    size_field = SIZE_OPTIONS[size_field_key]
    color_choice = st.sidebar.selectbox("Color por", list(COLOR_OPTIONS.keys()), index=1)
    color_field = COLOR_OPTIONS[color_choice]

    dur_min = float(df_incluidos["duracion_media"].min())
    dur_max = float(df_incluidos["duracion_media"].max())
    if dur_max <= dur_min:
        dur_max = dur_min + 1.0
    dur_default = float(df_incluidos["duracion_media"].median())
    duracion_umbral = st.sidebar.slider("Umbral duracion media (dias)", min_value=dur_min, max_value=dur_max, value=dur_default, step=0.5)

    share_max = float(df_incluidos["share_episodios_pct"].max())
    if share_max <= 0:
        share_max = 1.0
    share_default = float(df_incluidos["share_episodios_pct"].median())
    share_umbral_pct = st.sidebar.slider("Umbral % de episodios", min_value=0.0, max_value=share_max, value=share_default, step=0.1)

    scatter_kwargs = dict(
        data_frame=df_incluidos,
        x="duracion_media",
        y="share_episodios_pct",
        size=size_field,
        hover_name=COL_DIAG,
        hover_data={
            COL_TOTAL_EPIS: ":,",
            "share_episodios_pct": ":.2f",
            "share_dias15_pct": ":.2f",
            "duracion_media": ":.1f",
            COL_DIAS_GT15: ":,",
        },
        size_max=60,
    )
    if color_field is not None:
        scatter_kwargs["color"] = df_incluidos[color_field]

    fig_prior = px.scatter(**scatter_kwargs)
    fig_prior.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color="#333")))
    fig_prior.update_layout(
        title="Priorizar diagnosticos incluidos",
        xaxis_title="Duracion media del episodio (dias)",
        yaxis_title="% del total de episodios (diagnosticos incluidos)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=60),
    )
    fig_prior.update_yaxes(ticksuffix="%")
    fig_prior.add_vline(x=duracion_umbral, line_dash="dash", line_color="gray", annotation_text="Umbral duracion", annotation_position="top left")
    fig_prior.add_hline(y=share_umbral_pct, line_dash="dash", line_color="gray", annotation_text="Umbral % episodios", annotation_position="bottom right")
    st.plotly_chart(fig_prior, use_container_width=True)

    df_cuadrante = df_incluidos[
        (df_incluidos["duracion_media"] >= duracion_umbral)
        & (df_incluidos["share_episodios_pct"] >= share_umbral_pct)
    ].copy()

    st.markdown("#### Diagnosticos priorizados (cuadrante)")
    if df_cuadrante.empty:
        st.info("Ningun diagnostico cumple simultaneamente los umbrales de duracion y porcentaje de episodios.")
    else:
        diag_sel = int(df_cuadrante.shape[0])
        epis_sel = int(df_cuadrante[COL_TOTAL_EPIS].sum())
        dias_sel = int(df_cuadrante[COL_TOTAL_DIAS].sum())
        dias15_sel = int(df_cuadrante[COL_DIAS_GT15].sum())

        total_diag_incl = metricas["Incluido"]["diagnosticos"]
        total_epi_incl = metricas["Incluido"]["episodios"]
        total_dias_incl = metricas["Incluido"]["dias"]
        total_dias15_incl = metricas["Incluido"]["dias_mayor_15"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Diagnosticos seleccionados", f"{diag_sel:,}", delta=f"{ratio(diag_sel, total_diag_incl):.1f}% de incluidos")
        with c2:
            st.metric("Episodios seleccionados", f"{epis_sel:,}", delta=f"{ratio(epis_sel, total_epi_incl):.1f}% de incluidos")
        with c3:
            st.metric("Dias IT seleccionados", f"{dias_sel:,}", delta=f"{ratio(dias_sel, total_dias_incl):.1f}% de incluidos")
        with c4:
            st.metric("Dias >15 seleccionados", f"{dias15_sel:,}", delta=f"{ratio(dias15_sel, total_dias15_incl):.1f}% de incluidos")

        df_tabla = df_cuadrante[[
            COL_DIAG,
            COL_CAPITULO,
            COL_TIPO,
            COL_TOTAL_EPIS,
            COL_TOTAL_DIAS,
            COL_DIAS_GT15,
            "duracion_media",
            "share_episodios_pct",
            "share_dias15_pct",
        ]].copy()
        df_tabla[COL_TOTAL_EPIS] = df_tabla[COL_TOTAL_EPIS].apply(lambda x: f"{int(x):,}")
        df_tabla[COL_TOTAL_DIAS] = df_tabla[COL_TOTAL_DIAS].apply(lambda x: f"{int(x):,}")
        df_tabla[COL_DIAS_GT15] = df_tabla[COL_DIAS_GT15].apply(lambda x: f"{int(x):,}")
        df_tabla["duracion_media"] = df_tabla["duracion_media"].apply(lambda x: f"{x:.1f}")
        df_tabla["share_episodios_pct"] = df_tabla["share_episodios_pct"].apply(lambda x: f"{x:.2f}%")
        df_tabla["share_dias15_pct"] = df_tabla["share_dias15_pct"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(df_tabla.sort_values("share_episodios_pct", ascending=False), use_container_width=True, hide_index=True)
st.markdown("---")
st.markdown("### Tabla Resumen Detallada")

resumen_df = pd.DataFrame({
    'Categoria': ['Total', 'Incluido', 'ExclNum', 'ExclDur'],
    'Diagnosticos': [
        metricas['Total']['diagnosticos'],
        metricas['Incluido']['diagnosticos'],
        metricas['ExclNum']['diagnosticos'],
        metricas['ExclDur']['diagnosticos']
    ],
    'Episodios': [
        metricas['Total']['episodios'],
        metricas['Incluido']['episodios'],
        metricas['ExclNum']['episodios'],
        metricas['ExclDur']['episodios']
    ],
    'Dias Totales': [
        metricas['Total']['dias'],
        metricas['Incluido']['dias'],
        metricas['ExclNum']['dias'],
        metricas['ExclDur']['dias']
    ],
    'Dias >15': [
        metricas['Total']['dias_mayor_15'],
        metricas['Incluido']['dias_mayor_15'],
        metricas['ExclNum']['dias_mayor_15'],
        metricas['ExclDur']['dias_mayor_15']
    ]
})

for col in ['Episodios', 'Dias Totales', 'Dias >15']:
    total_valor = resumen_df[col].iloc[0]
    resumen_df[f'% {col}'] = (resumen_df[col] / total_valor * 100).round(1)

for col in ['Diagnosticos', 'Episodios', 'Dias Totales', 'Dias >15']:
    resumen_df[col] = resumen_df[col].apply(lambda x: f"{int(x):,}")

st.dataframe(resumen_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.info(
    """
**Criterios de exclusion aplicados:**
- **ExclNum**: Diagnosticos con menos de 500 episodios en 10 años
- **ExclDur**: Duracion media <15 dias y porcentaje de episodios <=15 dias > 90%
- **Incluido**: No cumple ningun criterio de exclusion (gestion prioritaria)
"""
)
