import pandas as pd
import numpy as np
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
COL_SHARE_TOTAL_EPIS = "%Totsobre total epis"
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

COLOR_OPTIONS = {
    "Sin color": None,
    "Capitulo": COL_CAPITULO,
    "Tipo": COL_TIPO,
}


def ratio(value: float, total: float) -> float:
    return (value / total * 100.0) if total else 0.0


def logit(p):
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def estimar_pct_esperado(duracion: pd.Series, pct: pd.Series, bins: int = 20) -> pd.Series:
    datos = pd.DataFrame({
        'duracion': duracion,
        'pct': pct,
    }).dropna()
    if datos.empty:
        return pd.Series(0.0, index=duracion.index)
    n_bins = min(bins, max(2, datos['duracion'].nunique()))
    datos['bin'] = pd.qcut(datos['duracion'], q=n_bins, labels=False, duplicates='drop')
    medias = datos.groupby('bin')['pct'].mean().sort_index()
    medias_monotona = medias.cummax()
    datos['esperado'] = datos['bin'].map(medias_monotona)
    esperado = datos['esperado']
    relleno = float(esperado.mean()) if not esperado.empty else 0.0
    return esperado.reindex(duracion.index).fillna(relleno)


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
    share_total_prop = df_incluidos[COL_SHARE_TOTAL_EPIS].astype(float).clip(1e-6, 1 - 1e-6)
    pct_gt15_prop = df_incluidos[COL_PCT_EPI_GT15].astype(float).clip(0.0, 1.0)
    df_incluidos = df_incluidos.assign(
        duracion_media=df_incluidos[COL_DURACION_MEDIA],
        share_total_prop=share_total_prop,
        share_total_pct=share_total_prop * 100,
        share_gt15_pct=pct_gt15_prop * 100,
        share_total_logit=logit(share_total_prop),
    )

    size_cap = float(df_incluidos[COL_TOTAL_DIAS].quantile(0.95))
    if size_cap <= 0:
        size_cap = float(df_incluidos[COL_TOTAL_DIAS].max() or 1.0)
    df_incluidos['size_dias'] = df_incluidos[COL_TOTAL_DIAS].clip(upper=size_cap)

    color_choice = st.sidebar.selectbox("Color por", list(COLOR_OPTIONS.keys()), index=1)
    color_field = COLOR_OPTIONS[color_choice]

    m_zona = st.sidebar.slider("Umbral zona muerta (m)", min_value=15.0, max_value=25.0, value=18.0, step=0.5)
    alpha_curv = st.sidebar.slider("a (curvatura %>15)", min_value=0.2, max_value=0.9, value=0.6, step=0.05)
    phi_peso = st.sidebar.slider("phi (peso base %>15)", min_value=0.0, max_value=0.8, value=0.4, step=0.05)
    psi_bonus = st.sidebar.slider("psi (bonus por lift)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    esperado_pct = estimar_pct_esperado(df_incluidos[COL_DURACION_MEDIA], pct_gt15_prop)
    lift_pct = np.clip(pct_gt15_prop - esperado_pct, 0.0, None)
    severidad_cruda = np.maximum(df_incluidos[COL_DURACION_MEDIA] - m_zona, 0.0)
    severidad_factor = (1 - phi_peso) + phi_peso * np.power(pct_gt15_prop, alpha_curv) + psi_bonus * lift_pct
    s1_valor = severidad_cruda * severidad_factor

    df_incluidos = df_incluidos.assign(
        esperado_pct=esperado_pct,
        lift_pct=lift_pct,
        S1_hibrida=s1_valor,
    )

    s1_series = df_incluidos['S1_hibrida']
    s1_min = float(s1_series.quantile(0.05))
    s1_max = float(s1_series.quantile(0.95))
    if s1_max <= s1_min:
        s1_max = s1_min + 0.1
    s1_default = float(s1_series.quantile(0.75))
    if s1_default < s1_min or s1_default > s1_max:
        s1_default = (s1_min + s1_max) / 2
    s1_step = max((s1_max - s1_min) / 100, 0.05)
    severidad_umbral = st.sidebar.slider("Umbral severidad (S1h)", min_value=s1_min, max_value=s1_max, value=float(round(s1_default, 3)), step=float(round(s1_step, 3)))

    share_pct_series = df_incluidos['share_total_pct']
    share_max = float(share_pct_series.quantile(0.95))
    if share_max <= 0:
        share_max = float(share_pct_series.max() or 0.1)
    share_default = float(share_pct_series.quantile(0.75))
    if share_default > share_max:
        share_default = share_max
    share_step = max(share_max / 200, 0.0005)
    share_umbral_pct = st.sidebar.slider("Umbral % del total de episodios", min_value=0.0, max_value=share_max, value=float(round(share_default, 4)), step=float(round(share_step, 4)))

    scatter_kwargs = dict(
        data_frame=df_incluidos,
        x='S1_hibrida',
        y='share_total_logit',
        size='size_dias',
        hover_data={
            COL_DIAG: True,
            COL_CAPITULO: True,
            COL_SHARE_TOTAL_EPIS: ':.4%',
            COL_PCT_EPI_GT15: ':.2%',
            COL_DURACION_MEDIA: ':.1f',
            COL_TOTAL_EPIS: ':,',
            COL_TOTAL_DIAS: ':,',
            'S1_hibrida': ':.2f',
            'esperado_pct': ':.2%',
            'lift_pct': ':.2%',
        },
        size_max=60,
    )
    if color_field is not None:
        scatter_kwargs['color'] = df_incluidos[color_field]

    fig_prior = px.scatter(**scatter_kwargs)
    fig_prior.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color='#333')))
    fig_prior.update_layout(
        title="Priorizar diagnosticos incluidos",
        xaxis_title="Severidad S1h = duracion media * [(1 - phi) + phi*(%>15)^a + psi*lift]",
        yaxis_title="% del total de episodios (logit)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=60),
        showlegend=False,
    )

    min_prop = float(df_incluidos['share_total_prop'].min())
    max_prop = float(df_incluidos['share_total_prop'].max())
    candidate_perc = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
    tick_props = candidate_perc[(candidate_perc / 100 >= min_prop) & (candidate_perc / 100 <= max_prop)] / 100
    if tick_props.size == 0:
        tick_props = np.array([min_prop, max_prop])
    tick_props = np.clip(tick_props, 1e-6, 1 - 1e-6)
    tick_vals = logit(tick_props)
    tick_text = [f"{prop * 100:.4f}%" if prop < 0.01 else f"{prop * 100:.2f}%" for prop in tick_props]
    fig_prior.update_yaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_text)

    fig_prior.add_vline(x=severidad_umbral, line_dash='dash', line_color='gray', annotation_text='Umbral S1h', annotation_position='top left')
    share_umbral_prop = share_umbral_pct / 100 if share_umbral_pct else 0.0
    share_umbral_logit = logit(share_umbral_prop) if share_umbral_prop else logit(1e-6)
    fig_prior.add_hline(y=share_umbral_logit, line_dash='dash', line_color='gray', annotation_text=f'Umbral % episodios ({share_umbral_pct:.4f}%)', annotation_position='bottom right')
    st.plotly_chart(fig_prior, use_container_width=True)

    df_cuadrante = df_incluidos[(df_incluidos['S1_hibrida'] >= severidad_umbral) & (df_incluidos['share_total_pct'] >= share_umbral_pct) & (df_incluidos[COL_TOTAL_EPIS] >= episodios_min)].copy()

    st.markdown("#### Diagnosticos priorizados (cuadrante)")
    if df_cuadrante.empty:
        st.info("Ningun diagnostico supera simultaneamente los umbrales de severidad S1h y % del total de episodios.")
    else:
        diag_sel = int(df_cuadrante.shape[0])
        epis_sel = int(df_cuadrante[COL_TOTAL_EPIS].sum())
        dias_sel = int(df_cuadrante[COL_TOTAL_DIAS].sum())
        dias15_sel = int(df_cuadrante[COL_DIAS_GT15].sum())

        total_diag = metricas['Total']['diagnosticos']
        total_epi = metricas['Total']['episodios']
        total_dias = metricas['Total']['dias']
        total_dias15 = metricas['Total']['dias_mayor_15']

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Diagnosticos priorizados", f"{diag_sel:,}", delta=f"{ratio(diag_sel, total_diag):.1f}% del total")
        with c2:
            st.metric("Episodios priorizados", f"{epis_sel:,}", delta=f"{ratio(epis_sel, total_epi):.1f}% del total")
        with c3:
            st.metric("Dias IT priorizados", f"{dias_sel:,}", delta=f"{ratio(dias_sel, total_dias):.1f}% del total")
        with c4:
            st.metric("Dias >15 priorizados", f"{dias15_sel:,}", delta=f"{ratio(dias15_sel, total_dias15):.1f}% del total")

        df_tabla = df_cuadrante[[
            COL_DIAG,
            COL_CAPITULO,
            COL_TIPO,
            COL_TOTAL_EPIS,
            COL_TOTAL_DIAS,
            COL_DIAS_GT15,
            'duracion_media',
            'share_gt15_pct',
            'esperado_pct',
            'lift_pct',
            'share_total_pct',
            'S1_hibrida',
        ]].copy()
        df_tabla = df_tabla.sort_values('share_total_pct', ascending=False)
        totales = {
            COL_DIAG: 'TOTAL',
            COL_CAPITULO: '',
            COL_TIPO: '',
            COL_TOTAL_EPIS: int(df_tabla[COL_TOTAL_EPIS].sum()),
            COL_TOTAL_DIAS: int(df_tabla[COL_TOTAL_DIAS].sum()),
            COL_DIAS_GT15: int(df_tabla[COL_DIAS_GT15].sum()),
            'duracion_media': df_tabla['duracion_media'].mean(),
            'share_gt15_pct': df_tabla['share_gt15_pct'].mean(),
            'share_total_pct': df_tabla['share_total_pct'].sum(),
            'esperado_pct': df_tabla['esperado_pct'].mean(),
            'lift_pct': df_tabla['lift_pct'].mean(),
            'S1_hibrida': df_tabla['S1_hibrida'].mean(),
        }
        df_tabla = pd.concat([df_tabla, pd.DataFrame([totales])], ignore_index=True)
        df_tabla_format = df_tabla.rename(columns={
            'duracion_media': 'Tod durac media',
            'share_gt15_pct': '%TotEpis>15dias (%)',
            'esperado_pct': 'E[%>15 | media]',
            'lift_pct': 'Lift %>15',
            'share_total_pct': '%Totsobre total epis (%)',
            'S1_hibrida': 'S1h',
        }).copy()
        df_tabla_format['%TotEpis>15dias (%)'] = (df_tabla_format['%TotEpis>15dias (%)'] * 100).round(2)
        df_tabla_format['E[%>15 | media]'] = (df_tabla_format['E[%>15 | media]'] * 100).round(2)
        df_tabla_format['Lift %>15'] = (df_tabla_format['Lift %>15'] * 100).round(2)
        df_tabla_format['%Totsobre total epis (%)'] = df_tabla_format['%Totsobre total epis (%)'].round(4)
        df_tabla_format['Tod durac media'] = df_tabla_format['Tod durac media'].round(1)
        df_tabla_format['S1h'] = df_tabla_format['S1h'].round(2)
        for col in [COL_TOTAL_EPIS, COL_TOTAL_DIAS, COL_DIAS_GT15]:
            df_tabla_format[col] = df_tabla_format[col].astype('Int64')
        st.dataframe(df_tabla_format, use_container_width=True, hide_index=True)

        tipo_norm = df_incluidos[COL_TIPO].astype(str).str.strip().str.lower()
        restantes_mask = (tipo_norm == 'principal') & (~df_incluidos.index.isin(df_cuadrante.index))
        df_principal_fuera = df_incluidos[restantes_mask][[
            COL_DIAG,
            COL_CAPITULO,
            COL_TOTAL_EPIS,
            COL_TOTAL_DIAS,
            COL_DIAS_GT15,
            'duracion_media',
            'share_gt15_pct',
            'esperado_pct',
            'lift_pct',
            'share_total_pct',
            'S1_hibrida',
        ]].copy()

        st.markdown('#### Diagnosticos PRINCIPAL fuera del cuadrante')
        if df_principal_fuera.empty:
            st.info('Ningun diagnostico de tipo PRINCIPAL queda fuera del cuadrante con los umbrales actuales.')
        else:
            df_principal_fuera = df_principal_fuera.sort_values('share_total_pct', ascending=False)
            totales_fuera = {
                COL_DIAG: 'TOTAL',
                COL_CAPITULO: '',
                COL_TOTAL_EPIS: int(df_principal_fuera[COL_TOTAL_EPIS].sum()),
                COL_TOTAL_DIAS: int(df_principal_fuera[COL_TOTAL_DIAS].sum()),
                COL_DIAS_GT15: int(df_principal_fuera[COL_DIAS_GT15].sum()),
                'duracion_media': df_principal_fuera['duracion_media'].mean(),
                'share_gt15_pct': df_principal_fuera['share_gt15_pct'].mean(),
                'esperado_pct': df_principal_fuera['esperado_pct'].mean(),
                'lift_pct': df_principal_fuera['lift_pct'].mean(),
                'share_total_pct': df_principal_fuera['share_total_pct'].sum(),
                'S1_hibrida': df_principal_fuera['S1_hibrida'].mean(),
            }
            df_principal_fuera = pd.concat([df_principal_fuera, pd.DataFrame([totales_fuera])], ignore_index=True)
            df_principal_format = df_principal_fuera.rename(columns={
                'duracion_media': 'Tod durac media',
                'share_gt15_pct': '%TotEpis>15dias (%)',
                'esperado_pct': 'E[%>15 | media]',
                'lift_pct': 'Lift %>15',
                'share_total_pct': '%Totsobre total epis (%)',
                'S1_hibrida': 'S1h',
            }).copy()
            df_principal_format['%TotEpis>15dias (%)'] = (df_principal_format['%TotEpis>15dias (%)'] * 100).round(2)
            df_principal_format['E[%>15 | media]'] = (df_principal_format['E[%>15 | media]'] * 100).round(2)
            df_principal_format['Lift %>15'] = (df_principal_format['Lift %>15'] * 100).round(2)
            df_principal_format['%Totsobre total epis (%)'] = df_principal_format['%Totsobre total epis (%)'].round(4)
            df_principal_format['Tod durac media'] = df_principal_format['Tod durac media'].round(1)
            df_principal_format['S1h'] = df_principal_format['S1h'].round(2)
            for col in [COL_TOTAL_EPIS, COL_TOTAL_DIAS, COL_DIAS_GT15]:
                df_principal_format[col] = df_principal_format[col].astype('Int64')
            st.dataframe(df_principal_format, use_container_width=True, hide_index=True)
