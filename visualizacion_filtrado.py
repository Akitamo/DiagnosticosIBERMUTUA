import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import math

st.set_page_config(page_title="Diagnosticos IBERMUTUA", layout="wide")

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

COLOR_OPTIONS = [
    {"label": "Percentil % epis >15", "value": "pct_gt15_percentil", "type": "continuous"},
    {"label": "Capitulo", "value": COL_CAPITULO, "type": "categorical"},
    {"label": "Tipo", "value": COL_TIPO, "type": "categorical"},
    {"label": "Sin color", "value": None, "type": "none"},
]


SIZE_OPTIONS = {
    "Numero de episodios": COL_TOTAL_EPIS,
    "Numero de dias": COL_TOTAL_DIAS,
}


def ratio(value: float, total: float) -> float:
    return (value / total * 100.0) if total else 0.0


def share_general_text(value: float, total: float) -> str:
    total_int = int(total) if total else 0
    porcentaje = ratio(value, total)
    return f"{porcentaje:.1f}% del total general ({total_int:,})"


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

st.title("Analisis de Filtrado de Diagnosticos IBERMUTUA")
st.markdown("### Visualizacion del impacto de los criterios de exclusion")

st.sidebar.header("Parametros de filtrado")
max_episodios = int(df_base[COL_TOTAL_EPIS].max())

episodios_min = st.sidebar.slider(
    "Episodios minimos para gestion activa",
    min_value=0,
    max_value=max_episodios,
    value=2000,
    step=50,
)

pct_cortos = st.sidebar.slider(
    "% episodios que terminan antes de 15 dias",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=0.5,
)
pct_cortos_prop = pct_cortos / 100.0

excluir_tipo_ex2000 = st.sidebar.checkbox("Excluir Tipo EXCLUIDO_2000", value=True)

st.sidebar.caption("Aplica filtros directos sobre el conjunto base antes de priorizar.")

df_filtrado = df_base.copy()
if excluir_tipo_ex2000:
    tipo_normalizado = df_filtrado[COL_TIPO].astype(str).str.strip().str.lower()
    df_filtrado = df_filtrado[tipo_normalizado != 'excluido_2000']

df_filtrado = df_filtrado[df_filtrado[COL_TOTAL_EPIS] >= episodios_min]

pct_lt15_series = df_filtrado[COL_PCT_EPI_LT15].astype(float)
if pct_lt15_series.max() > 1.0:
    pct_lt15_series = pct_lt15_series / 100.0
df_filtrado = df_filtrado.assign(pct_lt15_prop=pct_lt15_series)
df_filtrado = df_filtrado[df_filtrado['pct_lt15_prop'] <= pct_cortos_prop]

if df_filtrado.empty:
    st.warning("No hay diagnosticos tras aplicar los filtros seleccionados.")
    st.stop()

total_diag_global = int(df_base.shape[0])
total_epi_global = int(df_base[COL_TOTAL_EPIS].sum())
total_dias_global = int(df_base[COL_TOTAL_DIAS].sum())
total_dias15_global = int(df_base[COL_DIAS_GT15].sum())

total_diag_base = int(df_filtrado.shape[0])
total_epi_base = int(df_filtrado[COL_TOTAL_EPIS].sum())
total_dias_base = int(df_filtrado[COL_TOTAL_DIAS].sum())
total_dias15_base = int(df_filtrado[COL_DIAS_GT15].sum())

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Diagnosticos tras filtros", f"{total_diag_base:,}", delta=share_general_text(total_diag_base, total_diag_global))
with col2:
    st.metric("Episodios tras filtros", f"{total_epi_base:,}", delta=share_general_text(total_epi_base, total_epi_global))
with col3:
    st.metric("Dias IT tras filtros", f"{total_dias_base:,}", delta=share_general_text(total_dias_base, total_dias_global))
with col4:
    st.metric("Dias >15 tras filtros", f"{total_dias15_base:,}", delta=share_general_text(total_dias15_base, total_dias15_global))

st.markdown("---")
st.markdown("### Matriz de priorizacion de diagnosticos")
st.sidebar.markdown("---")
st.sidebar.subheader("Parametros matriz de priorizacion")

pct_gt15_prop = df_filtrado[COL_PCT_EPI_GT15].astype(float)
if pct_gt15_prop.max() > 1.0:
    pct_gt15_prop = pct_gt15_prop / 100.0
pct_gt15_prop = pct_gt15_prop.clip(0.0, 1.0)
pct_gt15_percentil = pct_gt15_prop.rank(pct=True, method='average').fillna(0.0) * 100

share_total_prop = df_filtrado[COL_SHARE_TOTAL_EPIS].astype(float).clip(1e-6, 1 - 1e-6)
share_dias_prop = df_filtrado[COL_TOTAL_DIAS] / total_dias_base if total_dias_base else 0.0
share_dias15_prop = df_filtrado[COL_DIAS_GT15] / total_dias15_base if total_dias15_base else 0.0
if isinstance(share_dias_prop, pd.Series):
    share_dias_prop = share_dias_prop.clip(1e-6, 1 - 1e-6)
if isinstance(share_dias15_prop, pd.Series):
    share_dias15_prop = share_dias15_prop.clip(1e-6, 1 - 1e-6)

esperado_pct = estimar_pct_esperado(df_filtrado[COL_DURACION_MEDIA], pct_gt15_prop)
lift_pct = np.clip(pct_gt15_prop - esperado_pct, 0.0, None)
impacto_delta = np.maximum(df_filtrado[COL_DURACION_MEDIA] - 15.0, 0.0) * df_filtrado[COL_TOTAL_EPIS] * pct_gt15_prop

df_trabajo = df_filtrado.assign(
    duracion_media=df_filtrado[COL_DURACION_MEDIA],
    share_total_prop=share_total_prop,
    share_total_pct=share_total_prop * 100,
    share_gt15_pct=pct_gt15_prop * 100,
    share_total_logit=logit(share_total_prop),
    share_dias_prop=share_dias_prop,
    share_dias_pct=share_dias_prop * 100,
    share_dias_logit=logit(np.clip(share_dias_prop, 1e-6, 1 - 1e-6)),
    share_dias15_prop=share_dias15_prop,
    share_dias15_pct=share_dias15_prop * 100,
    share_dias15_logit=logit(np.clip(share_dias15_prop, 1e-6, 1 - 1e-6)),
    esperado_pct=esperado_pct,
    lift_pct=lift_pct,
    impacto_delta=impacto_delta,
    pct_gt15_percentil=pct_gt15_percentil,
)

size_choices = list(SIZE_OPTIONS.keys())
size_choice_label = st.sidebar.radio("Tamano burbuja por", size_choices, index=0)
size_column = SIZE_OPTIONS[size_choice_label]
size_series = df_trabajo[size_column].astype(float)
size_cap = float(size_series.quantile(0.95)) if not size_series.empty else 0.0
if size_cap <= 0:
    size_cap = float(size_series.max() or 1.0)
df_trabajo['size_metric'] = size_series.clip(lower=0, upper=size_cap)

color_labels = [opt["label"] for opt in COLOR_OPTIONS]
color_choice_label = st.sidebar.selectbox("Color por", color_labels, index=0)
color_option = next(opt for opt in COLOR_OPTIONS if opt["label"] == color_choice_label)
color_field = color_option["value"]
color_mode = color_option["type"]

dur_series = df_trabajo[COL_DURACION_MEDIA]
if dur_series.dropna().empty:
    dur_min_quant = 0.0
    dur_max_quant = 30.0
else:
    dur_min_quant = float(dur_series.quantile(0.05))
    dur_max_quant = float(dur_series.quantile(0.95))
dur_min = math.floor(dur_min_quant)
dur_max = math.ceil(dur_max_quant)
target_dur_default = 30.0
if dur_min > target_dur_default:
    dur_min = target_dur_default
if dur_max < target_dur_default:
    dur_max = target_dur_default
if dur_max <= dur_min:
    dur_max = dur_min + 1
dur_default = min(max(target_dur_default, float(dur_min)), float(dur_max))
duracion_umbral = st.sidebar.slider("Umbral duracion media (dias)", min_value=float(dur_min), max_value=float(dur_max), value=float(dur_default), step=1.0)

y_metric_option = st.sidebar.radio("Metrica eje Y", ("Dias totales", "Dias >15"), index=0)
if y_metric_option == "Dias >15":
    share_prop_target = df_trabajo['share_dias15_prop']
    share_pct_target = df_trabajo['share_dias15_pct']
    share_logit_target = df_trabajo['share_dias15_logit']
    share_label = "% dias >15"
    share_pct_column = 'share_dias15_pct'
else:
    share_prop_target = df_trabajo['share_dias_prop']
    share_pct_target = df_trabajo['share_dias_pct']
    share_logit_target = df_trabajo['share_dias_logit']
    share_label = "% dias"
    share_pct_column = 'share_dias_pct'

share_pct_series = share_pct_target
share_max = float(share_pct_series.quantile(0.95))
if share_max <= 0:
    share_max = float(share_pct_series.max() or 0.1)
share_target_default = 0.22
share_max = max(share_max, share_target_default)
share_default_data = float(share_pct_series.quantile(0.75))
if share_default_data > share_max:
    share_default_data = share_max
share_default = share_target_default if share_target_default <= share_max else share_default_data
share_default = max(share_default, 0.0)
share_step = max(share_max / 200, 0.0005)
share_umbral_pct = st.sidebar.slider(
    f"Umbral {share_label}",
    min_value=0.0,
    max_value=share_max,
    value=float(round(share_default, 4)),
    step=float(round(share_step, 4)),
)

y_axis_title = "% del total de dias >15 (logit)" if y_metric_option == "Dias >15" else "% del total de dias (logit)"

scatter_kwargs = dict(
    data_frame=df_trabajo,
    x=COL_DURACION_MEDIA,
    y=share_logit_target,
    size='size_metric',
    hover_data={
        COL_DIAG: True,
        COL_CAPITULO: True,
        COL_SHARE_TOTAL_EPIS: ':.4%',
        COL_PCT_EPI_GT15: ':.2%',
        'share_dias_pct': ':.4%',
        'share_dias15_pct': ':.4%',
        COL_DURACION_MEDIA: ':.1f',
        COL_TOTAL_EPIS: ':,',
        COL_TOTAL_DIAS: ':,',
        COL_DIAS_GT15: ':,',
        'share_total_pct': ':.4%',
        'esperado_pct': ':.2%',
        'lift_pct': ':.2%',
        'pct_gt15_percentil': ':.1f',
        'impacto_delta': ':,.0f',
    },
    size_max=60,
)
if color_field is not None:
    scatter_kwargs['color'] = df_trabajo[color_field]
    if color_mode == 'continuous':
        scatter_kwargs['color_continuous_scale'] = 'YlOrRd'
        scatter_kwargs['range_color'] = (0, 100)

fig_prior = px.scatter(**scatter_kwargs)
if color_field is not None and color_mode == 'continuous':
    fig_prior.update_coloraxes(colorbar_title='% epis >15 (percentil)')
fig_prior.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color='#333')))
fig_prior.update_layout(
    title="Priorizar diagnosticos incluidos",
    xaxis_title="Duracion media del episodio (dias)",
    yaxis_title=y_axis_title,
    template="plotly_white",
    margin=dict(l=40, r=20, t=60, b=60),
    showlegend=False,
)

min_prop = float(share_prop_target.min()) if not share_prop_target.empty else 1e-6
max_prop = float(share_prop_target.max()) if not share_prop_target.empty else 1 - 1e-6
candidate_perc = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
tick_props = candidate_perc[(candidate_perc / 100 >= min_prop) & (candidate_perc / 100 <= max_prop)] / 100
if tick_props.size == 0:
    tick_props = np.array([min_prop, max_prop])
tick_vals = logit(tick_props)
tick_text = [f"{prop * 100:.4f}%" if prop < 0.01 else f"{prop * 100:.2f}%" for prop in tick_props]
fig_prior.update_yaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_text)

fig_prior.add_vline(x=duracion_umbral, line_dash='dash', line_color='gray', annotation_text='Umbral duracion', annotation_position='top left')
share_umbral_prop = share_umbral_pct / 100 if share_umbral_pct else 0.0
share_umbral_logit = logit(share_umbral_prop) if share_umbral_prop else logit(1e-6)
fig_prior.add_hline(y=share_umbral_logit, line_dash='dash', line_color='gray', annotation_text=f'Umbral {share_label} ({share_umbral_pct:.4f}%)', annotation_position='bottom right')
st.plotly_chart(fig_prior, use_container_width=True)

df_cuadrante = df_trabajo[(df_trabajo[COL_DURACION_MEDIA] >= duracion_umbral) & (df_trabajo[share_pct_column] >= share_umbral_pct) & (df_trabajo[COL_TOTAL_EPIS] >= episodios_min)].copy()

st.markdown('#### Diagnosticos priorizados (cuadrante)')
if df_cuadrante.empty:
    st.info(f'Ningun diagnostico supera simultaneamente los umbrales de duracion y {share_label}.')
else:
    diag_sel = int(df_cuadrante.shape[0])
    epis_sel = int(df_cuadrante[COL_TOTAL_EPIS].sum())
    dias_sel = int(df_cuadrante[COL_TOTAL_DIAS].sum())
    dias15_sel = int(df_cuadrante[COL_DIAS_GT15].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Diagnosticos priorizados", f"{diag_sel:,}", delta=share_general_text(diag_sel, total_diag_global))
    with c2:
        st.metric("Episodios priorizados", f"{epis_sel:,}", delta=share_general_text(epis_sel, total_epi_global))
    with c3:
        st.metric("Dias IT priorizados", f"{dias_sel:,}", delta=share_general_text(dias_sel, total_dias_global))
    with c4:
        st.metric("Dias >15 priorizados", f"{dias15_sel:,}", delta=share_general_text(dias15_sel, total_dias15_global))

    df_tabla = df_cuadrante[[
        COL_DIAG,
        COL_CAPITULO,
        COL_TIPO,
        COL_TOTAL_EPIS,
        COL_TOTAL_DIAS,
        COL_DIAS_GT15,
        COL_DURACION_MEDIA,
        COL_PCT_EPI_GT15,
        COL_SHARE_TOTAL_EPIS,
    ]].copy()
    df_tabla = df_tabla.sort_values(COL_TOTAL_EPIS, ascending=False)
    df_tabla_format = df_tabla.rename(columns={
        COL_DURACION_MEDIA: 'Tod durac media',
        COL_PCT_EPI_GT15: '%TotEpis>15dias (%)',
        COL_SHARE_TOTAL_EPIS: '%Totsobre total epis (%)',
    }).copy()
    df_tabla_format['%TotEpis>15dias (%)'] = df_tabla_format['%TotEpis>15dias (%)'].round(2)
    df_tabla_format['%Totsobre total epis (%)'] = df_tabla_format['%Totsobre total epis (%)'].round(4)
    df_tabla_format['Tod durac media'] = df_tabla_format['Tod durac media'].round(1)
    for col in [COL_TOTAL_EPIS, COL_TOTAL_DIAS, COL_DIAS_GT15]:
        df_tabla_format[col] = df_tabla_format[col].astype('Int64')
    st.dataframe(df_tabla_format, use_container_width=True, hide_index=True)

    df_capitulos = df_cuadrante.groupby(COL_CAPITULO).agg(
        diagnosticos=(COL_DIAG, 'nunique'),
        episodios=(COL_TOTAL_EPIS, 'sum'),
        dias=(COL_TOTAL_DIAS, 'sum'),
    ).reset_index()
    for col in ['diagnosticos', 'episodios', 'dias']:
        df_capitulos[col] = df_capitulos[col].astype(int)

    if not df_capitulos.empty:
        st.markdown('##### Distribucion por capitulo (diagnosticos priorizados)')
        cap_cols = st.columns(3)
        metric_map = [
            ("Diagnosticos", 'diagnosticos'),
            ("Episodios", 'episodios'),
            ("Dias IT", 'dias'),
        ]
        for fig_col, (metric_label, metric_field) in zip(cap_cols, metric_map):
            with fig_col:
                df_plot = df_capitulos.sort_values(metric_field, ascending=False)
                fig = px.bar(
                    df_plot,
                    x=metric_field,
                    y=COL_CAPITULO,
                    orientation='h',
                    title=metric_label,
                    text=[f"{int(v):,}" for v in df_plot[metric_field]],
                )
                fig.update_layout(
                    margin=dict(l=0, r=10, t=40, b=0),
                    height=320,
                    xaxis=dict(tickformat=','),
                    yaxis=dict(categoryorder='array', categoryarray=df_plot[COL_CAPITULO].tolist()),
                )
                fig.update_traces(
                    marker_color='#1f77b4',
                    textposition='outside',
                    hovertemplate='%{y}<br>'+metric_label+': %{x:,}'
                )
                st.plotly_chart(fig, use_container_width=True)

tipo_norm = df_trabajo[COL_TIPO].astype(str).str.strip().str.lower()
restantes_mask = (tipo_norm == 'principal') & (~df_trabajo.index.isin(df_cuadrante.index))
df_principal_fuera = df_trabajo[restantes_mask][[
    COL_DIAG,
    COL_CAPITULO,
    COL_TIPO,
    COL_TOTAL_EPIS,
    COL_TOTAL_DIAS,
    COL_DIAS_GT15,
    COL_DURACION_MEDIA,
    COL_PCT_EPI_GT15,
    COL_SHARE_TOTAL_EPIS,
]].copy()

with st.expander('Diagnosticos PRINCIPAL fuera del cuadrante', expanded=False):
    if df_principal_fuera.empty:
        st.info('Ningun diagnostico de tipo PRINCIPAL queda fuera del cuadrante con los umbrales actuales.')
    else:
        df_principal_fuera = df_principal_fuera.sort_values(COL_TOTAL_EPIS, ascending=False)
        df_principal_format = df_principal_fuera.rename(columns={
            COL_DURACION_MEDIA: 'Tod durac media',
            COL_PCT_EPI_GT15: '%TotEpis>15dias (%)',
            COL_SHARE_TOTAL_EPIS: '%Totsobre total epis (%)',
        }).copy()
        df_principal_format['%TotEpis>15dias (%)'] = df_principal_format['%TotEpis>15dias (%)'].round(2)
        df_principal_format['%Totsobre total epis (%)'] = df_principal_format['%Totsobre total epis (%)'].round(4)
        df_principal_format['Tod durac media'] = df_principal_format['Tod durac media'].round(1)
        for col in [COL_TOTAL_EPIS, COL_TOTAL_DIAS, COL_DIAS_GT15]:
            df_principal_format[col] = df_principal_format[col].astype('Int64')
        st.dataframe(df_principal_format, use_container_width=True, hide_index=True)
