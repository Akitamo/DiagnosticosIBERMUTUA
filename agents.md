# PROYECTO: ANALISIS Y VISUALIZACION DE DIAGNOSTICOS IBERMUTUA

## OBJETIVO
Herramienta de visualizacion interactiva para analizar el proceso de filtrado de diagnosticos de Incapacidad Temporal (IT) y su impacto economico en IBERMUTUA.

## DATOS
- Archivo: Dx_consolidadov3.xlsx (no modificar)
- Periodo: 10 anos de historico
- Volumen: 2,179 diagnosticos unicos (3.86M episodios, 179.6M dias IT)
- Nota: la fila 2180 del Excel contiene totales agregados; excluir de los calculos

## CRITERIOS DE EXCLUSION
1. ExclNum: diagnosticos con menos de 500 episodios en 10 anos
2. ExclDur: duracion media <15 dias y >90% de episodios finalizan antes de 15 dias
3. Incluido: no cumple criterios anteriores (379 diagnosticos para gestion activa)

## METRICAS RELEVANTES
- 17.4% de diagnosticos gestionados concentran 67.8% de episodios, 90.4% de dias IT y 93% de dias >15
- Dia 16 marca el inicio del coste directo para la mutua (60% base reguladora, 75% desde el dia 21)

## STACK TECNICO
Python 3.12+, Streamlit, Plotly, Pandas, Openpyxl.

## FUNCIONALIDAD PRINCIPAL
- Dashboard con KPIs globales y tras filtros, funnel de diagnosticos/episodios/dias/dias>15 y grafico Pareto.
- Matriz de priorizacion (bubble chart logit) con coloracion por percentil de % episodios >15 (gradiente YlOrRd) y selector de tamano por episodios/dias, tabla de cuadrante, graficas por capitulo y listado de tipo PRINCIPAL fuera del umbral.
- Sankey inicial para distribucion de diagnosticos.

## PARAMETROS INICIALES (25/09/2025)
- Episodios minimos: 2000
- % episodios <15 dias: 90
- Checkbox "Excluir Tipo EXCLUIDO_2000": activo
- Color por: Percentil % episodios >15 (gradiente YlOrRd)
- Tamano burbuja: Numero de episodios
- Metrica eje Y: Dias totales
- Umbral duracion media: 30 dias (slider con paso entero)
- Umbral % dias: 0.22%

## ESTADO ACTUAL
- Sliders recalculan las categorias ExclNum y ExclDur en vivo; la matriz y KPIs se sincronizan con los filtros.
- Escala logit, winsor p95 para el tamano de burbuja (con selector episodios/dias) y gradiente continuo por percentil %>15; leyenda de categorias sigue oculta y la tabla de PRINCIPAL queda en expander colapsado.
- Graficas horizontales por capitulo muestran diagnosticos, episodios y dias retenidos tras los filtros.
- Lado derecho muestra KPIs de cuadrante y tabla priorizada con formatos porcentuales y totales.

## PROXIMOS PASOS SUGERIDOS
1. Documentar buenas practicas/umbrales para usuarios finales.
2. Evaluar pruebas automaticas ligeras para validar metricas clave tras cambios.
3. Revisar si se necesita un toggle para mostrar leyendas o agrupar colores en la matriz.

## FORMA DE TRABAJO
- Proponer plan breve antes de implementar.
- Verificar numerica y claramente cualquier cambio.
- Mantener el codigo en ASCII.
- Commit de referencia para rollback: `git reset --hard 414f678`.
