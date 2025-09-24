# PROYECTO: ANALISIS Y VISUALIZACION DE DIAGNOSTICOS IBERMUTUA

## OBJETIVO
Herramienta de visualizacion interactiva para analizar el proceso de filtrado de diagnosticos de Incapacidad Temporal (IT) y su impacto economico en IBERMUTUA.

## DATOS
- **Archivo**: Dx_consolidadov3.xlsx
- **Periodo**: 10 anos de historico
- **Volumen**: 2,179 diagnosticos unicos (3.86M episodios, 179.6M dias IT)
- **IMPORTANTE**: La fila 2180 del Excel contiene totales agregados; ignorar en calculos

## CONCEPTOS CLAVE

### Criterios de Exclusion
1. **ExclNum**: Diagnosticos con menos de 500 episodios en 10 anos (1,763 diagnosticos)
2. **ExclDur**: Duracion media inferior a 15 dias y mas del 90% de episodios finalizan antes de 15 dias (37 diagnosticos)
3. **Incluido**: No cumple ningun criterio de exclusion (379 diagnosticos para gestion activa)

### Metricas de Impacto
- Con solo el 17.4% de diagnosticos (379 de 2,179):
  - Se captura el 67.8% de episodios totales
  - Se captura el 90.4% de dias totales de IT
  - Se captura el 93% de dias >15 (coste directo para la mutua)

### Importancia del Dia 16
- Dias 1-3: sin subsidio
- Dias 4-15: pago por la empresa
- Dia 16 en adelante: pago por la mutua (60% base reguladora, 75% desde dia 21)
- Por eso la metrica "Dias IT >15" es critica para el impacto economico

## FORMA DE TRABAJO PREFERIDA

### Metodologia
1. PROPONER ANTES DE HACER: siempre presentar un plan claro antes de implementar
2. SIMPLICIDAD: soluciones directas y funcionales
3. RIGOR CON NUMEROS: verificar siempre que los calculos cuadren con las tablas
4. DESARROLLO ITERATIVO: empezar simple y mejorar progresivamente

### Estilo de Comunicacion
- Ser directo y conciso
- Mostrar calculos y verificaciones
- Alertar de discrepancias inmediatamente
- No hacer suposiciones sobre los datos sin verificar
- Presentar opciones claras antes de implementar

## STACK TECNICO
- Python 3.12+
- Streamlit: aplicacion web interactiva
- Plotly: graficos interactivos (funnel y sankey)
- Pandas: procesamiento y analisis de datos
- Openpyxl: lectura de archivos Excel

## ESTRUCTURA DEL PROYECTO
```
DiagnosticosIBERMUTUA/
|- Dx_consolidadov3.xlsx           # Datos originales (no modificar)
|- visualizacion_filtrado.py       # Aplicacion Streamlit principal
|- requirements.txt                # Dependencias Python
|- agents.md                       # Este documento
```

## FUNCIONALIDADES IMPLEMENTADAS

### Dashboard de Filtrado
- Metricas clave en cards (diagnosticos, episodios, % dias totales y % dias >15 retenidos)
- Barra lateral para ajustar los criterios de exclusion (episodios minimos, duracion media, % episodios cortos)
- Clasificacion dinamica ExclNum, ExclDur, Incluido recalculada en vivo segun los sliders
- Graficos funnel (4 paneles) mostrando evolucion por diagnosticos, episodios, dias totales y dias >15
- Grafico de eficiencia tipo Pareto con los porcentajes retenidos
- Matriz de priorizacion interactiva (duracion vs % episodios) con umbrales ajustables y tabla de diagnosticos prioritarios sin leyenda fija

### Otras visualizaciones
- Diagrama Sankey de distribucion de diagnosticos (version inicial)

## ESTADO ACTUAL (24/09/2025)
- Matriz de priorizacion operativa; se eliminó la leyenda lateral y el diagnostico queda solo en los tooltips.
- Slider de umbral de % episodios con paso 0.01 para permitir ajustes finos.
- Se retiraron la tabla resumen estatica y el recordatorio fijo de criterios; el dashboard cierra con la tabla priorizada dinamica.
- KPIs, funnels, Pareto y controles se recalculan correctamente con los filtros actuales; Sankey mantiene la version inicial.

## SIGUIENTES PASOS PROPUESTOS
1. Validar la experiencia en producción y decidir si hace falta un toggle para mostrar leyendas o agrupar colores.
2. Documentar umbrales recomendados y ejemplos de uso para usuarios finales.
3. Evaluar incorporar pruebas ligeras que verifiquen las metricas clave tras modificaciones de lógica.

## FORMA DE PROCEDER
- Proponer plan breve antes de modificar codigo.
- Confirmar con el usuario cambios estructurales.
- Mantener el script en ASCII para evitar problemas de codificacion.
- Usar el commit de referencia (`git reset --hard 414f678`) si se necesita volver a este punto.
