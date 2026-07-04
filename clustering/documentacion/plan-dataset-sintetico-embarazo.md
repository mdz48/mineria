# Plan: Dataset sintético de pacientes embarazadas para clustering

## 1. Propósito y alcance

Este documento define el plan para generar un **dataset sintético** orientado a **probar algoritmos de clustering** en un dominio clínico realista: **pacientes embarazadas en control prenatal**.

El dataset debe cumplir simultáneamente:

- Tener **estructura de grupos** (clusters) reconocibles en el espacio de características.
- Incluir **contaminación por atípicos** bajo mecanismos aleatorios y plausibles.
- Incluir **valores nulos** generados bajo los tres mecanismos clásicos: **MCAR**, **MAR** y **MNAR**.
- Respetar **restricciones físicas y clínicas** coherentes con embarazo humano.
- Entregar **metadatos de verdad de referencia** (ground truth) para evaluar clustering, detección de outliers e imputación.

**No es objetivo** de este plan simular un EMR completo ni reemplazar datos clínicos reales. Es un banco de prueba controlado para minería de datos.

**Restricción de alcance (variables excluidas):** el dataset **no incluirá** variables de **laboratorio** (glucosa, hemoglobina, hematocrito, creatinina, albumina, proteinuria, etc.) ni variables **fetales** (altura uterina, peso fetal estimado, etc.). El clustering se apoyará en datos demográficos, obstétricos, antropométricos, signos vitales y variables categóricas de contexto.

---

## 2. Principios de diseño


| Principio                    | Descripción                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------- |
| Clusters interpretables      | Cada grupo debe corresponder a un perfil clínico reconocible (no solo blobs geométricos).           |
| Correlaciones realistas      | Variables relacionadas clínicamente deben co-moverse (IMC ↔ ganancia de peso, edad ↔ riesgo, etc.). |
| Restricciones duras          | Rangos, relaciones entre variables e inconsistencias imposibles se validan en generación.           |
| Contaminación parametrizable | Proporción de outliers y nulos configurable por mecanismo.                                          |
| Trazabilidad                 | Cada fila lleva flags de cluster verdadero, tipo de outlier y mecanismo de missingness.             |
| Reproducibilidad             | Semilla aleatoria fija y registro de parámetros en archivo de configuración.                        |


---

## 3. Unidad de observación y tamaño

- **Unidad:** una paciente embarazada en un **punto de control prenatal** (visita).
- **Granularidad temporal:** una fila = una visita. Opcionalmente se puede generar solo **una visita por paciente** (cross-sectional) o **serie longitudinal** (múltiples visitas por paciente). Para la primera versión se recomienda **cross-sectional** con N pacientes.
- **Tamaño sugerido inicial:** N = 2 000 pacientes.
- **Distribución de clusters:** desigual y realista (no 5 grupos de exactamente 400). Ejemplo orientativo:
  - Bajo riesgo / evolución normal: ~45 %
  - Riesgo metabólico (obesidad / comorbilidad metabólica): ~18 %
  - Riesgo hipertensivo: ~12 %
  - Bajo peso nutricional / ganancia insuficiente: ~15 %
  - Embarazo múltiple o alto riesgo obstétrico: ~7 %
  - Residual / mezcla: ~3 %

Los porcentajes son configurables; lo importante es que **no sean uniformes**.

---

## 4. Perfiles de cluster (verdad de referencia)

Cada paciente se asigna primero a un **cluster latente** k. Las variables se generan condicionadas a k.

### 4.1 Cluster C0 — Embarazo bajo riesgo

- Edad 22–35 años, IMC pre-gestacional 20–26 kg/m².
- Ganancia de peso acorde a IOM según trimestre.
- PA sistólica 100–120, diastólica 60–75 mmHg.
- Frecuencia cardíaca en rango normal de embarazo (70–90 lpm).
- Sin comorbilidades crónicas.

### 4.2 Cluster C1 — Riesgo metabólico

- IMC pre-gestacional ≥ 30 o IMC 27–29.9 con `diabetes_previa` = 1.
- Ganancia de peso en límite superior o excesiva según IMC.
- Mayor probabilidad de tabaquismo previo (variable categórica).
- PA en rango alto-normal o levemente elevada.

### 4.3 Cluster C2 — Riesgo hipertensivo / preeclampsia leve

- PA sistólica 130–160, diastólica 85–105 mmHg.
- `hipertension_cronica` = 1 con mayor frecuencia, o PA elevada de novo.
- Edad > 35 o primer embarazo como factores de riesgo frecuentes.
- Posible restricción de ganancia de peso.

### 4.4 Cluster C3 — Bajo peso nutricional / ganancia insuficiente

- IMC pre-gestacional < 18.5.
- Ganancia de peso por debajo de lo esperado según semanas e IMC.
- Menor uso de suplementos (`suplemento_hierro`, `suplemento_acido_folico`).
- Nivel socioeconómico bajo con mayor frecuencia (MAR/MNAR útil aquí).

### 4.5 Cluster C4 — Alto riesgo obstétrico

- Embarazo múltiple, edad materna avanzada (≥ 38), comorbilidades.
- Semanas de gestación con distribución distinta (partos pretérmino posibles).
- Mayor paridad, múltiples comorbilidades simultáneas (diabetes + hipertensión).

### 4.6 Cluster C5 — Residual / transición

- Perfiles en frontera entre dos clusters; útil para evaluar robustez del clustering.
- Mezcla de 2 centroides con ruido alto.

---

## 5. Esquema de variables

### 5.1 Identificadores y metadatos (no usar en clustering)


| Variable            | Tipo                        | Descripción                                                                  |
| ------------------- | --------------------------- | ---------------------------------------------------------------------------- |
| `paciente_id`       | entero                      | Identificador único                                                          |
| `cluster_verdadero` | entero 0–5                  | Etiqueta latente usada en generación                                         |
| `es_outlier`        | binario                     | 1 si la fila fue contaminada como atípico                                    |
| `tipo_outlier`      | categórico                  | `medicion`, `registro`, `combinacion_imposible`, `patologia_rara`, `ninguno` |
| `mecanismo_nulo`    | categórico por celda o JSON | Registro de MCAR / MAR / MNAR por variable                                   |


### 5.2 Variables demográficas y obstétricas


| Variable                | Tipo              | Rango / dominio         | Restricciones                                                                 |
| ----------------------- | ----------------- | ----------------------- | ----------------------------------------------------------------------------- |
| `edad_anios`            | continua          | 15–49                   | Embarazo < 15 o > 49: marcar como outlier de registro o excluir en validación |
| `imc_pregestacional`    | continua          | 14–55 kg/m²             | Relacionado con peso y talla si se modelan por separado                       |
| `semanas_gestacion`     | continua o entera | 6–42                    | Coherente con trimestre                                                       |
| `trimestre`             | ordinal 1–3       | derivado                | Debe coincidir con semanas                                                    |
| `num_embarazos_previos` | entera            | 0–10                    | Paridad                                                                       |
| `num_partos_previos`    | entera            | 0–num_embarazos_previos | No puede exceder gestaciones previas                                          |
| `embarazo_multiple`     | binario           | 0/1                     | En C4 con mayor probabilidad                                                  |
| `tabaquismo_activo`     | binario           | 0/1                     | Prevalencia baja en embarazo (~3–8 %)                                         |
| `diabetes_previa`       | binario           | 0/1                     | Más frecuente en C1; correlacionada con IMC alto                              |
| `hipertension_cronica`  | binario           | 0/1                     | Condiciona PA basal; más frecuente en C2                                      |


### 5.3 Antropometría y signos vitales


| Variable              | Tipo     | Rango típico | Restricciones                                |
| --------------------- | -------- | ------------ | -------------------------------------------- |
| `peso_kg`             | continua | 40–150       | peso ≈ IMC × (talla_m)² si talla existe      |
| `talla_cm`            | continua | 140–185      | Estable por paciente                         |
| `ganancia_peso_kg`    | continua | −2 a 25      | Depende de semanas e IMC inicial             |
| `presion_sistolica`   | continua | 80–180 mmHg  | sistólica > diastólica + 20 mínimo           |
| `presion_diastolica`  | continua | 45–120 mmHg  | Ver arriba                                   |
| `frecuencia_cardiaca` | continua | 55–110 lpm   | Mayor en embarazo (~+10–20 vs no embarazada) |


### 5.4 Variables categóricas adicionales (opcionales para enriquecer clusters)


| Variable                  | Tipo       | Valores                        |
| ------------------------- | ---------- | ------------------------------ |
| `nivel_educacion`         | ordinal    | primaria, secundaria, superior |
| `area_residencia`         | categórica | urbana, rural                  |
| `suplemento_acido_folico` | binario    | sí / no                        |
| `suplemento_hierro`       | binario    | sí / no                        |


---

## 6. Restricciones físicas y clínicas (reglas duras)

Estas reglas se aplican **después** de generar valores por cluster y **antes** de exportar. Si se violan, se corrige o se marca como outlier intencional.

1. **Presión arterial:** sistólica > diastólica; diferencia mínima 15 mmHg; sistólica ≤ 220, diastólica ≤ 130 (fuera de eso → outlier extremo).
2. **Paridad:** `num_partos_previos` ≤ `num_embarazos_previos`.
3. **Trimestre:** derivado de semanas (T1: 1–13, T2: 14–27, T3: 28–42).
4. **Ganancia de peso:** en T1 casi nula o leve; acumulativa con semanas; límites según IOM por categoría de IMC.
5. **Peso–IMC–talla:** coherencia `peso_kg ≈ imc_pregestacional × (talla_m)²` con tolerancia por ganancia gestacional acumulada.
6. **Diabetes_previa–IMC:** si `diabetes_previa` = 1, IMC pre-gestacional tendencialmente ≥ 27 (salvo outliers marcados).
7. **Hipertension_cronica–PA:** si `hipertension_cronica` = 1, PA basal desplazada hacia arriba.
8. **Edad–riesgo:** probabilidades de comorbilidades aumentan con edad > 35.
9. **IMC extremo:** IMC < 14 o > 50 generado solo como outlier de `patologia_rara` o error de registro.
10. **Embarazo múltiple–semanas:** en C4, mayor probabilidad de parto pretérmino (semanas < 37).

---

## 7. Pipeline de generación (orden de pasos)

```
[Paso 0] Cargar configuración (N, semilla, proporciones, tasas de contaminación)
    ↓
[Paso 1] Asignar cluster_verdadero según distribución desigual
    ↓
[Paso 2] Generar variables independientes condicionadas al cluster (edad, IMC, semanas...)
    ↓
[Paso 3] Generar variables dependientes (PA, peso, ganancia) usando correlaciones intra-cluster
    ↓
[Paso 4] Aplicar restricciones duras y coherencia cruzada
    ↓
[Paso 5] Inyectar outliers (Sección 8)
    ↓
[Paso 6] Inyectar valores nulos MCAR / MAR / MNAR (Sección 9)
    ↓
[Paso 7] Validación automática + reporte de calidad
    ↓
[Paso 8] Exportar: datos.csv, metadatos.csv, config.json, reporte.md
```

---

## 8. Contaminación por atípicos (outliers)

### 8.1 Objetivo

Simular **ruido realista** que dificulte el clustering sin destruir la estructura global. Proporción sugerida: **3–8 %** de filas con al menos un atributo atípico; **1–2 %** de filas completamente aberrantes.

### 8.2 Mecanismos aleatorios


| Mecanismo               | Descripción                                                                             | Ejemplo en dominio                                                             |
| ----------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `medicion`              | Error de instrumento; una variable numérica salta ± k desviaciones estándar del cluster | PA sistólica 200 mmHg en paciente C0 por manguito mal colocado                 |
| `registro`              | Error humano de digitación                                                              | Edad 340, peso 340 kg, decimal mal puesto                                      |
| `combinacion_imposible` | Valores individualmente posibles pero **juntos incoherentes**                           | IMC 35 + ganancia −5 kg en semana 38 + peso 45 kg                              |
| `patologia_rara`        | Caso clínico extremo pero real                                                          | PA muy alta + embarazo múltiple + edad 17 años (perfil de alto riesgo extremo) |
| `posicion`              | Punto en frontera lejos del centroide pero no imposible                                 | Perfil C1 con IMC límite y PA normal                                           |


### 8.3 Reglas de inyección

1. Elegir fila al azar **estratificado por cluster** (no solo del cluster mayor).
2. Elegir mecanismo según tabla de probabilidades (ej.: 40 % medicion, 25 % registro, 20 % combinacion, 10 % patologia_rara, 5 % posicion).
3. Para `medicion`: perturbar 1–3 variables numéricas con multiplicador o suma aleatoria.
4. Para `registro`: aplicar transformaciones tipo ×10, permutar dígitos, signo incorrecto.
5. Marcar `es_outlier=1` y `tipo_outlier`; **no cambiar** `cluster_verdadero` (permite evaluar robustez).

### 8.4 Parámetros configurables

- `tasa_outlier_fila`: proporción de filas afectadas.
- `tasa_outlier_extremo`: subconjunto con múltiples variables corruptas.
- `intensidad_perturbacion`: en unidades de desviación estándar del cluster.

---

## 9. Contaminación por valores nulos

### 9.1 Tasas globales sugeridas

- **MCAR:** 2–5 % de celdas en variables seleccionadas.
- **MAR:** 3–7 % adicional, concentrado en variables específicas.
- **MNAR:** 1–3 % adicional, sesgado hacia valores extremos.

Total efectivo de missingness: ~6–12 % según solapamiento. Documentar tasas por variable en el reporte.

### 9.2 MCAR — Missing Completely At Random

**Definición operativa:** la probabilidad de que falte un valor **no depende** de otros datos observados ni del valor faltante.

**Implementación:**

- Para cada celda elegible, con probabilidad p_mcar fija → `NaN`.
- Variables candidatas: signos vitales (`presion_sistolica`, `presion_diastolica`, `frecuencia_cardiaca`), antropometría (`peso_kg`, `talla_cm`, `ganancia_peso_kg`), variables demográficas/obstétricas.
- **No aplicar MCAR** a identificadores ni a `cluster_verdadero`.

**Ejemplo clínico plausible:** formulario incompleto aleatorio, dato no transcrito en una visita.

### 9.3 MAR — Missing At Random

**Definición operativa:** el missingness **depende de variables observadas**, no del valor faltante en sí.

**Patrones propuestos para embarazo:**


| Variable con nulo     | Condición observada que aumenta P(missing)                                   |
| --------------------- | ---------------------------------------------------------------------------- |
| `presion_diastolica`  | `presion_sistolica` normal (control rápido, solo se anota sistólica)         |
| `ganancia_peso_kg`    | trimestre 1 (aún no se evalúa ganancia acumulada)                            |
| `num_partos_previos`  | `num_embarazos_previos` = 0 (primigesta, dato no relevado en urgencias)      |
| `suplemento_hierro`   | cluster C0 y trimestre 1 (perfil bajo riesgo, menos registro administrativo) |
| `frecuencia_cardiaca` | semanas_gestacion < 12 (primera visita incompleta)                           |


**Implementación:** P(missing | X_obs) = sigmoid(a + b × indicador_condición).

### 9.4 MNAR — Missing Not At Random

**Definición operativa:** la probabilidad de missing **depende del valor no observado** (o de un proxy no medido).

**Patrones propuestos:**


| Variable                  | Mecanismo MNAR                                | Interpretación clínica                                   |
| ------------------------- | --------------------------------------------- | -------------------------------------------------------- |
| `tabaquismo_activo`       | Mayor P(missing) si valor verdadero = 1       | Negación / vergüenza social                              |
| `peso_kg`                 | Mayor P(missing) si IMC > 35                  | Evitar pesaje por incomodidad                            |
| `presion_sistolica`       | Mayor P(missing) si PA verdadera > 160        | Omisión en contexto de estrés o negación del diagnóstico |
| `suplemento_acido_folico` | Mayor P(missing) si no lo consume (valor = 0) | No reportar por percepción de cumplimiento deficiente    |


**Implementación:** generar valor completo primero; luego aplicar P(missing | valor) monótona creciente respecto al extremo sensibles.

**Importante:** guardar copia **sin nulos** (`datos_completos.csv`) solo para evaluación interna; no entregar al pipeline de clustering en producción de prueba.

### 9.5 Registro de missingness

Matriz auxiliar `missingness_log.csv`:

- `paciente_id`, `variable`, `mecanismo` ∈ {MCAR, MAR, MNAR, ninguno}
- Permite evaluar imputación y sesgo introducido.

---

## 10. Correlaciones y dependencias a modelar

Para que el clustering tenga señal, las variables no deben ser independientes. Generación sugerida:

1. **Núcleo latente por cluster:** vector μ_k de dimensiones d (edad, IMC, PA, peso, ganancia, FC, ...).
2. **Covarianza Σ_k** por cluster (matrices distintas: C2 con alta varianza en PA, C3 en IMC/ganancia).
3. Muestrear **multivariante normal** truncada a rangos físicos.
4. Variables categóricas: **regresión logística** con logits dependientes del cluster y de variables continuas generadas.
5. **Ganancia de peso:** regresión sobre semanas + IMC inicial + ruido + efecto cluster.

Variables con correlación objetivo aproximada (post-generación):


| Par                            | Correlación esperada |
| ------------------------------ | -------------------- |
| IMC ↔ ganancia_peso (ajustada) | 0.4 – 0.6            |
| semanas ↔ ganancia_peso        | 0.5 – 0.75           |
| peso_kg ↔ IMC                  | 0.85 – 0.95          |
| PA sistólica ↔ diastólica      | 0.65 – 0.80          |
| edad ↔ diabetes_previa         | débil positiva       |
| edad ↔ hipertension_cronica    | débil positiva       |


---

## 11. Archivos de salida


| Archivo                        | Contenido                                                |
| ------------------------------ | -------------------------------------------------------- |
| `datos_embarazo_sintetico.csv` | Dataset con nulos y outliers (entrada al clustering)     |
| `datos_embarazo_completo.csv`  | Sin nulos ni outliers (solo evaluación)                  |
| `metadatos_ground_truth.csv`   | cluster_verdadero, flags outlier, missingness            |
| `config_generacion.json`       | Todos los parámetros y semilla                           |
| `reporte_calidad.md`           | Estadísticas, tasas de missing, distribución por cluster |


---

## 12. Validación del dataset generado

### 12.1 Checks automáticos

- [ ] Ninguna restricción dura violada (salvo outliers marcados).
- [ ] Proporción de clusters dentro de ±2 % de lo configurado.
- [ ] Tasas MCAR/MAR/MNAR dentro de tolerancia.
- [ ] Correlaciones dentro de bandas esperadas.
- [ ] Distribuciones marginales visualmente plausibles (histogramas).

### 12.2 Checks para clustering

- Silhouette con `cluster_verdadero` > baseline aleatorio.
- ARI/NMI entre clustering (k=6) y `cluster_verdadero` > 0.5 en datos sin outliers (objetivo orientativo).
- Degradación controlada al añadir outliers y nulos.

### 12.3 Checks clínicos (revisión manual ligera)

- Revisar 20 filas aleatorias por cluster.
- Confirmar que perfiles C2 tienen PA alta coherente, C3 IMC bajo y ganancia insuficiente, etc.

---

## 13. Configuración propuesta (YAML / JSON)

```json
{
  "semilla": 42,
  "n_pacientes": 2000,
  "proporciones_cluster": [0.45, 0.18, 0.12, 0.15, 0.07, 0.03],
  "outliers": {
    "tasa_fila": 0.05,
    "tasa_extremo": 0.015,
    "prob_mecanismo": {
      "medicion": 0.40,
      "registro": 0.25,
      "combinacion_imposible": 0.20,
      "patologia_rara": 0.10,
      "posicion": 0.05
    }
  },
  "missingness": {
    "mcar_prob_celda": 0.03,
    "mar_reglas": ["diastolica_si_sistolica_baja", "ganancia_si_trimestre_1"],
    "mnar_variables": ["tabaquismo_activo", "peso_kg", "presion_sistolica"]
  }
}
```

---

## 14. Implementación técnica sugerida

- **Lenguaje:** Python 3.10+
- **Librerías:** `numpy`, `pandas`, `scipy.stats`, `pyyaml` o `json`, opcional `faker` para IDs.
- **Estructura de código** (futura, fuera de este documento):

```
clustering/
  documentacion/
    plan-dataset-sintetico-embarazo.md   ← este archivo
  src/
    config.py
    clusters.py          # perfiles μ_k, Σ_k
    generador_base.py    # pasos 1–4
    outliers.py          # paso 5
    missingness.py       # paso 6
    validacion.py        # paso 7
    main_generar.py
  data/
    (salidas CSV)
```

---

## 15. Riesgos y limitaciones

- Los perfiles son **simplificaciones**; embarazo real tiene más heterogeneidad.
- MNAR mal calibrado puede hacer imposible la imputación insesgada; es deseable para pruebas pero debe documentarse.
- Outliers de `combinacion_imposible` pueden ser detectados por reglas simples; útil para probar pipelines híbridos (reglas + ML).
- No incluye variables de laboratorio ni fetales (decisión de alcance).
- No incluye variables temporales longitudinales en v1; extensión futura posible.

---

## 16. Próximos pasos

1. Revisar y ajustar proporciones de clusters y lista de variables con el equipo.
2. Fijar tasas finales de outliers y missingness según experimentos de clustering previstos.
3. Implementar generador según pipeline (Sección 7).
4. Ejecutar validación (Sección 12) y iterar parámetros.
5. Publicar dataset + ground truth para experimentos de clustering, imputación y detección de anomalías.

---

## Apéndice A — Ecuaciones (referencia rápida, sin LaTeX)

> Este apéndice está separado del cuerpo del plan para lectura opcional posterior.

### A.1 Generación multivariante por cluster

```
x_i | cluster=k  ~  TruncNormal( mu_k, Sigma_k, limites_inferiores, limites_superiores )
```

Donde:

- `x_i` = vector de variables continuas del paciente i
- `mu_k` = vector medio del cluster k
- `Sigma_k` = matriz de covarianza del cluster k
- `TruncNormal` = normal multivariante truncada a rangos clínicos

### A.2 Presión arterial coherente

```
presion_sistolica = presion_diastolica + delta
delta ~ Uniform(20, 60)   (en mmHg)
```

### A.3 Ganancia de peso esperada (simplificada IOM)

```
ganancia_esperada = intercepto_imc_categoria + coef_semanas × semanas_gestacion + ruido
```

Categorías IOM de IMC:

- bajo peso: IMC < 18.5
- normal: 18.5–24.9
- sobrepeso: 25–29.9
- obesidad: ≥ 30

### A.4 MCAR

```
P( X_ij = missing ) = p_mcar    para toda celda (i,j) elegible
```

Independiente de i, j (salvo exclusiones de columnas).

### A.5 MAR

```
P( X_ij = missing | X_obs ) = sigmoid( alpha_j + sum_l( beta_jl × f_l(X_obs) ) )
```

Donde `f_l` son funciones de variables observadas (indicadores, semanas, PA, etc.).

### A.6 MNAR

```
P( X_ij = missing | X_ij = x ) = sigmoid( gamma_j0 + gamma_j1 × g(x) )
```

Donde `g(x)` crece hacia valores extremos (ej.: g(x) = PA si presión alta, g(x) = IMC si peso).

### A.7 Perturbación de outlier tipo medición

```
x_perturbado = x + epsilon
epsilon ~ Normal( 0, lambda × std_cluster )
```

`lambda` típico entre 3 y 8.

### A.8 Coherencia peso–IMC–talla

```
peso_esperado = imc_pregestacional × (talla_m)² + delta_gestacional
delta_gestacional ~ f(ganancia_peso_kg, semanas_gestacion)
```

Tolerancia configurable para outliers de registro.

### A.9 Métricas de evaluación de clustering

```
ARI = Adjusted Rand Index( labels_predichos, cluster_verdadero )
NMI = Normalized Mutual Information( labels_predichos, cluster_verdadero )
Silhouette = mean( s(i) )   con s(i) basado en distancias intra/inter cluster
```

---

*Documento versión 1.1 — dominio: pacientes embarazadas — sin variables de laboratorio ni fetales — objetivo: clustering con outliers y missingness MCAR/MAR/MNAR.*