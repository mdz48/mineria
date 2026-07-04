# Plan v2: Dataset sintético alineado con BD prenatal (clustering)

**Versión:** 2.0  
**Reemplaza operativamente:** `plan-dataset-sintetico-embarazo.md` (v1 queda como referencia histórica)  
**Documentos relacionados:** `bd_salud_pretanal.md`, `documentacion-preeclampsia.md`, `analisis-convergencia-documentacion-clinica.md`

---

## 1. Objetivo redefinido

Generar un **dataset sintético isomórfico a la BD prenatal**: mismos nombres de entidad, mismos campos y mismas reglas de derivación. El clustering no se hará sobre un CSV monolítico con nombres distintos, sino sobre una **vista analítica** (`ClusteringFeatureView`) construida exactamente igual si los datos vienen del generador sintético o de la BD real.

**Principio rector:** la BD define el vocabulario; el generador sintético lo reproduce; el pipeline de clustering consume la vista común.

**Alcance conservado de v1:**
- Sin variables de laboratorio ni fetales.
- Contaminación por outliers y missingness MCAR / MAR / MNAR.
- Ground truth de clusters para evaluación.
- N = 2 000 pacientes (configurable).

**Cambio principal respecto a v1:** pasar de un CSV plano en español a **cuatro tablas normalizadas + vista de clustering + metadatos**.

---

## 2. Arquitectura de datos unificada

### 2.1 Modelo entidad-relación (BD = dataset)

```
Patient (1) ──────< MedicalRecord (1) ──────< PatientDiary (N)
                         │
                         └──────< Consultation (N)   [opcional en v2, escasa]
```

| Entidad | Rol | Granularidad v2 sintética |
|---------|-----|---------------------------|
| `Patient` | Capa estática demográfica y antropométrica base | 1 fila / paciente |
| `MedicalRecord` | Factores de riesgo booleanos y obstétricos | 1 fila / paciente |
| `PatientDiary` | Monitoreo vitals + peso | 1 fila / paciente (snapshot T2–T3) o 3–6 filas (longitudinal v2.1) |
| `Consultation` | Eventos SOAP | 0–1 fila / paciente (mayoría vacía) |

### 2.2 Vista analítica para clustering

Vista lógica **`ClusteringFeatureView`** = join de las tres tablas principales en el punto de observación elegido (última bitácora o bitácora de semana objetivo).

Campos expuestos a clustering (sin IDs ni metadatos de evaluación):

```
age_years, bmi_initial, gestational_week, gestational_trimester,
height_cm, initial_weight, weight_kg, weight_gain,
systolic, diastolic, mean_arterial_pressure,
diabetes, chronic_hypertension, previous_preeclampsia,
family_history_hypertension, family_history_heart_disease,
chronic_kidney_disease, multiple_pregnancy, active_smoking,
previous_pregnancies, previous_deliveries, previous_miscarriages,
previous_cesareans, nulliparous,
education_level, residence, marital_status
```

**Campos eliminados respecto a v1** (no existen en BD y no se añaden):

| Campo v1 | Decisión |
|----------|----------|
| `suplemento_acido_folico`, `suplemento_hierro` | Eliminar (C3 se redefine sin suplementos) |
| `frecuencia_cardiaca` | Eliminar (no está en BD) |
| `paciente_id` español | → `patient_id` |
| Nombres en español | → nomenclatura BD en inglés snake_case |

**Campos nuevos** (presentes en BD o añadidos en extensión v2):

| Campo | Entidad | Justificación clínica |
|-------|---------|----------------------|
| `previous_preeclampsia` | MedicalRecord | RR 7.19 — gap crítico v1 |
| `family_history_hypertension` | MedicalRecord | C2 |
| `family_history_heart_disease` | MedicalRecord | RR 2.0–3.0 |
| `chronic_kidney_disease` | MedicalRecord | RR 1.80 — C4 |
| `previous_miscarriages`, `previous_cesareans` | MedicalRecord | C4 obstétrico |
| `nulliparous` | derivado | RR 2.91 nuliparidad — C2 |
| `bmi_initial` | Patient (derivado) | explícito en BD |
| `weight_gain` | PatientDiary (derivado) | explícito en BD |
| `mean_arterial_pressure` | PatientDiary (derivado) | MAP = (s + 2d) / 3 |
| `gestational_week` | PatientDiary (derivado) | alinea con `gestational_week_at_log` |
| `gestational_trimester` | derivado | T1/T2/T3 |
| `marital_status` | Patient | ya en BD, proxy socioeconómico |

---

## 3. Modificaciones propuestas a la BD (`bd_salud_pretanal.md`)

Estas extensiones aplican tanto al esquema real como al sintético.

### 3.1 Patient — campos derivados persistidos (opcional en BD real, obligatorio en sintético)

| Campo nuevo | Tipo | Fórmula |
|-------------|------|---------|
| `bmi_initial` | Float | `initial_weight / (height_cm/100)²` |
| `age_at_registration` | Integer | años desde `birthdate` a fecha de registro |

### 3.2 MedicalRecord — enriquecimiento de paridad

Sustituir solo-booleanos por **conteos enteros** donde el clustering necesita granularidad, manteniendo booleanos para compatibilidad:

| Campo nuevo | Tipo | Relación |
|-------------|------|----------|
| `parity_count` | Integer 0–10 | número de embarazos previos |
| `delivery_count` | Integer 0–parity_count | partos previos |
| `miscarriage_count` | Integer 0–5 | abortos previos |
| `cesarean_count` | Integer 0–delivery_count | cesáreas previas |

Los booleanos `previous_pregnancies`, `previous_deliveries`, etc. se derivan: `previous_pregnancies = parity_count > 0`.

### 3.3 PatientDiary — campos derivados

| Campo nuevo | Tipo | Fórmula |
|-------------|------|---------|
| `weight_gain` | Float | `weight_kg - Patient.initial_weight` |
| `mean_arterial_pressure` | Float | `(systolic + 2 * diastolic) / 3` |
| `gestational_week` | Float | semanas desde FUM o propagadas |
| `gestational_trimester` | Integer 1–3 | derivado de semana |
| `weekly_weight_gain_rate` | Float | `weight_gain / gestational_week` (proxy dinámico en snapshot) |

### 3.4 Nueva entidad lógica: ClusteringFeatureView

No es tabla transaccional; es **vista SQL / CSV exportado** documentada en BD. Definición:

```sql
-- Pseudocódigo de la vista
SELECT
  p.patient_id,
  p.age_at_registration AS age_years,
  p.bmi_initial,
  p.height_cm,
  p.initial_weight,
  p.education_level,
  p.residence,
  p.marital_status,
  mr.diabetes,
  mr.chronic_hypertension,
  mr.previous_preeclampsia,
  mr.family_history_hypertension,
  mr.family_history_heart_disease,
  mr.chronic_kidney_disease,
  mr.multiple_pregnancy,
  mr.active_smoking,
  mr.parity_count AS previous_pregnancies,
  mr.delivery_count AS previous_deliveries,
  mr.miscarriage_count AS previous_miscarriages,
  mr.cesarean_count AS previous_cesareans,
  (mr.parity_count = 0) AS nulliparous,
  pd.gestational_week,
  pd.gestational_trimester,
  pd.weight_kg,
  pd.weight_gain,
  pd.weekly_weight_gain_rate,
  pd.systolic,
  pd.diastolic,
  pd.mean_arterial_pressure
FROM Patient p
JOIN MedicalRecord mr ON mr.patient_id = p.patient_id
JOIN PatientDiary pd ON pd.medical_record_id = mr.medical_record_id
WHERE pd.is_clustering_snapshot = TRUE;   -- flag en bitácora elegida
```

### 3.5 Flag en PatientDiary

| Campo | Tipo | Uso |
|-------|------|-----|
| `is_clustering_snapshot` | Boolean | Marca la fila de bitácora usada para clustering (1 por paciente en v2) |
| `diary_id` | Integer PK | Identificador de registro |

---

## 4. Perfiles de cluster v2 (mapeados a campos BD)

Distribución objetivo (igual que v1, nomenclatura clínica refinada):

| ID | Nombre v2 | Proporción | Campos dominantes en BD |
|----|-----------|------------|-------------------------|
| C0 | Control prenatal sin riesgo mayor | 45 % | flags MR en false; PA normal; BMI 20–26 |
| C1 | Riesgo cardiometabólico | 18 % | `bmi_initial` ≥ 30 o ≥ 27 + `diabetes`; `weight_gain` alta |
| C2 | Perfil hipertensivo / riesgo preeclampsia | 12 % | `systolic` ≥ 140 o `diastolic` ≥ 90 si sem ≥ 20; `chronic_hypertension`, `previous_preeclampsia`, `nulliparous` |
| C3 | Déficit antropométrico-nutricional | 13 % | `bmi_initial` < 18.5; `weight_gain` baja; `education_level` bajo |
| C4 | Muy alto riesgo obstétrico | 7 % | `multiple_pregnancy`, edad ≥ 38, `chronic_kidney_disease`, comorbilidades múltiples |
| C5 | Perfil mixto / frontera | 5 % | mezcla de centroides C1–C2 o C2–C4 |

### 4.1 C0 — Sin riesgo mayor

```
age_years: 22–35
bmi_initial: 20–26
gestational_week: 24–38 (snapshot en T2–T3 estable)
systolic: 100–120, diastolic: 60–75
Todos los booleanos de riesgo en MedicalRecord = false
nulliparous: puede ser 0 o 1 con igual probabilidad
```

### 4.2 C1 — Riesgo cardiometabólico

```
bmi_initial: ≥ 30 (70 %) o 27–29.9 con diabetes (30 %)
diabetes: true en ≥ 40 % del cluster
weight_gain: percentil alto para semana gestacional
weekly_weight_gain_rate: > 0.5 kg/sem en T3 (doc. preeclampsia)
family_history_heart_disease: elevado vs C0
systolic: 115–135 (alto-normal)
```

### 4.3 C2 — Riesgo hipertensivo / preeclampsia

```
gestational_week: ≥ 20 en ≥ 80 % (criterio temporal PE)
systolic: 140–165, diastolic: 90–105 (umbral diagnóstico doc. PE)
chronic_hypertension: true en 50 %
previous_preeclampsia: true en 25 % (RR alto)
nulliparous: true en 45 %
family_history_hypertension: true en 35 %
age_years: > 35 en 40 % o nulliparous joven
weight_gain: moderada a baja (restricción clínica)
```

### 4.4 C3 — Déficit antropométrico

```
bmi_initial: 15.5–18.4
weight_gain: por debajo de banda IOM
weekly_weight_gain_rate: < 0.2 kg/sem
education_level: primaria frecuente
residence: rural frecuente
Eje ortogonal a preeclampsia — no forzar PA elevada
```

### 4.5 C4 — Muy alto riesgo

```
multiple_pregnancy: true en 45 %
age_years: ≥ 38
chronic_kidney_disease: true en 20 %
previous_preeclampsia + chronic_hypertension: co-ocurrencia en 15 %
diabetes + chronic_hypertension: co-ocurrencia en 20 %
gestational_week: distribución bimodal (32–36 o 24–32 pretérmino)
family_history_heart_disease: elevado
```

### 4.6 C5 — Frontera

```
Mezcla 50/50 de perfiles adyacentes (C1↔C2 o C2↔C4)
Ruido gaussiano elevado en systolic, bmi_initial, weight_gain
```

---

## 5. Reglas duras v2 (validación post-generación)

| # | Regla | Campos |
|---|-------|--------|
| R1 | `systolic > diastolic + 15` | PatientDiary |
| R2 | `delivery_count ≤ parity_count` | MedicalRecord |
| R3 | `cesarean_count ≤ delivery_count` | MedicalRecord |
| R4 | `bmi_initial = initial_weight / (height_cm/100)²` ± 0.3 | Patient |
| R5 | `weight_gain = weight_kg - initial_weight` ± 0.5 | PatientDiary |
| R6 | `gestational_trimester` coherente con `gestational_week` | PatientDiary |
| R7 | Si `diabetes = true` → `bmi_initial ≥ 27` (salvo outlier) | Patient + MR |
| R8 | Si `chronic_hypertension = true` → `systolic` desplazada +10 mmHg mínimo | MR + PD |
| R9 | Si C2 y `gestational_week ≥ 20` → al menos uno: `systolic ≥ 140` o `diastolic ≥ 90` | Vista |
| R10 | `previous_preeclampsia = true` → `parity_count ≥ 1` | MR |
| R11 | `nulliparous = (parity_count = 0)` | derivado |
| R12 | MAP recalculable desde systolic/diastolic | PatientDiary |

---

## 6. Pipeline de generación v2

```
[Paso 0]  Configuración (semilla, N, proporciones, tasas outlier/missing)
    ↓
[Paso 1]  Asignar cluster_verdadero (0–5) por paciente
    ↓
[Paso 2]  Generar tabla Patient (estática)
    ↓
[Paso 3]  Generar MedicalRecord condicionado a cluster + reglas RR
    ↓
[Paso 4]  Generar PatientDiary (snapshot is_clustering_snapshot=true)
    ↓
[Paso 5]  Calcular campos derivados (BMI, weight_gain, MAP, trimestre, nulliparous)
    ↓
[Paso 6]  Aplicar reglas duras R1–R12
    ↓
[Paso 7]  (Opcional) Generar 0–1 Consultation escueta
    ↓
[Paso 8]  Materializar ClusteringFeatureView
    ↓
[Paso 9]  Inyectar outliers sobre tablas + vista
    ↓
[Paso 10] Inyectar MCAR / MAR / MNAR (patrones alineados a entidad)
    ↓
[Paso 11] Validación automática + reporte
    ↓
[Paso 12] Exportar CSV por entidad + vista + ground truth
```

### 6.1 Orden de generación intra-paciente

1. `birthdate` → `age_at_registration`
2. `height_cm`, `initial_weight` → `bmi_initial`
3. Flags `MedicalRecord` según cluster y tabla RR (pesos probabilísticos)
4. `gestational_week` del snapshot → `gestational_trimester`
5. `weight_kg`, `systolic`, `diastolic` condicionados a cluster
6. Derivados: `weight_gain`, `mean_arterial_pressure`, `weekly_weight_gain_rate`

### 6.2 Pesos RR en generación (MedicalRecord)

Usar riesgos relativos de `documentacion-preeclampsia.md` como **multiplicadores de log-odds** base por cluster:

```
logit(p_previous_preeclampsia) = logit_base_Ck + 2.0 * I(cluster ∈ {C2,C4})
logit(p_chronic_hypertension)  = logit_base_Ck + 1.5 * I(cluster ∈ {C2,C4})
logit(p_diabetes)              = logit_base_Ck + 1.3 * I(cluster ∈ {C1,C4})
logit(p_multiple_pregnancy)    = logit_base_Ck + 1.1 * I(cluster = C4)
logit(p_nulliparous)           = logit_base_Ck + 1.0 * I(cluster = C2)
```

---

## 7. Missingness v2 por entidad

### 7.1 MCAR (2–4 % celda)

Variables elegibles en `PatientDiary`: `systolic`, `diastolic`, `weight_kg`  
Variables elegibles en `Patient`: `height_cm`, `initial_weight`  
Variables elegibles en `MedicalRecord`: conteos de paridad

### 7.2 MAR

| Variable | Condición observada → mayor P(missing) |
|----------|----------------------------------------|
| `diastolic` | `systolic < 120` (control rápido) |
| `weight_gain` | `gestational_trimester = 1` |
| `delivery_count` | `parity_count = 0` |
| `marital_status` | `education_level = superior` (dato admin omitido) |
| `gestational_week` | registro temprano incompleto |

### 7.3 MNAR

| Variable | Mecanismo |
|----------|-----------|
| `weight_kg` | mayor P si `bmi_initial > 35` |
| `systolic` | mayor P si valor > 160 |
| `previous_preeclampsia` | mayor P si true (estigma / omisión) |
| `chronic_kidney_disease` | mayor P si true |

---

## 8. Outliers v2

Mismos cinco mecanismos de v1, aplicados sobre campos BD:

| Mecanismo | Ejemplo en vocabulario BD |
|-----------|---------------------------|
| `medicion` | `systolic = 210` en C0 |
| `registro` | `height_cm = 1700` |
| `combinacion_imposible` | `bmi_initial = 38`, `weight_gain = -8`, `gestational_week = 39` |
| `patologia_rara` | `systolic = 185`, `multiple_pregnancy = true`, `age_years = 17` |
| `posicion` | C1 con PA normal |

Tasa objetivo: 5 % filas/pacientes; 1.5 % extremos.

---

## 9. Archivos de salida v2

```
data/v2/
  patient.csv
  medical_record.csv
  patient_diary.csv
  consultation.csv                    # puede estar vacío o escaso
  clustering_feature_view.csv         # entrada principal al clustering
  clustering_feature_view_completo.csv  # sin nulos ni outliers
  ground_truth.csv                    # patient_id, cluster_verdadero, es_outlier, missingness_log
  missingness_log.csv
  config_generacion_v2.json
  reporte_calidad_v2.md
  ddl_v2.sql                          # DDL de referencia alineado a BD
```

### 9.1 ground_truth.csv

| Campo | Descripción |
|-------|-------------|
| `patient_id` | PK |
| `medical_record_id` | FK |
| `diary_id` | FK del snapshot |
| `cluster_verdadero` | 0–5 |
| `cluster_nombre` | etiqueta legible |
| `es_outlier` | 0/1 |
| `tipo_outlier` | mecanismo |
| `mecanismo_nulo_json` | JSON por variable |

---

## 10. Uso conjunto BD real + dataset sintético

| Escenario | Flujo |
|-----------|-------|
| Desarrollo clustering | Entrenar/explorar con `clustering_feature_view.csv` sintético |
| Validación algoritmo | Comparar ARI/NMI contra `cluster_verdadero` |
| Integración futura BD real | Misma vista SQL sobre PostgreSQL/MySQL; mismos nombres |
| EDA | Notebooks leen `clustering_feature_view.csv` o join manual de tablas |
| Imputación | `missingness_log.csv` + vista completa |

**Contrato de clustering:** el algoritmo solo recibe columnas de `ClusteringFeatureView` excluyendo IDs. La BD real y el sintético deben producir exactamente las mismas columnas en el mismo orden (documentado en `ddl_v2.sql`).

---

## 11. Estructura de código propuesta

```
clustering/
  documentacion/
    plan-dataset-sintetico-embarazo-v2.md   ← este archivo
    bd_salud_pretanal.md                    ← actualizado sección clustering
    ddl_v2.sql                              ← generado en implementación
  src/v2/
    config_v2.py
    schema.py              # definición campos por entidad
    clusters_v2.py         # perfiles + RR weights
    generador_patient.py
    generador_medical_record.py
    generador_patient_diary.py
    derivados.py           # BMI, MAP, weight_gain, trimestre
    vista_clustering.py    # materializa ClusteringFeatureView
    outliers_v2.py
    missingness_v2.py
    validacion_v2.py
    main_generar_v2.py
  data/v2/
```

---

## 12. Validación v2

### 12.1 Checks automáticos

- Reglas R1–R12 cumplidas (salvo outliers marcados).
- Proporción clusters ± 2 % de objetivo.
- Columnas de vista idénticas a contrato DDL.
- C2 con gestational_week ≥ 20: ≥ 75 % cumple umbral PA 140/90.
- Correlaciones objetivo:
  - `bmi_initial` ↔ `weight_gain`: 0.35–0.65
  - `gestational_week` ↔ `weight_gain`: 0.45–0.75
  - `systolic` ↔ `diastolic`: 0.65–0.80

### 12.2 Métricas clustering

- KMeans k=6 sobre vista estandarizada vs `cluster_verdadero`: ARI > 0.45 (objetivo v2, subió por mejor separación C2).
- Silhouette > 0.12 en datos completos sin outliers.

---

## 13. Configuración JSON v2 (referencia)

```json
{
  "version": "2.0",
  "semilla": 42,
  "n_pacientes": 2000,
  "proporciones_cluster": [0.45, 0.18, 0.12, 0.13, 0.07, 0.05],
  "snapshot": {
    "gestational_week_range": [14, 38],
    "prefer_trimester": [2, 3]
  },
  "outliers": {
    "tasa_fila": 0.05,
    "tasa_extremo": 0.015
  },
  "missingness": {
    "mcar_prob_celda": 0.03,
    "mar_prob_base": 0.08,
    "mnar_prob_base": 0.06
  },
  "export": {
    "include_consultation": false,
    "idioma_columnas": "en",
    "vista_clustering": "clustering_feature_view.csv"
  }
}
```

---

## 14. Migración desde v1

| v1 (`datos_embarazo_sintetico.csv`) | v2 (BD alineado) |
|-------------------------------------|------------------|
| `paciente_id` | `patient_id` |
| `edad_anios` | `age_years` (Patient.age_at_registration) |
| `imc_pregestacional` | `bmi_initial` |
| `semanas_gestacion` | `gestational_week` |
| `trimestre` | `gestational_trimester` |
| `peso_kg` | `weight_kg` (PatientDiary) |
| `talla_cm` | `height_cm` (Patient) |
| `ganancia_peso_kg` | `weight_gain` (PatientDiary) |
| `presion_sistolica` | `systolic` |
| `presion_diastolica` | `diastolic` |
| `diabetes_previa` | `diabetes` |
| `hipertension_cronica` | `chronic_hypertension` |
| `embarazo_multiple` | `multiple_pregnancy` |
| `tabaquismo_activo` | `active_smoking` |
| `num_embarazos_previos` | `parity_count` / `previous_pregnancies` |
| `num_partos_previos` | `delivery_count` / `previous_deliveries` |
| `nivel_educacion` | `education_level` |
| `area_residencia` | `residence` |
| — | `previous_preeclampsia`, `family_history_*`, `chronic_kidney_disease`, `nulliparous`, `mean_arterial_pressure` |

Notebooks EDA y clustering deberán apuntar a `data/v2/clustering_feature_view.csv`.

---

## 15. Próximos pasos

1. Aprobar modificaciones a `bd_salud_pretanal.md` (sección 5 ya añadida).
2. Generar `ddl_v2.sql` con tipos y FKs.
3. Implementar `src/v2/` según pipeline sección 6.
4. Ejecutar generación y validar reporte v2.
5. Actualizar notebooks en `notebooks/` para leer vista v2.
6. (Opcional v2.1) PatientDiary longitudinal con múltiples snapshots por paciente.

---

## Apéndice A — Ecuaciones (referencia, sin LaTeX)

```
bmi_initial = initial_weight / (height_cm / 100)^2

weight_gain = weight_kg - initial_weight

mean_arterial_pressure = (systolic + 2 * diastolic) / 3

weekly_weight_gain_rate = weight_gain / gestational_week

gestational_trimester = 1  si gestational_week <= 13
                        2  si gestational_week <= 27
                        3  si gestational_week > 27

nulliparous = 1  si parity_count = 0,  else 0

logit(p_flag) = beta_0 + beta_cluster[k] + sum_j (w_j * RR_j)
```

---

*Plan v2 — BD prenatal y dataset sintético en vocabulario unificado para clustering.*
