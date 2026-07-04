# Estructura de Base de Datos - Salud Prenatal

Este documento describe la estructura del dominio de datos del sistema de control prenatal. Está diseñado para proveer contexto a agentes de análisis de datos y machine learning sobre las variables disponibles para modelado, predicción de riesgos (como preeclampsia) y análisis de series de tiempo.

## 1. Entidad Base: Paciente (`Patient`)
Contiene la información demográfica, biométrica base y antecedentes ginecológicos iniciales. Es información estática o de configuración inicial.

*   `patient_id` (Integer): Identificador único del paciente.
*   `birthdate` (Date): Fecha de nacimiento (útil para derivar la **edad** al momento de la gestación).
*   `blood_type` (Enum): Tipo de sangre.
*   `weeks_at_registration` (Integer): Semana de gestación en la que ingresa al sistema.
*   `last_menstrual_period` (Date): Fecha de la última menstruación (FUM, vital para calcular la edad gestacional en curso).
*   `residence` (String): Lugar de residencia.
*   `education_level` (String): Nivel educativo (variable socioeconómica).
*   `marital_status` (String): Estado civil (variable socioeconómica).
*   `height_cm` (Integer): Altura en centímetros.
*   `initial_weight` (Float): Peso pre-gestacional o inicial en kilogramos. (Junto con la altura permite derivar el **IMC inicial**).

## 2. Entidad Clínica: Expediente Médico (`MedicalRecord`)
Actúa como el registro maestro clínico de los factores de riesgo de la paciente. Establece la relación base entre un paciente y su médico (`patient_id`, `doctor_id`). Todas las variables aquí suelen ser booleanas (presencia/ausencia de factores de riesgo).

**Historial Médico General:**
*   `previous_hypertension` (Boolean): Hipertensión detectada previamente.
*   `diabetes` (Boolean): Presencia de diabetes.
*   `family_history_hypertension` (Boolean): Familiares directos con hipertensión.
*   `family_history_heart_disease` (Boolean): Familiares con enfermedades cardíacas.

**Historial Obstétrico (Embarazos Anteriores):**
*   `previous_pregnancies` (Boolean): Si ha tenido embarazos previos.
*   `previous_deliveries` (Boolean): Partos previos.
*   `previous_miscarriages` (Boolean): Abortos previos.
*   `previous_cesareans` (Boolean): Cesáreas previas.
*   `previous_preeclampsia` (Boolean): **Variable crítica**. Antecedente de preeclampsia en embarazos pasados.

**Historial Crónico/Patológico:**
*   `chronic_kidney_disease` (Boolean): Enfermedad renal crónica.
*   `chronic_hypertension` (Boolean): Hipertensión crónica (diagnosticada antes del embarazo).
*   `multiple_pregnancy` (Boolean): Embarazo actual múltiple (ej. gemelar).
*   `fetal_death` (Boolean): Antecedente de muerte fetal.
*   `fetal_growth_restriction` (Boolean): Restricción del crecimiento fetal previa.

## 3. Entidad de Monitoreo: Bitácora (`PatientDiary`)
Contiene datos de **series de tiempo**. Esta tabla es fundamental para modelos dinámicos, detección de anomalías o predicción temprana de preeclampsia mediante el rastreo de cambios en presión arterial y aumento de peso.

*   `medical_record_id` (Integer): Relación con el expediente base.
*   `created_at` (DateTime): Fecha/hora del registro (timestamp para análisis de series de tiempo).
*   `weight_kg` (Float): Peso actual en kilogramos.
*   `systolic` (Integer): Presión arterial sistólica.
*   `diastolic` (Integer): Presión arterial diastólica.
*   `symptoms` (Text): Datos cualitativos. Malestares registrados por la paciente.
*   `notes` (Text): Notas adicionales cualitativas.

**Métricas Derivadas Sugeridas:**
*   `weight_gain` (Float): Diferencia entre `PatientDiary.weight_kg` y `Patient.initial_weight`. Indica la ganancia de peso acumulada.
*   `mean_arterial_pressure` (MAP): Se puede derivar de sistólica y diastólica.
*   `gestational_week_at_log`: Derivado usando `PatientDiary.created_at` y `Patient.last_menstrual_period` o `Patient.weeks_at_registration`.

## 4. Entidad Clínica Dinámica: Consultas (`Consultation`)
Eventos clínicos formales (SOAP) realizados por el médico. Representa intervenciones o evaluaciones diagnósticas periódicas.

*   `medical_record_id` (Integer): Relación con el expediente.
*   `created_at` (DateTime): Fecha de la consulta.
*   `reported_facts` (String): **Subjetivo** (S). Reporte de la paciente en consulta.
*   `objective` (String): **Objetivo** (O). Hallazgos clínicos del médico.
*   `plan` (String): **Plan** (P). Tratamiento o indicaciones a seguir.
*   `notes` (String): Notas adicionales.

---
**Nota para el Agente de Análisis de Datos:** 
Para la predicción de riesgos gestacionales (especialmente hipertensión gestacional y preeclampsia), debes fusionar el **riesgo base** estático (Entidades `Patient` y `MedicalRecord`) con el **monitoreo temporal** (Entidad `PatientDiary`). Las tendencias al alza en `systolic`/`diastolic` junto con un rápido aumento en `weight_gain` en los trimestres finales son los principales predictores buscados.

---

## 5. Extensión v2 — Alineación con dataset sintético y clustering

Esta sección unifica el vocabulario de la BD con el dataset sintético de prueba (`plan-dataset-sintetico-embarazo-v2.md`). Los nombres aquí son la **fuente de verdad** para ambos sistemas.

### 5.1 Campos añadidos en Patient (v2)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `bmi_initial` | Float | IMC pregestacional: `initial_weight / (height_cm/100)²` |
| `age_at_registration` | Integer | Edad en años al ingreso al control prenatal |

### 5.2 Campos añadidos en MedicalRecord (v2)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `parity_count` | Integer | Número de embarazos previos (0–10). Reemplaza lógica ambigua de solo booleanos |
| `delivery_count` | Integer | Partos previos; debe ser ≤ `parity_count` |
| `miscarriage_count` | Integer | Abortos previos |
| `cesarean_count` | Integer | Cesáreas previas; debe ser ≤ `delivery_count` |

Los booleanos existentes (`previous_pregnancies`, `previous_deliveries`, etc.) se mantienen como derivados: `previous_pregnancies = (parity_count > 0)`.

### 5.3 Campos añadidos en PatientDiary (v2)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `diary_id` | Integer | Identificador del registro de bitácora |
| `weight_gain` | Float | `weight_kg - Patient.initial_weight` |
| `mean_arterial_pressure` | Float | `(systolic + 2 * diastolic) / 3` |
| `gestational_week` | Float | Semana gestacional al momento del registro |
| `gestational_trimester` | Integer | 1, 2 o 3; derivado de `gestational_week` |
| `weekly_weight_gain_rate` | Float | `weight_gain / gestational_week` |
| `is_clustering_snapshot` | Boolean | Marca el registro usado para minería de datos (1 por paciente en análisis cross-sectional) |

### 5.4 Vista analítica: ClusteringFeatureView

Vista lógica (SQL o CSV exportado) que une `Patient`, `MedicalRecord` y el registro de `PatientDiary` donde `is_clustering_snapshot = true`.

**Columnas expuestas al clustering** (sin IDs):

`age_years`, `bmi_initial`, `height_cm`, `initial_weight`, `education_level`, `residence`, `marital_status`, `diabetes`, `chronic_hypertension`, `previous_preeclampsia`, `family_history_hypertension`, `family_history_heart_disease`, `chronic_kidney_disease`, `multiple_pregnancy`, `previous_pregnancies`, `previous_deliveries`, `previous_miscarriages`, `previous_cesareans`, `nulliparous`, `gestational_week`, `gestational_trimester`, `weight_kg`, `weight_gain`, `weekly_weight_gain_rate`, `systolic`, `diastolic`, `mean_arterial_pressure`

**Excluido del clustering:** laboratorio, variables fetales, campos SOAP de `Consultation`, `blood_type`.

### 5.5 Relación con perfiles de cluster sintéticos

| Cluster | Perfil | Señales principales en la vista |
|---------|--------|----------------------------------|
| C0 | Sin riesgo mayor | MR limpio, PA normal, BMI 20–26 |
| C1 | Riesgo cardiometabólico | `bmi_initial` ≥ 30, `diabetes`, `weight_gain` alta |
| C2 | Riesgo hipertensivo / preeclampsia | PA ≥ 140/90 post sem 20, `previous_preeclampsia`, `nulliparous` |
| C3 | Déficit antropométrico | `bmi_initial` < 18.5, `weight_gain` baja |
| C4 | Muy alto riesgo | `multiple_pregnancy`, edad ≥ 38, comorbilidades múltiples |
| C5 | Frontera | Perfil mixto entre clusters adyacentes |

### 5.6 Exportación sintética (archivos CSV v2)

El generador produce tablas homónimas a la BD más:

- `clustering_feature_view.csv` — entrada al pipeline de clustering
- `ground_truth.csv` — `cluster_verdadero`, flags de outlier y missingness

El dataset sintético y la BD real deben poder ejecutar la **misma definición de vista** para que el clustering hable un solo idioma.
