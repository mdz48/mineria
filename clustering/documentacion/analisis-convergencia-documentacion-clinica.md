# Análisis de convergencia: documentación clínica verídica vs plan de clusters sintéticos

**Fecha de análisis:** junio 2026  
**Documentos revisados:**

| Documento | Rol |
|-----------|-----|
| `plan-dataset-sintetico-embarazo.md` | Plan original: clusters, variables, restricciones, contaminación |
| `bd_salud_pretanal.md` | Estructura verídica del dominio (entidades Patient, MedicalRecord, PatientDiary, Consultation) |
| `documentacion-preeclampsia.md` | Especificación clínica de preeclampsia: target, umbrales, factores de riesgo con RR |

**Dataset generado de referencia:** `data/datos_embarazo_sintetico.csv` (N=2000, cross-sectional)

---

## 1. Conclusión ejecutiva

La documentación nueva **refuerza y valida parcialmente** el diseño de clusters que ya teníamos, sobre todo para **C1 (metabólico)**, **C2 (hipertensivo / preeclampsia)** y **C4 (alto riesgo obstétrico)**. La base de datos prenatal verídica confirma que las variables que elegimos son las correctas del dominio real, aunque en una granularidad más rica (estática + longitudinal).

La documentación de preeclampsia aporta **criterios diagnósticos cuantitativos** (PA ≥ 140/90 mmHg, aparición después de semana 20, incidencia ~10 %) y una **jerarquía de factores de riesgo** con riesgo relativo (RR) que permite priorizar qué variables deben dominar cada cluster.

El punto más débil de la convergencia es **C3 (bajo peso nutricional)**: no aparece como perfil de riesgo relevante en la literatura de preeclampsia aportada. Sigue siendo clínicamente válido como grupo antropométrico, pero queda **desconectado del eje hipertensivo** que centra la documentación nueva.

En conjunto: **sí sirve para definir y afinar clusters**, pero conviene **revisar nomenclatura, umbrales de PA en C2** y **planificar variables faltantes** para una v2 del generador.

---

## 2. Qué aporta cada documento nuevo

### 2.1 `bd_salud_pretanal.md` — Modelo de dominio real

Describe cuatro capas de información:

```
Patient (estática)          →  edad, talla, peso inicial, IMC, educación, residencia
MedicalRecord (riesgo base) →  HTA crónica, diabetes, embarazo múltiple, antecedentes obstétricos
PatientDiary (serie temporal) →  peso, sistólica, diastólica, ganancia acumulada
Consultation (eventos SOAP) →  notas clínicas cualitativas
```

**Aporte principal:** confirma que el dominio prenatal real separa **riesgo basal estático** de **evolución temporal**. La nota final del documento es explícita: para predecir preeclampsia hay que fusionar ambas capas, y los predictores dinámicos clave son **tendencia al alza de PA** y **aumento rápido de ganancia de peso** en trimestres finales.

**Implicación para nuestro dataset:** generamos un snapshot cross-sectional (una fila = una visita). Eso es compatible con una fila de `PatientDiary` enriquecida con datos de `Patient` y `MedicalRecord`, pero **no captura tendencias temporales**. Para clustering esto es aceptable; para predicción de preeclampsia sería insuficiente.

### 2.2 `documentacion-preeclampsia.md` — Perfil clínico cuantificado

Aporta tres piezas accionables:

1. **Definición del target:** preeclampsia = hipertensión (sistólica ≥ 140 o diastólica ≥ 90) después de semana 20, con condiciones adversas asociadas (en el doc original se mencionan marcadores clínicos; nosotros excluimos laboratorio, por lo que proteinuria no está disponible).

2. **Prevalencia base:** ~10 % de gestaciones → referencia para calibrar proporción del cluster de riesgo hipertensivo si se interpreta como “perfil compatible con preeclampsia”.

3. **Tabla de RR:** priorización objetiva de features.

### 2.3 Checklists de Riesgo (ISSHP, ACOG, NICE) — Consenso Internacional

Se aporta una tabla de directrices internacionales que clasifica los factores en Alto y Moderado riesgo:
- **Alto Riesgo:** Preeclampsia previa, Enfermedad renal crónica, Hipertensión crónica, Diabetes mellitus, Enfermedades autoinmunes (SLE/APS), IMC ≥ 30 (ISSHP), Embarazo múltiple (ACOG).
- **Riesgo Moderado:** **Primer embarazo (primigesta)**, Edad ≥ 40 o 35, Embarazo múltiple (ISSHP), IMC ≥ 35 o 30, **Intervalo intergenésico > 10 años**, Historia familiar de preeclampsia.

**Aporte principal:** Esta tabla responde directamente la relevancia del historial obstétrico: **sí importa muchísimo saber si es multípara o unipara**. El mero hecho de ser "Primer embarazo" ("First pregnancy") está clasificado unánimemente como un factor de riesgo moderado por las 3 organizaciones médicas más grandes. El algoritmo K-Means descubrió esto por sí solo al partir geográficamente a las pacientes en el eje PC2 basándose en si eran primigestas o no. Clínicamente, impacta más esta distinción binaria (nuliparidad) que llevar un conteo exacto de 3, 4 o 5 embarazos previos (aunque un intervalo > 10 años entre embarazos también añade riesgo).

---

## 3. Mapeo variable a variable

### 3.1 Cobertura directa (ya en nuestro dataset)

| Variable sintética | Entidad verídica | Doc. preeclampsia | Cluster(es) donde domina |
|--------------------|------------------|-------------------|--------------------------|
| `edad_anios` | derivada de `birthdate` | RR 1.68–1.96 si ≥ 40 años | C2, C4 |
| `imc_pregestacional` | derivada de `height_cm` + `initial_weight` | RR 2.47 si IMC > 30 | C1, C2 |
| `semanas_gestacion` | `weeks_at_registration` / FUM | Preeclampsia post semana 20 | Todos; crítico en C2 |
| `talla_cm`, `peso_kg` | `Patient` | vía IMC | Todos |
| `ganancia_peso_kg` | derivada en `PatientDiary` | > 0.5 kg/semana como señal | C1 (exceso), C3 (déficit) |
| `presion_sistolica`, `presion_diastolica` | `PatientDiary` | Target ≥ 140/90; temprana > 130 | C2 |
| `diabetes_previa` | `MedicalRecord.diabetes` | RR 3.56–3.7 | C1, C4 |
| `hipertension_cronica` | `MedicalRecord.chronic_hypertension` | RR 3.6–5.4 | C2, C4 |
| `embarazo_multiple` | `MedicalRecord.multiple_pregnancy` | Alto riesgo (ACOG) / Moderado (ISSHP) | C4 |
| `num_embarazos_previos`, `num_partos_previos` | historial obstétrico | **"First pregnancy"** es factor moderado unánime (ISSHP, ACOG, NICE). Importa la nuliparidad más que el conteo exacto. | C2 (primigesta), C4 (multípara) |
| `nivel_educacion`, `area_residencia` | `Patient` | proxy socioeconómico | C3 |

**Veredicto:** ~70 % de las variables del plan tienen análogo directo en la BD verídica. El diseño del esquema fue acertado en su núcleo antropométrico-vital.

### 3.2 Variables verídicas ausentes en nuestro dataset

| Variable verídica | RR / relevancia | Impacto en clusters |
|-------------------|-----------------|---------------------|
| `previous_preeclampsia` | **RR 7.19** (la más alta después de autoinmunes) | Debería enriquecer **C2** y **C4**; hoy no modelada |
| `family_history_hypertension` | contexto familiar | C2 |
| `family_history_heart_disease` | RR 2.0–3.0 | C2, C4 |
| `chronic_kidney_disease` | RR 1.80 | C4 |
| Enfermedades autoinmunes (LES, antifosfolípidos) | RR 2.5–9.72 | C4 (alto riesgo) |
| `previous_miscarriages`, `previous_cesareans` | obstétrico | C4 |
| Etnia de riesgo | RR 2.0–4.0 | no modelada |
| Tendencia temporal de PA y peso | predictor dinámico principal | no modelable en cross-sectional v1 |
| Proteinuria / marcadores lab | parte del diagnóstico PE | excluido por decisión de alcance |

**Veredicto:** el generador actual **subrepresenta el riesgo de preeclampsia recurrente** (`previous_preeclampsia`) y **comorbilidades renales/autoinmunes**, que en la documentación clínica separan claramente un subgrupo de muy alto riesgo dentro de lo que hoy agrupamos en C4.

### 3.3 Variables nuestras sin respaldo fuerte en doc. preeclampsia

| Variable sintética | Observación |
|--------------------|-------------|
| `tabaquismo_activo` | no aparece en tabla de RR de preeclampsia |
| `suplemento_acido_folico`, `suplemento_hierro` | útiles para C3 nutricional, no para eje hipertensivo |
| `frecuencia_cardiaca` | presente en signos vitales generales, no en RR de PE |
| Cluster **C3** completo | perfil antropométrico válido en obstetricia general, **no anclado** en la doc. de preeclampsia |

---

## 4. Análisis cluster por cluster

### C0 — Bajo riesgo (~45 %)

| Aspecto | Evaluación |
|---------|------------|
| Alineación clínica | **Alta.** Inverso de los factores de RR elevado: IMC normal, PA normal, sin comorbilidades, edad 22–35. |
| Doc. preeclampsia | Pacientes sin HTA crónica, sin diabetes, sin embarazo múltiple, IMC < 30, edad moderada. |
| Doc. BD prenatal | Corresponde a `MedicalRecord` con flags en false y `PatientDiary` con PA y peso estables. |
| Ajuste sugerido | Mantener. Proporción ~45 % es mayor que prevalencia inversa de PE (~90 % sin PE), razonable para clustering. |

### C1 — Riesgo metabólico (~18 %)

| Aspecto | Evaluación |
|---------|------------|
| Alineación clínica | **Alta.** IMC ≥ 30 (RR 2.47) y diabetes pregestacional (RR 3.56–3.7) están explícitos en ambas fuentes. |
| Ganancia de peso excesiva | Doc. PE: capturar > 0.5 kg/semana; nuestro C1 tiene ganancia alta → coherente como factor de estrés metabólico. |
| Ajuste sugerido | Mantener. Opcional: renombrar a **“Riesgo cardiometabólico”** para alinearlo con terminología de la BD (`family_history_heart_disease` como variable futura). |

### C2 — Riesgo hipertensivo / preeclampsia (~12 %)

| Aspecto | Evaluación |
|---------|------------|
| Alineación clínica | **Media-alta**, con matices importantes. |
| Umbrales de PA | Plan original: sistólica 130–160, diastólica 85–105. Doc. PE diagnóstico: **≥ 140/90**. Nuestro rango incluye PA “elevada” pre-clínica (130–139), útil para clustering de **riesgo**, no solo casos diagnosticados. |
| Factores de riesgo | HTA crónica (RR 3.6–5.4), edad > 35, primigesta (RR 2.91) → ya modelados. |
| Ausencias críticas | Falta `previous_preeclampsia` (RR 7.19) y PA temprana > 130 antes de sem 20 (RR 2.40). |
| Prevalencia | Cluster al 12 % vs incidencia PE ~10 % → **proporción razonable** si C2 = “perfil de riesgo hipertensivo”, no solo PE confirmada. |
| Ajuste sugerido | **Renombrar** a **“Perfil de riesgo hipertensivo / compatible con preeclampsia”**. Subir piso de PA a ≥ 135/85 en T2–T3 para acercarse al umbral clínico. Planificar `preeclampsia_previa` en v2. |

### C3 — Bajo peso nutricional (~15 %)

| Aspecto | Evaluación |
|---------|------------|
| Alineación clínica | **Baja** respecto a documentación de preeclampsia. |
| Doc. PE | IMC bajo no figura como factor de riesgo principal; el eje es hipertensión + comorbilidades. |
| Doc. BD prenatal | IMC derivable, ganancia de peso en bitácora → válido como fenómeno obstétrico general. |
| Valor para clustering | **Sí.** Aporta heterogeneidad antropométrica y separación en el espacio IMC–ganancia, visible en EDA (C3: IMC 17.5, ganancia −1.2 kg). |
| Ajuste sugerido | Mantener como cluster, pero documentar que es **ortogonal al eje preeclampsia**. No forzar interpretación hipertensiva. |

### C4 — Alto riesgo obstétrico (~7 %)

| Aspecto | Evaluación |
|---------|------------|
| Alineación clínica | **Alta**, pero incompleta. |
| Doc. PE | Embarazo múltiple (RR 2.90), edad ≥ 40, diabetes + HTA simultáneas, ERC, autoinmunes → subconjunto de “muy alto riesgo”. |
| Nuestro perfil | Captura edad avanzada, múltiple, comorbilidades múltiples, pretérmino. |
| Ausencias | Autoinmunes, ERC, antecedente PE previo, historia cardiovascular familiar. |
| Ajuste sugerido | Mantener y en v2 ** subdividir lógicamente** o enriquecer con flags de comorbilidad severa. Proporción 7 % es plausible para cola de riesgo. |

### C5 — Residual / transición (~3 %)

| Aspecto | Evaluación |
|---------|------------|
| Alineación clínica | **Neutra.** No contradice la documentación clínica. |
| Valor | Pacientes en frontera entre perfiles (p. ej. IMC limítrofe + PA normal) existen en la práctica. |
| Ajuste sugerido | Mantener para evaluar robustez del clustering. |

---

## 5. Coherencia de restricciones y mecanismos

| Elemento del plan | ¿Respaldado por doc. verídica? | Nota |
|-------------------|-------------------------------|------|
| PA sistólica > diastólica + margen | Sí | Estándar clínico |
| HTA crónica → PA desplazada | Sí | RR 3.6–5.4 en doc. PE |
| Diabetes → IMC alto | Sí | RR 3.56–3.7 |
| Embarazo múltiple → pretérmino | Sí | coherente con alto riesgo obstétrico |
| Ganancia de peso vs semanas | Parcial | doc. PE sugiere > 0.5 kg/sem; nuestro generador tiene correlación semanas–ganancia débil (r≈0.08 en reporte) → **gap de implementación**, no de diseño |
| Exclusión de laboratorio | Tensiona doc. PE | diagnóstico completo de PE incluye proteinuria; nuestro C2 no puede simular PE confirmada, solo **riesgo hipertensivo** |
| Missingness MAR en PA diastólica si sistólica normal | Sí | coherente con control rápido en `PatientDiary` |
| Cross-sectional vs longitudinal | Tensiona doc. BD | bitácora es serie temporal; nosotros capturamos un punto |

---

## 6. Matriz de convergencia global

```
                    Doc. BD prenatal    Doc. preeclampsia    Plan / dataset actual
C0 Bajo riesgo           ✓✓                  ✓✓                  ✓✓
C1 Metabólico            ✓✓                  ✓✓                  ✓✓
C2 Hipertensivo          ✓✓                  ✓ (con matices)     ✓ (umbrales, faltan vars)
C3 Bajo peso             ✓                   ✗                   ✓ (clustering, no PE)
C4 Alto riesgo           ✓✓                  ✓ (parcial)         ✓ (faltan autoinmunes, PE previa)
C5 Residual              ✓                   —                   ✓
Variables núcleo         ✓✓                  ✓✓                  ✓✓
Variables de alto RR     parcial             ✓✓                  ✗ (gap principal)
Granularidad temporal    ✓✓                  ✓✓                  ✗ (cross-sectional)
Diagnóstico PE completo  —                   ✓                   ✗ (sin labs)
```

Leyenda: ✓✓ fuerte | ✓ parcial | ✗ no cubierto | — no aplica

---

## 7. Recomendaciones priorizadas

### Prioridad alta (afectan interpretación clínica de clusters)

1. **Documentar en el plan** que C2 representa **riesgo hipertensivo / perfil compatible con preeclampsia**, no diagnóstico confirmado (por ausencia de proteinuria y series temporales).

2. **Ajustar umbrales de PA en C2** hacia rangos más cercanos al criterio ≥ 140/90 en gestaciones ≥ 20 semanas, reservando 130–139 para C5 o transición.

3. **Agregar en v2** la variable `preeclampsia_embarazo_previo` (boolean), dado RR 7.19 — impacto directo en C2 y C4.

### Prioridad media (enriquecen realismo sin cambiar arquitectura de 6 clusters)

4. Incorporar `historia_familiar_hipertension` y `enfermedad_renal_cronica` como binarios en C4.

5. Mejorar correlación **semanas ↔ ganancia_peso** en el generador (hoy r≈0.08 vs objetivo 0.5–0.75 del plan).

6. Calibrar prevalencia conjunta de factores en C4 (múltiple + edad ≥ 40 + comorbilidad) usando tabla de RR como pesos en la generación.

### Prioridad baja (extensión futura)

7. Versión longitudinal con entidad tipo `PatientDiary` (múltiples filas por paciente).

8. Variable `etnia_riesgo` si el contexto poblacional lo requiere.

9. Mantener C3 como cluster antropométrico, explicitando que no proviene de la doc. de preeclampsia.

---

## 8. Propuesta de nomenclatura revisada (opcional)

| ID | Nombre actual | Nombre sugerido (anclado en doc. clínica) |
|----|---------------|---------------------------------------------|
| C0 | Bajo riesgo | Control prenatal sin factores de riesgo mayor |
| C1 | Riesgo metabólico | Riesgo cardiometabólico (IMC / diabetes) |
| C2 | Riesgo hipertensivo | Perfil hipertensivo / riesgo de preeclampsia |
| C3 | Bajo peso nutricional | Déficit antropométrico-nutricional |
| C4 | Alto riesgo obstétrico | Muy alto riesgo (múltiple, edad, comorbilidades) |
| C5 | Residual | Perfil mixto / frontera |

---

## 9. Síntesis para el equipo

La documentación verídica **no invalida** el plan de clusters; lo **confirma en el eje hipertensivo-metabólico-obstétrico** que ya habíamos diseñado. Lo más valioso de `documentacion-preeclampsia.md` es la **tabla de RR**, que permite justificar por qué C1, C2 y C4 existen y qué variables deben correlacionarse dentro de cada uno.

`bd_salud_pretanal.md` valida el **esquema de variables** y señala la **limitación estructural** de nuestro dataset actual: es un corte transversal, mientras el dominio real es mayormente temporal. Eso no impide clustering exploratorio, pero sí debe quedar explícito al interpretar resultados.

El cluster **C3** sigue siendo legítimo para minería de datos (separación clara en IMC y ganancia), pero debe tratarse como **eje nutricional independiente**, no como derivado de la documentación de preeclampsia.

**Próximo paso recomendado:** actualizar `plan-dataset-sintetico-embarazo.md` sección 4 con referencias cruzadas a estos documentos y las recomendaciones de prioridad alta, antes de regenerar el dataset en v2.

---

*Análisis elaborado a partir de la documentación en `clustering/documentacion/` y el reporte de calidad del dataset generado.*
