# Resultados del Análisis de Clustering (Versión 3)

## Objetivo
Incorporar antecedentes obstétricos más detallados al modelo de clustering para identificar perfiles clínicos más precisos. Las nuevas variables incluidas en el dataset `v3` son:
- `preterm_count`: Número de partos prematuros previos.
- `emergency_cesarean_count`: Número de cesáreas de emergencia previas.

Se aplicaron reglas estrictas durante la generación de datos para asegurar que las pacientes nulíparas (`nulliparous = 1`) tengan siempre 0 en estas variables de historial, dado que biológicamente es imposible tener embarazos previos si son primerizas.

## Resultados de Ejecución
El pipeline de preprocesamiento, PCA y K-Means (con K=4) se ejecutó satisfactoriamente sobre el dataset de 2000 pacientes y 30 variables.

**Análisis de Componentes Principales (PCA)**:
- Se retuvieron **14 componentes** principales para alcanzar un 80% de varianza explicada.

**Métricas de Calidad de Clustering**:
- Silhouette Score para K=4: **0.1423**. Aunque relativamente bajo, las agrupaciones siguen reflejando diferencias clínicas muy notorias.

## Perfil Clínico Obtenido (K=4)

| Cluster | Edad Media | BMI Medio | Presión Arterial | % Nulíparas | Embarazos Previos | Partos Prematuros | Cesáreas Emerg. | Riesgo (Preeclampsia/HTA/Diabetes) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | ~27 años | 22.1 (Normal) | 81.8 (Normal) | 0% (Multíparas) | ~3.4 | Casi nulo | Casi nulo | Sin comorbilidades. |
| **1** | ~33 años | 26.0 (Sobrepeso) | 113.0 (Alta) | 33% | ~2.0 | Alto (0.16 media) | Alto (0.15 media) | Muy alto riesgo (HTA 60%, Preeclampsia 15%). |
| **2** | ~31 años | 31.5 (Obesidad) | 97.4 (Alta) | 31% | ~2.0 | Alto (0.15 media) | Medio (0.11 media) | Alto riesgo metabólico (Diabetes 35%). |
| **3** | ~26 años | 21.6 (Normal) | 81.8 (Normal) | **83% (Casi todas nulíparas)** | ~0.17 | Nulo (0.0) | Nulo (0.0) | Sin comorbilidades. |

## Interpretaciones y Conclusiones Clínicas

1. **Separación de Multíparas Sanas vs Nulíparas Sanas**: Como discutimos anteriormente, matemáticamente se siguen separando. El **Cluster 3** agrupa a pacientes predominantemente sanas y jóvenes, en su gran mayoría primerizas (`nulliparous` = 0.829, historial adverso nulo). El **Cluster 0** agrupa a pacientes también jóvenes y sanas pero con un historial comprobado de múltiples embarazos sin complicaciones (0% nulíparas, ~3.4 embarazos previos). 
2. **Historial Adverso y Riesgo**: Los clusters de riesgo (1 y 2) ahora evidencian que las pacientes con condiciones adversas actuales (Hipertensión crónica, Diabetes) **también tienen un historial obstétrico más complicado**. Las tasas medias de partos prematuros (0.16) y cesáreas de emergencia (0.15) son visiblemente mayores en el Cluster 1 (pacientes con HTA/Preeclampsia) comparado con los clusters sanos.
3. **DBSCAN**: DBSCAN validó la presencia de densidades identificando también 4 agrupaciones principales, con unos 378 puntos considerados como ruido (pacientes atípicos o casos extremos combinados).

## Próximos Pasos
- Integrar este modelo en el backend de FastAPI (para la evaluación V3).
- Probar con otros algoritmos o realizar ajuste de hiperparámetros si se considera necesario refinar la clasificación.
