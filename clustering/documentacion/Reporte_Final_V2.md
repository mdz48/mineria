# Reporte Final de Minería de Datos: Clustering V2 (Salud Prenatal)

## 1. Introducción y Objetivo
El objetivo de esta fase de minería de datos fue segmentar a una población de pacientes embarazadas utilizando técnicas de aprendizaje no supervisado (Clustering). La meta clínica era identificar perfiles de riesgo subyacentes sin depender de un diagnóstico pre-etiquetado, permitiendo a la aplicación de Salud Prenatal personalizar alertas, seguimientos y recomendaciones basadas en el grupo de riesgo al que pertenece cada nueva paciente de forma automática.

## 2. Diseño y Generación del Dataset Sintético (Ingeniería Clínica)
Dado que los algoritmos de clustering son tan buenos como la calidad de los datos que ingieren, diseñamos un pipeline riguroso para generar datos de embarazadas que fueran realistas a nivel fisiológico y estadístico.

### 2.1 Los Perfiles Teóricos Inyectados ("Clusters Pensados")
Originalmente, programamos el generador para inyectar a la población con **6 perfiles clínicos teóricos**, distribuidos de manera hiperrealista:
- **Perfil Control (45%)**: Pacientes sin riesgo, presión arterial normal y BMI estándar.
- **Perfil Cardiometabólico (18%)**: Riesgo dominado por obesidad clínica (BMI > 30) y diabetes, con alta ganancia de peso.
- **Perfil Hipertensivo (12%)**: Riesgo focalizado en presión arterial elevada (Sistólica > 140 / Diastólica > 90) a partir de la semana 20 de gestación, además de historial previo de preeclampsia.
- **Déficit Antropométrico (13%)**: Pacientes con bajo peso inicial (BMI < 18.5) y ganancia nutricional deficiente.
- **Alto Riesgo Obstétrico (7%)**: Edad avanzada, enfermedad renal crónica, embarazos múltiples y cruce de morbilidades.
- **Perfil Mixto / Frontera (5%)**: Casos de solapamiento entre perfiles.

*Nota interesante*: Aunque inyectamos 6 grupos con ruido y solapamiento intencional, el algoritmo de Machine Learning (K-Means) demostró posteriormente que matemáticamente la geometría de estos datos se condensa de manera natural en **4 grupos dominantes**.

### 2.2 Reglas Duras (Hard Rules Fisiológicas)
Para garantizar la coherencia médica y evitar que el Machine Learning aprendiera "basura", el generador fue forzado a respetar estrictas leyes físicas y biológicas:
- **Reglas de Paridad**: Es matemáticamente imposible tener más partos que embarazos (`delivery_count <= parity_count`), o más cesáreas que partos (`cesarean_count <= delivery_count`). Las pacientes primerizas (`nulliparous`) fueron forzadas estrictamente a cero en todos sus historiales.
- **Reglas Hemodinámicas**: La presión sistólica siempre debe ser mayor a la diastólica por un margen fisiológico (`systolic > diastolic + 15`).
- **Reglas Metabólicas y Temporales**: El trimestre gestacional debe estar matemáticamente alineado con las semanas de gestación.

### 2.3 Inyección de Caos (Outliers y Missingness)
Para preparar a los algoritmos para la cruda realidad del mundo clínico, contaminamos el dataset intencionalmente:
- **Outliers**: Inyectamos errores de medición (presiones imposibles de 210 mmHg) y errores de tipografía humana (tallas de 1700 cm).
- **Valores Nulos Controlados**: Simulamos omisiones aleatorias (Missing Completely at Random), sesgos lógicos donde a veces no se toma la presión (Missing at Random), y el clásico sesgo psicológico donde pacientes con peso extremo deciden omitir subirse a la báscula (Missing Not at Random).

## 3. Análisis Exploratorio de Datos (EDA)
Previo a la fase de Machine Learning, se ejecutó un escrutinio profundo del dataset simulado mediante dos notebooks especializados:

1. **EDA Tabular (`eda_tablas_v2.ipynb`)**: Nos permitió validar matemáticamente las dimensiones de la matriz, la tasa exacta de valores faltantes (*Missingness* inyectado), distribuciones estadísticas (percentiles) y, fundamentalmente, **comprobar que las reglas duras de coherencia clínica se cumplían al 100%**.
2. **EDA Gráfico (`eda_graficos_v2.ipynb`)**: A través de la visualización de distribuciones, diagramas de caja (boxplots), matrices de correlación de Pearson y gráficos de dispersión multidimensionales (pairplots), pudimos corroborar visualmente que las correlaciones clínicas entre variables (por ejemplo, el peso y la presión arterial) existían de forma natural y no eran aleatorias.

## 4. Preprocesamiento de Datos (Pipeline ML)
- **Integración**: Todos los datos generados se consolidaron directamente en una única matriz analítica plana (`clustering_feature_view.csv`).
- **Preprocesamiento Pipeline**: 
  - *Imputación*: Estrategia de mediana para variables numéricas, moda para categóricas.
  - *Codificación*: One-Hot Encoding para permitir que el algoritmo asimilara variables categóricas.
  - *Escalado*: StandardScaler para homogeneizar el peso de todas las variables, evitando que campos con magnitudes altas opacaran a los demás.

## 5. Reducción de Dimensionalidad (PCA)
Para evitar la "maldición de la dimensionalidad" y eliminar el ruido provocado por variables altamente correlacionadas, aplicamos el Análisis de Componentes Principales (PCA). Durante esta fase nos enfrentamos a un clásico dilema de *Bias-Variance* (Información vs Separación), evaluando dos escenarios:

1. **Escenario A (5 Componentes)**: Al comprimir el dataset a solo 5 dimensiones, el índice matemático de separación (Silhouette Score) subía significativamente a **0.2582**, mostrando clústeres más definidos geométricamente. Sin embargo, esto provocaba una pérdida masiva de información clínica, reteniendo apenas el **53.6% de la varianza original**.
2. **Escenario B (12 Componentes - Elegido)**: Al utilizar 12 dimensiones, el Silhouette Score descendió a **0.1628** (mayor solapamiento geométrico en los bordes de los grupos), pero logramos retener un masivo **80.39% de la información pura (varianza)**.

**Interpretación Médica vs Matemática**:
Decidimos sacrificar la "estética matemática" (un Silhouette perfecto de 5 componentes) a favor de la **riqueza clínica**. En el contexto médico, perder casi la mitad de los signos vitales y antecedentes de la paciente (como hubiera ocurrido usando 5 componentes) es inaceptable, ya que esos pequeños detalles son los que marcan la diferencia entre un perfil de hipertensión crónica y una preeclampsia incipiente. Por tanto, **12 Componentes Principales** fue la decisión técnica definitiva.

![Varianza Acumulada](/c:/Universidad/mineria/clustering/notebooks/pca_varianza_acumulada.png)

## 6. Determinación del Algoritmo y Número de Clústeres (K)
Se evaluaron múltiples algoritmos (DBSCAN y K-Means) y métricas matemáticas para determinar el número óptimo de divisiones clínicas:

1. **DBSCAN**: Se utilizó un algoritmo espacial basado en densidad (DBSCAN) sin especificarle el número de grupos por adelantado. Con un `eps=3.0`, DBSCAN descubrió de forma natural la existencia de **4 clústeres**.
2. **K-Means (Seleccionado para Producción)**: Al converger con los resultados de DBSCAN, se utilizó K-Means configurado en **K=4**. Esta decisión fue doblemente respaldada por el **Método del Codo (Elbow Method)** y el **Silhouette Score**, cuya evidencia matemática apuntó sólidamente a que 4 grupos era la división más óptima y balanceada.
3. **Clustering Jerárquico (Dendrograma)**: Para tener una confirmación visual de la topología de los datos, construimos un Dendrograma utilizando el método de Ward (que minimiza la varianza interna). Al analizar las distancias verticales más largas en el árbol jerárquico antes de una fusión, el gráfico demostró una división natural y clara en **4 ramas principales** (colores), proporcionando la tercera prueba definitiva para nuestra elección.

![Métodos K Óptimo](/c:/Universidad/mineria/clustering/notebooks/multi_metodo_k_optimo.png)
![DBSCAN K-Distance](/c:/Universidad/mineria/clustering/notebooks/dbscan_k_distance.png)
![Dendrograma](file:///C:/Users/MAXIMILIANO/.gemini/antigravity/brain/4c77bb59-640a-4763-89f1-8d8a7a3f7184/scratch/dendrograma_4colores.png)

## 7. Interpretación de los Perfiles Clínicos (Resultados K=4)
Al aplicar K-Means para forzar la separación en 4 grupos, se revelaron perfiles de riesgo que hacían total sentido médico y clínico:

- **Clúster 0 (Sanas Multíparas - ~37%)**: Pacientes con presión arterial (MAP), peso (BMI) y signos vitales dentro de rangos normales. Destacan por tener un historial comprobado de múltiples embarazos previos. Su riesgo general es bajo.
- **Clúster 1 (Riesgo Hipertensivo - ~11%)**: Pacientes caracterizadas por tasas alarmantes de Hipertensión Crónica y un historial fuerte de episodios de Preeclampsia. Presentan una presión arterial media muy superior al resto. Tienen el riesgo obstétrico más agudo.
- **Clúster 2 (Riesgo Metabólico - ~18%)**: Pacientes agrupadas por padecer Obesidad clínica (BMI > 30) y una elevadísima prevalencia de Diabetes Gestacional o Pre-gestacional. Su foco de riesgo gira en torno a la macrosomía y el metabolismo.
- **Clúster 3 (Sanas Primerizas - ~34%)**: Pacientes predominantemente jóvenes, completamente saludables y sin enfermedades previas, pero unidas por una característica en común: **son nulíparas (es su primer embarazo)**. Clínicamente requieren un protocolo distinto (seguimiento preventivo de rutina) al carecer de historial de partos.

## 8. Clasificación Lista para Producción (KNN vs XGBoost)
El flujo de clustering no supervisado es computacionalmente pesado para ejecutarse en tiempo real. Para que el Backend (FastAPI) pueda asignar rápidamente a una nueva usuaria a uno de estos 4 grupos, entrenamos y comparamos dos modelos supervisados utilizando como variable objetivo las etiquetas descubiertas por K-Means:

1. **K-Nearest Neighbors (KNN)**: Alcanzó un **accuracy casi perfecto del 99.8%**. Al ser un algoritmo basado en distancias espaciales (igual que K-Means), asimiló de manera natural los umbrales multidimensionales de los clústeres.
2. **XGBoost (Extreme Gradient Boosting)**: Evaluamos este potente modelo basado en árboles de decisión. XGBoost alcanzó un altísimo **accuracy del 98.25%**. 

**Interpretación**: Aunque XGBoost es considerado un modelo "State-of-the-Art" para datos tabulares, en este caso particular KNN fue ligeramente superior. Esto hace sentido matemático absoluto: los clústeres de K-Means son, por definición, regiones delimitadas por distancias euclidianas a un centroide. Un algoritmo espacial como KNN captura esa geometría espacial mucho mejor que los cortes ortogonales de los árboles de XGBoost. 

Ambos modelos son excepcionales, pero **recomendamos KNN** por su 99.8% de precisión en este espacio vectorial. Todos los artefactos están disponibles en `/models/`:
  - `preprocessor.pkl`
  - `knn_model.pkl`
  - `xgboost_model.pkl` (Alternativa)
  - `feature_columns.pkl`

## 9. Conclusión
La **Versión 2** del ecosistema de clustering logró capturar y separar con éxito la estructura clínica subyacente de la población obstétrica. Tras un experimento exploratorio (V3), se comprobó matemáticamente que añadir un exceso de variables detalladas pero infrecuentes no lograba sobreponerse al peso dominante de signos vitales como el BMI y la Presión Arterial, reafirmando que **la V2 es matemáticamente estable, explicable para el cuerpo médico y lo suficientemente liviana para operar en producción.**
