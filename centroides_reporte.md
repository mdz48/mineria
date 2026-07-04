# Reporte de Centroides Clínicos

A continuación se presentan los valores promedio (centroides) de las variables clínicas más importantes para cada algoritmo.

## 1. Centroides generados por KNN (Etiquetas de Producción)

Dado que KNN es un modelo de clasificación espacial, sus grupos imitan fielmente los centroides originales de K-Means. Estos son los promedios clínicos de cada perfil de riesgo detectado:

|   knn_cluster |   age_years |   bmi_initial |   systolic |   diastolic |   mean_arterial_pressure |   weight_kg |   previous_pregnancies |
|--------------:|------------:|--------------:|-----------:|------------:|-------------------------:|------------:|-----------------------:|
|             0 |       26.58 |         21.65 |     110.04 |       67.68 |                    81.63 |       61.99 |                   0.17 |
|             1 |       33.27 |         26.32 |     148.56 |       94.44 |                   111.57 |       72.77 |                   1.88 |
|             2 |       27.33 |         22.14 |     112.04 |       67.77 |                    81.93 |       64.27 |                   3.4  |
|             3 |       31.57 |         31.65 |     128.15 |       82.07 |                    97.29 |       92.1  |                   2.28 |

## 2. Centroides detectados por DBSCAN

DBSCAN agrupa por densidad. El clúster -1 representa ruido (casos atípicos que no entraron a ningún clúster, omitidos aquí). Estos son los promedios de los grupos densos encontrados:

|   dbscan_cluster |   age_years |   bmi_initial |   systolic |   diastolic |   mean_arterial_pressure |   weight_kg |   previous_pregnancies |
|-----------------:|------------:|--------------:|-----------:|------------:|-------------------------:|------------:|-----------------------:|
|                0 |       28.61 |         24.53 |     118.74 |       73.79 |                    88.44 |       70.38 |                   1.93 |
|                1 |       30.22 |         26.59 |     120.42 |       75.46 |                    91    |       77.58 |                   1.86 |
|                2 |       25.7  |         20.78 |     111.65 |       68.65 |                    82.18 |       59.19 |                   1.41 |
|                3 |       32.85 |         25.56 |     149.82 |       96.97 |                   113.84 |       69.84 |                   2.48 |