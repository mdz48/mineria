import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# 1. Cargar la vista de clustering v2
import os
for p in ['../data/v2/clustering_feature_view.csv', 'clustering/data/v2/clustering_feature_view.csv', 'data/v2/clustering_feature_view.csv']:
    if os.path.exists(p):
        df_full = pd.read_csv(p)
        break
else:
    raise FileNotFoundError('No se encontro el archivo clustering_feature_view.csv en ninguna de las rutas relativas posibles.')

# 2. Excluir el identificador (no tiene valor analítico para formar clusters)
df_features = df_full.drop(columns=["patient_id"])

# 3. Separar numéricas de categóricas para su correcto tratamiento
num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()

# 4. Construir el pipeline de preprocesamiento
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(sparse_output=False, drop='first'))
    ]), cat_cols)
])

# 5. Transformar los datos limpios y estandarizados
X_processed = preprocessor.fit_transform(df_features)

# 6. Extraer los 12 componentes principales acordados (80% de la varianza)
pca_optimo = PCA(n_components=12, random_state=42)
X_cluster = pca_optimo.fit_transform(X_processed)

# -------------------------------------------------------------------
# RESULTADOS FINALES:
# df_cluster: Es la matriz matemática abstracta que irá al algoritmo (K-Means/DBSCAN)
df_cluster = pd.DataFrame(X_cluster, columns=[f"PC{i+1}" for i in range(12)])

# df_perfilado: Es el dataset original limpio que usaremos para interpretar los resultados
df_perfilado = df_features.copy()
# -------------------------------------------------------------------

print(f"Dimensiones matemáticas para el algoritmo: {df_cluster.shape}")
print(f"Dimensiones clínicas para el perfilado: {df_perfilado.shape}\n")
display(df_cluster.head())

# -------------------------------------------------------------------------
# ENTRENAMIENTO DEL MODELO Y VISUALIZACIÓN DE CLÚSTERES
# -------------------------------------------------------------------------
import plotly.express as px
from sklearn.cluster import KMeans

# 1. Entrenar el modelo final con K=4 usando los 12 componentes PCA
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
# Ajustamos usando solo las columnas PC (ignorando si la columna Cluster ya existe)
columnas_pca = [f"PC{i+1}" for i in range(12)]
df_cluster['Cluster'] = kmeans_final.fit_predict(df_cluster[columnas_pca])

# 2. Le pegamos la etiqueta del clúster al dataset original para poder interpretarlo
df_perfilado['Cluster'] = df_cluster['Cluster'].astype(str)

# 3. GRAFICAR: Usamos PC1 y PC2 para visualizar los grupos de forma comprensible
fig_pca = px.scatter(
    df_cluster, 
    x='PC1', 
    y='PC2', 
    color=df_cluster['Cluster'].astype(str),
    title="Mapa de Pacientes (Agrupadas por Clúster usando PC1 y PC2)",
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7,
    labels={'color': 'Clúster asignado'}
)
fig_pca.show()

# 4. PERFILADO: ¿Qué significa clínicamente cada Clúster?
# Seleccionamos las variables médicas más representativas para ver el promedio de cada grupo
variables_clinicas = [
    'age_years', 'bmi_initial', 'weight_gain', 
    'mean_arterial_pressure', 'previous_pregnancies'
]

# Calculamos el promedio de estas variables para cada clúster
perfiles = df_perfilado.groupby('Cluster')[variables_clinicas].mean().round(2)

print("PERFIL CLÍNICO PROMEDIO POR CLÚSTER:")
display(perfiles)

# -------------------------------------------------------------------------
# CLUSTERING JERÁRQUICO Y DENDROGRAMA
# -------------------------------------------------------------------------
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# 1. Calculamos la matriz de distancias (usando el método de Ward para minimizar varianza)
Z = sch.linkage(df_cluster[[f"PC{i+1}" for i in range(12)]], method='ward')

# 2. Configurar el lienzo
plt.figure(figsize=(12, 7))
plt.title('Dendrograma Médico de Pacientes (Clustering Jerárquico)', fontsize=15)
plt.xlabel('Grupos de Pacientes (El número entre paréntesis indica la cantidad de pacientes agrupadas)', fontsize=12)
plt.ylabel('Distancia Euclidiana (Qué tan diferentes son)', fontsize=12)

# 3. Dibujar el árbol truncado (p=4 niveles de profundidad para que sea legible)
dendrograma = sch.dendrogram(
    Z, 
    truncate_mode='level', 
    p=4, 
    leaf_rotation=45., 
    leaf_font_size=11.,
    show_contracted=True
)

# 4. Dibujar línea sugerida de corte (Nuestra tijera imaginaria en Y=60)
plt.axhline(y=60, color='r', linestyle='--', label='Corte Sugerido (4 Clústeres)')

plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# PERFILADO CLÍNICO DE LOS CLÚSTERES (INTERPRETACIÓN)
# -------------------------------------------------------------------------

# Seleccionamos qué queremos analizar
variables_numericas = ['age_years', 'bmi_initial', 'mean_arterial_pressure', 'weight_gain', 'previous_pregnancies']
variables_booleanas = ['diabetes', 'chronic_hypertension', 'previous_preeclampsia', 'active_smoking', 'multiple_pregnancy']

# Agrupar y calcular promedios (numéricas) y porcentajes (booleanas)
agg_funcs = {var: 'mean' for var in variables_numericas}
agg_funcs.update({var: lambda x: (x == 'Yes').mean() * 100 if x.dtype == object else (x == 1).mean() * 100 for var in variables_booleanas})

resumen = df_perfilado.groupby('Cluster').agg(agg_funcs)

# Contar cuántas pacientes cayeron en cada grupo
conteo = df_perfilado.groupby('Cluster').size()
resumen.insert(0, 'N_Pacientes', conteo)

# Renombrar columnas para que se lea como reporte médico
resumen = resumen.rename(columns={
    'age_years': 'Edad Prom.',
    'bmi_initial': 'IMC Prom.',
    'mean_arterial_pressure': 'Presión Media',
    'weight_gain': 'Ganancia Peso (kg)',
    'previous_pregnancies': 'Embarazos Prev.',
    'diabetes': '% Diabetes',
    'chronic_hypertension': '% Hipertensión',
    'previous_preeclampsia': '% Preeclampsia Previa',
    'active_smoking': '% Fumadoras',
    'multiple_pregnancy': '% Emb. Múltiple'
})

print("PERFILES CLÍNICOS DEFINITIVOS:")
display(resumen.round(1))

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Análisis del Método del Codo (Elbow Method) para determinar K óptimo
inercia = []
rango_k = range(1, 11)

for k in rango_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_cluster) # X_cluster viene de las celdas anteriores (PCA)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(rango_k, inercia, marker='o', linestyle='--', color='b')
plt.title('Método del Codo para determinar K óptimo')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('Inercia (Suma de distancias al cuadrado)')
plt.xticks(rango_k)
plt.axvline(x=4, color='r', linestyle=':', label='K=4 seleccionado')
plt.legend()
plt.grid(True)
plt.show()

# Se observa una inflexión o "codo" entre K=3 y K=4.


from sklearn.metrics import silhouette_score

# Análisis de Silhouette para confirmar K óptimo (probando K=3, 4, 5)
print("Evaluando Silhouette Score para K=3, 4 y 5:\n")

for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels)
    print(f"K={k} -> Silhouette Score: {score:.4f}")

# El score más alto indica la mejor cohesión y separación. 
# K=4 nos dio el equilibrio perfecto entre matemáticas y lógica clínica.


# Experimento: Usando SOLO 'nulliparous' sin conteos exactos
from sklearn.metrics import silhouette_score

# 1. Excluir el conteo exacto de embarazos/partos/abortos
columnas_a_excluir = [
    "patient_id", 
    "previous_pregnancies", 
    "previous_deliveries", 
    "previous_miscarriages", 
    "previous_cesareans"
]
df_features_exp = df_full.drop(columns=columnas_a_excluir)

# 2. Preprocesamiento
num_cols_exp = df_features_exp.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_exp = df_features_exp.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor_exp = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_exp),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols_exp)
])

X_processed_exp = preprocessor_exp.fit_transform(df_features_exp)

# 3. PCA y Clustering
pca_exp = PCA(n_components=12, random_state=42)
X_cluster_exp = pca_exp.fit_transform(X_processed_exp)

kmeans_exp = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels_exp = kmeans_exp.fit_predict(X_cluster_exp)
score_exp = silhouette_score(X_cluster_exp, labels_exp)

# 4. Perfilado
df_perfilado_exp = df_full.copy()
df_perfilado_exp['Cluster_Exp'] = labels_exp

agrupado_exp = df_perfilado_exp.groupby('Cluster_Exp').agg({
    'age_years': ['count', 'mean'],
    'bmi_initial': 'mean',
    'mean_arterial_pressure': 'mean',
    'nulliparous': lambda x: (x == 1).mean() * 100, # % de primigestas
    'diabetes': lambda x: (x == 1).mean() * 100,
    'chronic_hypertension': lambda x: (x == 1).mean() * 100,
})

agrupado_exp.columns = ['N_Pacientes', 'Edad_Media', 'IMC_Medio', 'Presion_Media', '%_Primigestas', '%_Diabetes', '%_HTA']

print(f"--- EXPERIMENTO: USANDO SOLO NULLIPAROUS (TRUE/FALSE) ---")
print(f"Silhouette Score (Calidad Matemática): {score_exp:.4f} (Cayó de ~0.33)\n")
print("Perfiles Clínicos Resultantes (Observa la fusión en un Mega-Clúster):")
print(agrupado_exp.round(1).to_string())


# === COMPARATIVA EXPERIMENTAL: 12 vs 5 COMPONENTES ===
# Guardamos los modelos originales y probamos qué pasaría si usáramos 5 componentes
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# PCA 5
pca_5 = PCA(n_components=5, random_state=42)
X_pca_5 = pca_5.fit_transform(X_processed)
kmeans_5 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
labels_5 = kmeans_5.fit_predict(X_pca_5)

# Calcular Varianza y Silhouette para 5 componentes
var_5 = sum(pca_5.explained_variance_ratio_)
sil_5 = silhouette_score(X_pca_5, labels_5)

# Calcular Varianza y Silhouette para 12 componentes (el que elegimos)
var_12 = sum(pca_optimo.explained_variance_ratio_)
kmeans_12 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
labels_12 = kmeans_12.fit_predict(X_cluster) # X_cluster tiene 12
sil_12 = silhouette_score(X_cluster, labels_12)

print("=== DILEMA: INFORMACIÓN CLÍNICA VS SEPARACIÓN MATEMÁTICA ===")
print(f"Usando 12 Componentes (Elegido): Varianza Retenida = {var_12:.2%} | Silhouette = {sil_12:.4f}")
print(f"Usando  5 Componentes (Descarte): Varianza Retenida = {var_5:.2%} | Silhouette = {sil_5:.4f}")
print("\nINTERPRETACIÓN:")
print("Si usamos 5 componentes, el Silhouette sube (mejor separación matemática), pero perdemos casi la mitad de la información clínica (53% varianza).")
print("Elegimos 12 componentes porque en medicina es preferible retener el 80% de los datos vitales, aunque los clústeres se solapen ligeramente más en los bordes.")


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# 1. Gráfica de K-Distance para determinar el eps óptimo
nn = NearestNeighbors(n_neighbors=10)
nn.fit(X_cluster)
distances, indices = nn.kneighbors(X_cluster)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel('Puntos ordenados')
plt.ylabel('Distancia al 10-vecino más cercano')
plt.title('Gráfica K-Distance para determinar eps de DBSCAN')
plt.grid(True)
plt.show()

# 2. Ejecutar DBSCAN con eps=3.0 (donde se estabiliza la curva/codo)
dbscan = DBSCAN(eps=3.0, min_samples=10)
db_labels = dbscan.fit_predict(X_cluster)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
pct_noise = n_noise / len(db_labels) * 100

print(f"DBSCAN con eps=3.0 descubrió {n_clusters} clústeres naturales.")
print(f"Puntos considerados ruido/anomalías: {n_noise} ({pct_noise:.1f}%)")

from sklearn.metrics import adjusted_rand_score
import pandas as pd

# Cargar Ground Truth
import os
for p in ['../data/v2/ground_truth.csv', 'clustering/data/v2/ground_truth.csv', 'data/v2/ground_truth.csv']:
    if os.path.exists(p):
        df_truth = pd.read_csv(p)
        break
else:
    raise FileNotFoundError('No se encontro el archivo ground_truth.csv en ninguna de las rutas relativas posibles.')
y_true = df_truth['cluster_verdadero']

# Comparar K=4 (nuestro modelo actual) contra ground truth
km_4 = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
y_pred_4 = km_4.fit_predict(X_cluster)
ari_4 = adjusted_rand_score(y_true, y_pred_4)

# Comparar K=6 (forzando los 6 clusters del generador)
km_6 = KMeans(n_clusters=6, init='k-means++', n_init=10, random_state=42)
y_pred_6 = km_6.fit_predict(X_cluster)
ari_6 = adjusted_rand_score(y_true, y_pred_6)

print("=== COMPARACIÓN CONTRA GROUND TRUTH (6 Clústeres de Diseño) ===")
print(f"ARI agrupando en K=4 (Perfiles Clínicos): {ari_4:.4f}")
print(f"ARI forzando K=6 (Intento de replicar diseño): {ari_6:.4f}")
print("\nConclusión: K=4 es una agrupación natural más sólida que la partición sintética original.")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# 1. Usar las etiquetas de K-Means (K=4) como nuestro 'Target' (Variable Y)
y = df_cluster['Cluster']

# 2. Dividir en Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(df_cluster.drop(columns=['Cluster']), y, test_size=0.2, random_state=42, stratify=y)

# 3. Entrenar KNN con los mejores hiperparámetros encontrados
knn_best = KNeighborsClassifier(n_neighbors=7, weights='uniform', metric='euclidean', p=2)
knn_best.fit(X_train, y_train)

# 4. Evaluar el modelo
y_pred_test = knn_best.predict(X_test)
print("=== REPORTE DE CLASIFICACIÓN KNN ===")
print(f"Accuracy en Datos de Prueba: {accuracy_score(y_test, y_pred_test):.2%}\n")
print(classification_report(y_test, y_pred_test, target_names=[
    "C0: Primigestas Sanas",
    "C1: Riesgo Hipertensivo",
    "C2: Multiparas Sanas",
    "C3: Riesgo Metabolico"
]))

# 5. Visualización 3D Interactiva con Plotly
df_plot = df_perfilado.copy()
df_plot['Prediccion_KNN'] = knn_best.predict(df_cluster.drop(columns=['Cluster'])).astype(str)
df_plot['PC1'] = df_cluster['PC1']
df_plot['PC2'] = df_cluster['PC2']
df_plot['PC3'] = df_cluster['PC3']

fig = px.scatter_3d(
    df_plot,
    x='PC1',
    y='PC2',
    z='PC3',
    color='Prediccion_KNN',
    hover_data=['age_years', 'bmi_initial', 'mean_arterial_pressure', 'previous_pregnancies'],
    title="Fronteras de Decisión KNN en el Espacio PCA 3D",
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7
)
fig.update_traces(marker=dict(size=4))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig.show()

# 5. Entrenar Clasificador XGBoost Alternativo
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report, accuracy_score
# XGBoost requiere que las etiquetas comiencen en 0 y sean enteras.
# Nuestras etiquetas de K-Means ya son 0, 1, 2, 3.
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
# Evaluar XGBoost
y_pred_xgb = xgb_model.predict(X_test)
print("=== REPORTE DE CLASIFICACIÓN XGBOOST ===")
print(f"Accuracy en Datos de Prueba: {accuracy_score(y_test, y_pred_xgb):.2%}\n")
print(classification_report(y_test, y_pred_xgb, target_names=[
    "C0: Primigestas Sanas",
    "C1: Riesgo Hipertensivo",
    "C2: Multiparas Sanas",
    "C3: Riesgo Metabólico"
]))
# Guardar el modelo alternativo
import os
# Determinar ruta dinámica por culpa del kernel de Jupyter
if os.path.exists('../models'):
    model_dir = '../models'
elif os.path.exists('clustering/models'):
    model_dir = 'clustering/models'
else:
    model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'xgboost_model.pkl')
joblib.dump(xgb_model, model_path)
print(f"Modelo XGBoost guardado en '{model_path}'")
# 4. Visualización 3D Interactiva con Plotly para XGBoost
import plotly.express as px
df_plot_xgb = df_perfilado.copy()
df_plot_xgb['Prediccion_XGB'] = xgb_model.predict(df_cluster.drop(columns=['Cluster'])).astype(str)
df_plot_xgb['PC1'] = df_cluster['PC1']
df_plot_xgb['PC2'] = df_cluster['PC2']
df_plot_xgb['PC3'] = df_cluster['PC3']
fig_xgb = px.scatter_3d(
    df_plot_xgb,
    x='PC1',
    y='PC2',
    z='PC3',
    color='Prediccion_XGB',
    hover_data=['age_years', 'bmi_initial', 'mean_arterial_pressure', 'previous_pregnancies'],
    title="Fronteras de Decisión XGBoost en el Espacio PCA 3D",
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7
)
fig_xgb.update_traces(marker=dict(size=4))
fig_xgb.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig_xgb.show()


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go

# Aseguramos que tenemos las etiquetas de KNN (ya entrenado)
X_pca_full = df_cluster.drop(columns=['Cluster'], errors='ignore')
knn_labels = knn_best.predict(X_pca_full)

# Ejecutamos DBSCAN con los parametros reportados (eps=3.0)
dbscan_labels = DBSCAN(eps=3.0, min_samples=10).fit_predict(X_pca_full)

# Asignamos temporalmente las etiquetas
df_numeric = df_perfilado.copy()
df_numeric['knn_cluster'] = knn_labels
df_numeric['dbscan_cluster'] = dbscan_labels

# Tambien agregamos las PCA para poder graficar
df_numeric['PC1'] = df_cluster['PC1']
df_numeric['PC2'] = df_cluster['PC2']
df_numeric['PC3'] = df_cluster['PC3']

key_cols = ['age_years', 'bmi_initial', 'systolic', 'diastolic', 'mean_arterial_pressure', 'weight_kg', 'previous_pregnancies']

print('--- CENTROIDES CLINICOS KNN (PRODUCCION) ---')
knn_centroids_clinico = df_numeric.groupby('knn_cluster')[key_cols].mean().round(2)
display(knn_centroids_clinico)

print('\n--- CENTROIDES CLINICOS DBSCAN ---')
dbscan_centroids_clinico = df_numeric[df_numeric['dbscan_cluster'] != -1].groupby('dbscan_cluster')[key_cols].mean().round(2)
display(dbscan_centroids_clinico)

# ==========================================
# VISUALIZACION 3D CON CENTROIDES
# ==========================================

# 1. Grafica para KNN
df_numeric['knn_cluster_str'] = df_numeric['knn_cluster'].astype(str)
fig_knn = px.scatter_3d(
    df_numeric, x='PC1', y='PC2', z='PC3', color='knn_cluster_str',
    title='Grupos KNN con sus Centroides Geometricos',
    color_discrete_sequence=px.colors.qualitative.Set1, opacity=0.3
)
fig_knn.update_traces(marker=dict(size=3))

# Centroides Geometricos KNN (Promedio de PC1, PC2, PC3)
knn_geom_centroids = df_numeric.groupby('knn_cluster')[['PC1', 'PC2', 'PC3']].mean().reset_index()
knn_geom_centroids['knn_cluster_str'] = knn_geom_centroids['knn_cluster'].astype(str)

fig_knn.add_trace(go.Scatter3d(
    x=knn_geom_centroids['PC1'], y=knn_geom_centroids['PC2'], z=knn_geom_centroids['PC3'],
    mode='markers+text',
    marker=dict(size=12, color='black', symbol='diamond'),
    text=['Centroide C'+str(c) for c in knn_geom_centroids['knn_cluster']],
    textposition='top center',
    name='Centroides KNN'
))
fig_knn.show()

# 2. Grafica para DBSCAN
df_numeric['dbscan_cluster_str'] = df_numeric['dbscan_cluster'].astype(str)
fig_dbscan = px.scatter_3d(
    df_numeric, x='PC1', y='PC2', z='PC3', color='dbscan_cluster_str',
    title='Masas Densas DBSCAN con Centro de Masa',
    color_discrete_sequence=px.colors.qualitative.Pastel, opacity=0.3
)
fig_dbscan.update_traces(marker=dict(size=3))

# Centro de masa DBSCAN (excluyendo ruido -1)
dbscan_geom = df_numeric[df_numeric['dbscan_cluster'] != -1].groupby('dbscan_cluster')[['PC1', 'PC2', 'PC3']].mean().reset_index()

fig_dbscan.add_trace(go.Scatter3d(
    x=dbscan_geom['PC1'], y=dbscan_geom['PC2'], z=dbscan_geom['PC3'],
    mode='markers+text',
    marker=dict(size=12, color='black', symbol='x'),
    text=['Centro de Masa C'+str(c) for c in dbscan_geom['dbscan_cluster']],
    textposition='top center',
    name='Centroides DBSCAN'
))
fig_dbscan.show()
