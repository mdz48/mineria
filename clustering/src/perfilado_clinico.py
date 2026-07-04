import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Cargar datos
df_full = pd.read_csv("data/v2/clustering_feature_view.csv")
df_features = df_full.drop(columns=["patient_id"])

num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols)
])

X_processed = preprocessor.fit_transform(df_features)

pca_optimo = PCA(n_components=12, random_state=42)
X_cluster = pca_optimo.fit_transform(X_processed)

# Utilizamos K-Means con la inicialización robusta que comprobamos
kmeans_final = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
clusters = kmeans_final.fit_predict(X_cluster)

# Etiquetamos el dataset original
df_perfilado = df_features.copy()
df_perfilado['Cluster'] = clusters

# Variables clave a analizar
variables_numericas = ['age_years', 'bmi_initial', 'mean_arterial_pressure', 'weight_gain', 'previous_pregnancies']
variables_booleanas = ['diabetes', 'chronic_hypertension', 'previous_preeclampsia', 'active_smoking', 'multiple_pregnancy']

# Agrupar y calcular
# Numéricas -> Promedio
# Booleanas -> Porcentaje de incidencia en ese cluster
agg_funcs = {var: 'mean' for var in variables_numericas}
agg_funcs.update({var: lambda x: (x == 'Yes').mean() * 100 if x.dtype == object else (x == 1).mean() * 100 for var in variables_booleanas})

resumen = df_perfilado.groupby('Cluster').agg(agg_funcs)

# Añadir conteo total de pacientes
conteo = df_perfilado.groupby('Cluster').size()
resumen.insert(0, 'N_Pacientes', conteo)

# Renombrar columnas para el print
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

print(resumen.round(1).to_markdown())
