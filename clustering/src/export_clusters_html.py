import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import os

# 1. Cargar datos y preprocesar
df_full = pd.read_csv("data/v2/clustering_feature_view.csv")
df_features = df_full.drop(columns=["patient_id"])

num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols)
])

X_processed = preprocessor.fit_transform(df_features)

# 2. PCA y Clustering
pca_optimo = PCA(n_components=12, random_state=42)
X_cluster = pca_optimo.fit_transform(X_processed)
df_cluster = pd.DataFrame(X_cluster, columns=[f"PC{i+1}" for i in range(12)])

kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans_final.fit_predict(df_cluster[[f"PC{i+1}" for i in range(12)]])

# 3. Preparar datos para el Hover (para que al pasar el mouse veas las variables reales)
df_perfilado = df_features.copy()
df_cluster['Cluster'] = 'Clúster ' + df_cluster['Cluster'].astype(str)

df_cluster['Edad'] = df_perfilado['age_years']
df_cluster['IMC'] = df_perfilado['bmi_initial'].round(2)
df_cluster['Presión Media'] = df_perfilado['mean_arterial_pressure'].round(2)
df_cluster['Gestaciones Previas'] = df_perfilado['previous_pregnancies']

# 4. Generar gráfico Plotly
fig_pca = px.scatter(
    df_cluster, 
    x='PC1', 
    y='PC2', 
    color='Cluster',
    title="Mapa Bidimensional de Pacientes (K-Means, K=4)",
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7,
    hover_data=['Edad', 'IMC', 'Presión Media', 'Gestaciones Previas'],
    labels={'color': 'Asignación'}
)

# 5. Exportar a HTML
output_path = os.path.abspath("mapa_clusteres_kmeans.html")
fig_pca.write_html(output_path)
print(f"HTML generado en: {output_path}")
