import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
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

pca_optimo = PCA(n_components=12, random_state=42)
X_cluster = pca_optimo.fit_transform(X_processed)
df_cluster = pd.DataFrame(X_cluster, columns=[f"PC{i+1}" for i in range(12)])

kmeans_final = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
df_cluster['Cluster_ID'] = kmeans_final.fit_predict(df_cluster[[f"PC{i+1}" for i in range(12)]])
df_cluster['Cluster'] = 'Clúster ' + df_cluster['Cluster_ID'].astype(str)

df_perfilado = df_features.copy()
min_pc4 = df_cluster['PC4'].min()
df_cluster['PC4_Tamaño'] = df_cluster['PC4'] - min_pc4 + 1 
df_cluster['Edad'] = df_perfilado['age_years']
df_cluster['IMC'] = df_perfilado['bmi_initial'].round(2)
df_cluster['Presión Media'] = df_perfilado['mean_arterial_pressure'].round(2)

# 2. Base scatter 3D
fig = px.scatter_3d(
    df_cluster, 
    x='PC1', 
    y='PC2', 
    z='PC3', 
    color='Cluster',
    size='PC4_Tamaño',
    hover_data=['Edad', 'IMC', 'Presión Media'],
    title="Mapa de Clústeres con Volúmenes (Convex Hulls)",
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.6
)

# 3. Agregar Convex Hulls (Volúmenes 3D) para agrupar visualmente cada clúster
colores = px.colors.qualitative.Set1
for i, cluster_id in enumerate(sorted(df_cluster['Cluster'].unique())):
    subset = df_cluster[df_cluster['Cluster'] == cluster_id]
    
    # Agregar malla 3D (Envolvente convexa)
    fig.add_trace(go.Mesh3d(
        x=subset['PC1'],
        y=subset['PC2'],
        z=subset['PC3'],
        alphahull=0, # 0 indica convex hull (envolvente que contiene todos los puntos)
        opacity=0.15, # Muy transparente para no tapar los puntos
        color=colores[i % len(colores)],
        name=f"Volumen {cluster_id}"
    ))

fig.update_layout(
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    margin=dict(l=0, r=0, b=0, t=40)
)

output_path = os.path.abspath("mapa_clusteres_volumen_3d.html")
fig.write_html(output_path)
print(f"HTML generado en: {output_path}")
