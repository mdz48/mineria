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
df_cluster['Cluster'] = kmeans_final.fit_predict(df_cluster[[f"PC{i+1}" for i in range(12)]])

df_perfilado = df_features.copy()
df_cluster['Cluster'] = 'Clúster ' + df_cluster['Cluster'].astype(str)

# Escalamos PC4 para el tamaño de las esferas (Plotly exige valores positivos para el size)
min_pc4 = df_cluster['PC4'].min()
df_cluster['PC4_Tamaño'] = df_cluster['PC4'] - min_pc4 + 1 

# Agregamos metadata para la inspección manual (Hover)
df_cluster['Edad'] = df_perfilado['age_years']
df_cluster['IMC'] = df_perfilado['bmi_initial'].round(2)
df_cluster['Presión Media'] = df_perfilado['mean_arterial_pressure'].round(2)
df_cluster['Fumadora'] = df_perfilado['active_smoking'].astype(str)

# Creación del gráfico 5D interactivo pero coloreado por Clúster
fig = px.scatter_3d(
    df_cluster, 
    x='PC1', 
    y='PC2', 
    z='PC3', 
    color='Cluster',
    size='PC4_Tamaño',
    hover_data=['Edad', 'IMC', 'Presión Media', 'Fumadora', 'PC4', 'PC5'],
    title="Mapa Multidimensional de Clústeres",
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.8
)

fig.update_layout(
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

output_path = os.path.abspath("mapa_clusteres_5d_interactivo.html")
fig.write_html(output_path)
print(f"HTML generado en: {output_path}")
