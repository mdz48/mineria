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

# 1. Cargar datos
df_full = pd.read_csv("data/v2/clustering_feature_view.csv")

# 2. Excluir el conteo exacto de embarazos
columnas_a_excluir = [
    "patient_id", 
    "previous_pregnancies", 
    "previous_deliveries", 
    "previous_miscarriages", 
    "previous_cesareans"
]
df_features_exp = df_full.drop(columns=columnas_a_excluir)

# 3. Preprocesamiento
num_cols_exp = df_features_exp.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_exp = df_features_exp.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor_exp = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_exp),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols_exp)
])

X_processed_exp = preprocessor_exp.fit_transform(df_features_exp)

# 4. PCA y Clustering (usando 12 componentes como en el experimento real)
pca_exp = PCA(n_components=12, random_state=42)
X_pca_exp = pca_exp.fit_transform(X_processed_exp)

kmeans_exp = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels_exp = kmeans_exp.fit_predict(X_pca_exp)

# 5. Crear DataFrame para el gráfico usando los 3 primeros componentes
df_plot = pd.DataFrame(X_pca_exp[:, :3], columns=['PC1', 'PC2', 'PC3'])
df_plot['Cluster_Exp'] = labels_exp.astype(str)
df_plot['Age'] = df_full['age_years']
df_plot['BMI'] = df_full['bmi_initial']
df_plot['Nulliparous'] = df_full['nulliparous'].map({1: 'Sí (Primigesta)', 0: 'No (Multípara)'})

# Crear figura 3D
fig = px.scatter_3d(
    df_plot, 
    x='PC1', 
    y='PC2', 
    z='PC3',
    color='Cluster_Exp',
    hover_data=['Age', 'BMI', 'Nulliparous'],
    title="Experimento K-Means: Colapso de Clústeres al omitir magnitud de embarazos (Solo True/False)",
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7
)

fig.update_traces(marker=dict(size=4))
fig.update_layout(scene=dict(xaxis_title='PC1 (Eje Metabólico)', yaxis_title='PC2 (Eje Obstétrico - Colapsado)', zaxis_title='PC3'))

# Guardar en HTML
output_path = os.path.abspath("notebooks/experimento_nulliparous_3d.html")
fig.write_html(output_path)
print(f"Gráfico 3D guardado exitosamente en:\\n{output_path}")
