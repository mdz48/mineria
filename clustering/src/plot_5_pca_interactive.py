import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import os

# 1. Cargar y preprocesar los datos
df = pd.read_csv("data/v2/clustering_feature_view.csv")
if "patient_id" in df.columns:
    df = df.drop(columns=["patient_id"])

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

X_processed = preprocessor.fit_transform(df)

# 2. Correr PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_processed)

pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]
pc3 = X_pca[:, 2]
pc4 = X_pca[:, 3]
pc5 = X_pca[:, 4]

# Para WebGL en Plotly 3D, la opacidad individual puede no ser estable,
# así que mapearemos la "intensidad" (PC5) al TAMAÑO del punto, lo que
# da un efecto 3D excelente, ademas de indicarlo en el hover.
norm_pc5_size = MinMaxScaler(feature_range=(2, 12))
sizes = norm_pc5_size.fit_transform(pc5.reshape(-1, 1)).flatten()

# 3. Crear figura interactiva con Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=pc1,
    y=pc2,
    z=pc3,
    mode='markers',
    marker=dict(
        size=sizes,               # Intensidad / Tamaño basado en PC5
        color=pc4,                # Color basado en PC4
        colorscale='Viridis',     # Escala de color
        opacity=0.8,
        colorbar=dict(title="PC4 (Color)"),
        line=dict(width=0)
    ),
    text=[f"PC1: {x:.2f}<br>PC2: {y:.2f}<br>PC3: {z:.2f}<br>PC4 (Color): {c:.2f}<br>PC5 (Intensidad): {s:.2f}" 
          for x, y, z, c, s in zip(pc1, pc2, pc3, pc4, pc5)],
    hoverinfo='text'
)])

fig.update_layout(
    title="Proyección Interactiva PCA - 5 Componentes",
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

output_path = os.path.abspath("pca_interactivo.html")
fig.write_html(output_path)
print(f"Grafico interactivo guardado en: {output_path}")
