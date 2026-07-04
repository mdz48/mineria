import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os
import matplotlib

matplotlib.use('Agg')

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

# 2. Correr PCA extrayendo los primeros 5 componentes
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_processed)

pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]
pc3 = X_pca[:, 2]
pc4 = X_pca[:, 3]
pc5 = X_pca[:, 4]

# 3. Mapeos para Color (PC4) e Intensidad/Opacidad (PC5)
# Mapeo de PC4 a un mapa de colores (viridis)
norm_pc4 = plt.Normalize(pc4.min(), pc4.max())
cmap = plt.cm.viridis
colors = cmap(norm_pc4(pc4)) # Genera un array (N, 4) con RGBA

# Mapeo de PC5 a alpha (transparencia/intensidad) de 0.1 a 1.0
norm_pc5 = MinMaxScaler(feature_range=(0.1, 1.0))
alphas = norm_pc5.fit_transform(pc5.reshape(-1, 1)).flatten()

# Sobreescribimos el canal Alpha (el índice 3) de la matriz de colores
colors[:, 3] = alphas

# 4. Crear el gráfico 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Graficamos el scatter en 3D
# Pasamos la matriz 'colors' directamente a 'c'. Al ser una matriz RGBA, matplotlib la usa sin problema.
sc = ax.scatter(pc1, pc2, pc3, c=colors, s=40, marker='o', edgecolors='none')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Proyección de los 5 Primeros Componentes Principales\n(Color = PC4, Intensidad = PC5)')

# Agregar la barra de color (solo como referencia para PC4)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_pc4)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('PC4 (Color)')

# Guardar la imagen
output_dir = r"C:\Users\MAXIMILIANO\.gemini\antigravity\brain\4c77bb59-640a-4763-89f1-8d8a7a3f7184\scratch"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "pca_5_components.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print("Grafico guardado en:", plot_path)
