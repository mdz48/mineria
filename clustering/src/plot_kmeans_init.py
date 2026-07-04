import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import matplotlib

matplotlib.use('Agg')

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

# Evaluar diferentes semillas e inicializaciones (n_init=1 para ver el efecto puro de la semilla)
inertias_kmpp = []
inertias_rand = []

best_inertia = float('inf')
best_labels = None

# Probaremos 50 semillas diferentes para ver qué tan estable es cada método
for seed in range(1, 51):
    # Método k-means++ (inteligente)
    km_pp = KMeans(n_clusters=4, init='k-means++', n_init=1, random_state=seed)
    km_pp.fit(X_cluster)
    inertias_kmpp.append(km_pp.inertia_)
    if km_pp.inertia_ < best_inertia:
        best_inertia = km_pp.inertia_
        best_labels = km_pp.labels_
        
    # Método Random (completamente al azar)
    km_r = KMeans(n_clusters=4, init='random', n_init=1, random_state=seed)
    km_r.fit(X_cluster)
    inertias_rand.append(km_r.inertia_)
    if km_r.inertia_ < best_inertia:
        best_inertia = km_r.inertia_
        best_labels = km_r.labels_

# 1. Graficar comparación de Inercias (Boxplot)
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot([inertias_kmpp, inertias_rand], tick_labels=['k-means++\n(Inteligente)', 'random\n(Al azar)'])
ax.set_ylabel('Inercia Final (Menos es mejor)', fontsize=12)
ax.set_title('Estabilidad de la Inicialización (50 corridas para K=4)', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6, axis='y')

output_dir = r"C:\Users\MAXIMILIANO\.gemini\antigravity\brain\4c77bb59-640a-4763-89f1-8d8a7a3f7184\scratch"
os.makedirs(output_dir, exist_ok=True)
plot_path_box = os.path.join(output_dir, "kmeans_init_boxplot.png")
plt.savefig(plot_path_box, dpi=150, bbox_inches='tight')

# 2. Graficar el resultado del Mejor Modelo Absoluto
plt.figure(figsize=(10, 8))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=best_labels, cmap='Set1', alpha=0.7, s=40, edgecolors='none')
plt.title(f"Mejor Agrupamiento Encontrado (K=4)\nInercia Mínima Absoluta: {best_inertia:.2f}", fontsize=14)
plt.xlabel("Componente Principal 1 (PC1)")
plt.ylabel("Componente Principal 2 (PC2)")
plt.grid(True, linestyle='--', alpha=0.6)

plot_path_scatter = os.path.join(output_dir, "kmeans_best_init_scatter.png")
plt.savefig(plot_path_scatter, dpi=150, bbox_inches='tight')
