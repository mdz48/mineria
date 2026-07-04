import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_cluster, kmeans.labels_))

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Número de Clústeres (K)', fontsize=12)
ax1.set_ylabel('Inercia (Método del Codo)', color=color, fontsize=12)
ax1.plot(K_range, inertias, marker='o', color=color, linewidth=2, label="Inercia")
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xticks(K_range)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Coeficiente de Silhouette', color=color, fontsize=12)  
ax2.plot(K_range, silhouettes, marker='s', color=color, linewidth=2, label="Silhouette")
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Evaluación de K óptimo: Inercia vs Silhouette', fontsize=14)
fig.tight_layout()

output_dir = r"C:\Users\MAXIMILIANO\.gemini\antigravity\brain\4c77bb59-640a-4763-89f1-8d8a7a3f7184\scratch"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "kmeans_evaluation.png")
plt.savefig(plot_path, dpi=150)
print(f"Grafico guardado en: {plot_path}")
print("Silhouettes:", [f"K={k}: {s:.4f}" for k, s in zip(K_range, silhouettes)])
