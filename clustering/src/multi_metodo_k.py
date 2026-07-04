import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df_full = pd.read_csv("data/v2/clustering_feature_view.csv")
df_features = df_full.drop(columns=["patient_id"])

num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols)
])

X_processed = preprocessor.fit_transform(df_features)

pca_12 = PCA(n_components=12, random_state=42)
X_pca = pca_12.fit_transform(X_processed)

print("=" * 70)
print("  ANALISIS MULTI-METODO PARA DETERMINAR K OPTIMO")
print("=" * 70)

# --- 1. Varianza explicada PCA ---
print("\n--- VARIANZA EXPLICADA POR PCA (12 componentes) ---")
var_acum = np.cumsum(pca_12.explained_variance_ratio_)
for i, v in enumerate(var_acum):
    print(f"  PC{i+1:2d}: {pca_12.explained_variance_ratio_[i]*100:5.1f}%  | Acumulada: {v*100:5.1f}%")

plt.figure(figsize=(10, 5))
plt.plot(range(1, 13), var_acum * 100, marker='o', linestyle='--', color='b')
plt.axhline(y=80, color='r', linestyle=':', label='80% varianza')
plt.xlabel('Numero de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulada (%)')
plt.title('Curva de Varianza Explicada por PCA')
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("notebooks/pca_varianza_acumulada.png", dpi=150)
print("\n  Grafica guardada: notebooks/pca_varianza_acumulada.png")

# --- 2. Metodo del Codo ---
print("\n--- METODO DEL CODO (Inercia) ---")
inercias = []
rango_k = range(2, 11)
for k in rango_k:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_pca)
    inercias.append(km.inertia_)
    print(f"  K={k}: Inercia = {km.inertia_:.1f}")

# --- 3. Silhouette, Calinski-Harabasz y Davies-Bouldin ---
print("\n--- SILHOUETTE | CALINSKI-HARABASZ | DAVIES-BOULDIN ---")
print(f"  {'K':>3} | {'Silhouette':>12} | {'Calinski-H':>12} | {'Davies-B':>12}")
print("  " + "-" * 55)

sil_scores = []
ch_scores = []
db_scores = []

for k in rango_k:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_pca)
    
    sil = silhouette_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    
    sil_scores.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)
    
    print(f"  {k:>3} | {sil:>12.4f} | {ch:>12.1f} | {db:>12.4f}")

# Grafica comparativa
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(rango_k, inercias, marker='o', linestyle='--', color='b')
axes[0, 0].axvline(x=4, color='r', linestyle=':', alpha=0.7, label='K=4')
axes[0, 0].set_title('Metodo del Codo (Inercia)')
axes[0, 0].set_xlabel('K')
axes[0, 0].set_ylabel('Inercia')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(rango_k, sil_scores, marker='s', linestyle='--', color='g')
axes[0, 1].axvline(x=4, color='r', linestyle=':', alpha=0.7, label='K=4')
axes[0, 1].set_title('Silhouette Score (mayor = mejor)')
axes[0, 1].set_xlabel('K')
axes[0, 1].set_ylabel('Silhouette')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(rango_k, ch_scores, marker='^', linestyle='--', color='purple')
axes[1, 0].axvline(x=4, color='r', linestyle=':', alpha=0.7, label='K=4')
axes[1, 0].set_title('Calinski-Harabasz (mayor = mejor)')
axes[1, 0].set_xlabel('K')
axes[1, 0].set_ylabel('Calinski-Harabasz')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(rango_k, db_scores, marker='D', linestyle='--', color='orange')
axes[1, 1].axvline(x=4, color='r', linestyle=':', alpha=0.7, label='K=4')
axes[1, 1].set_title('Davies-Bouldin (menor = mejor)')
axes[1, 1].set_xlabel('K')
axes[1, 1].set_ylabel('Davies-Bouldin')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.suptitle('Analisis Multi-Metodo para Determinar K Optimo', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("notebooks/multi_metodo_k_optimo.png", dpi=150)
print("\n  Grafica guardada: notebooks/multi_metodo_k_optimo.png")

# --- 4. DBSCAN ---
print("\n--- DBSCAN: BUSQUEDA DE CLUSTERS NATURALES ---")

# Primero: Grafica k-distance para encontrar eps optimo
nn = NearestNeighbors(n_neighbors=10)
nn.fit(X_pca)
distances, _ = nn.kneighbors(X_pca)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(10, 5))
plt.plot(distances)
plt.xlabel('Puntos ordenados')
plt.ylabel('Distancia al 10-vecino mas cercano')
plt.title('Grafica K-Distance para determinar eps de DBSCAN')
plt.grid(True)
plt.tight_layout()
plt.savefig("notebooks/dbscan_k_distance.png", dpi=150)
print("  Grafica k-distance guardada: notebooks/dbscan_k_distance.png")

# Probar varios eps
print("\n  Resultados DBSCAN con min_samples=10:")
print(f"  {'eps':>6} | {'Clusters':>8} | {'Ruido':>6} | {'% Ruido':>8}")
print("  " + "-" * 40)

for eps in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    dbscan = DBSCAN(eps=eps, min_samples=10)
    db_labels = dbscan.fit_predict(X_pca)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = (db_labels == -1).sum()
    pct_noise = n_noise / len(db_labels) * 100
    print(f"  {eps:>6.1f} | {n_clusters:>8} | {n_noise:>6} | {pct_noise:>7.1f}%")

print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)

best_sil_k = list(rango_k)[np.argmax(sil_scores)]
best_ch_k = list(rango_k)[np.argmax(ch_scores)]
best_db_k = list(rango_k)[np.argmin(db_scores)]

print(f"\n  Mejor K segun Silhouette:       K={best_sil_k} ({max(sil_scores):.4f})")
print(f"  Mejor K segun Calinski-Harabasz: K={best_ch_k} ({max(ch_scores):.1f})")
print(f"  Mejor K segun Davies-Bouldin:    K={best_db_k} ({min(db_scores):.4f})")
print(f"\n  K=4 Silhouette:       {sil_scores[2]:.4f}")
print(f"  K=4 Calinski-Harabasz: {ch_scores[2]:.1f}")
print(f"  K=4 Davies-Bouldin:    {db_scores[2]:.4f}")
