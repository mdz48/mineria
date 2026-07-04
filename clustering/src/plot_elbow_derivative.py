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

# 2. Calcular Inercias desde K=1 hasta K=11 para tener la derivada desde K=2 hasta K=11
K_range = range(1, 12)
inertias = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)

# 3. Calcular la Derivada (Primera diferencia finita) en valor absoluto
# np.diff hace Inercia(K+1) - Inercia(K). Al tomar el valor absoluto, 
# tenemos la magnitud de cuánto bajó la inercia al agregar un clúster más.
derivada_abs = np.abs(np.diff(inertias))
K_diffs = range(2, 12)

# 4. Graficar
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_diffs, derivada_abs, marker='o', color='purple', linewidth=2, markersize=8)
ax.set_xlabel('Número de Clústeres (K)', fontsize=12)
ax.set_ylabel('| $\Delta$ Inercia | (Magnitud de la Derivada)', fontsize=12)
ax.set_title('Derivada del Método del Codo (Valor Absoluto)', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xticks(K_diffs)

# Marcar K=4
ax.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='K=4')
ax.legend()

output_dir = r"C:\Users\MAXIMILIANO\.gemini\antigravity\brain\4c77bb59-640a-4763-89f1-8d8a7a3f7184\scratch"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "elbow_derivative.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Grafico guardado en: {plot_path}")
