import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
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

Z = sch.linkage(X_cluster, method='ward')

plt.figure(figsize=(12, 7))
plt.title('Dendrograma de Pacientes (4 Clústeres Forzados por Color)', fontsize=15)
plt.xlabel('Grupos de Pacientes', fontsize=12)
plt.ylabel('Distancia Euclidiana', fontsize=12)

# El default de scipy es 0.7 * max(Z[:,2]). Nosotros lo forzaremos a 60
dendrograma = sch.dendrogram(
    Z, 
    truncate_mode='level', 
    p=4, 
    leaf_rotation=45., 
    leaf_font_size=11.,
    show_contracted=True,
    color_threshold=60
)

plt.axhline(y=60, color='black', linestyle='--', label='Corte Exacto para 4 Clústeres (y=60)')

plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.legend()
plt.tight_layout()

output_dir = r"C:\Users\MAXIMILIANO\.gemini\antigravity\brain\4c77bb59-640a-4763-89f1-8d8a7a3f7184\scratch"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "dendrograma_4colores.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Dendrograma guardado en: {plot_path}")
