import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Cargar datos
df_full = pd.read_csv("data/v2/clustering_feature_view.csv")

# 2. Excluir el conteo exacto de embarazos/partos/abortos y dejar SOLO 'nulliparous'
columnas_a_excluir = [
    "patient_id", 
    "previous_pregnancies", 
    "previous_deliveries", 
    "previous_miscarriages", 
    "previous_cesareans"
]
df_features = df_full.drop(columns=columnas_a_excluir)

# 3. Preprocesamiento
num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols)
])

X_processed = preprocessor.fit_transform(df_features)

# 4. PCA y Clustering
pca_12 = PCA(n_components=12, random_state=42)
X_pca_12 = pca_12.fit_transform(X_processed)

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X_pca_12)
score = silhouette_score(X_pca_12, labels)

# 5. Perfilado para ver si los grupos cambian
df_perfilado = df_full.copy()
df_perfilado['Cluster'] = labels

agrupado = df_perfilado.groupby('Cluster').agg({
    'age_years': ['count', 'mean'],
    'bmi_initial': 'mean',
    'mean_arterial_pressure': 'mean',
    'nulliparous': lambda x: (x == 1).mean() * 100, # % de primigestas
    'previous_pregnancies': 'mean', # Solo para observar, aunque el modelo no lo usó
    'diabetes': lambda x: (x == 1).mean() * 100,
    'chronic_hypertension': lambda x: (x == 1).mean() * 100,
})

agrupado.columns = ['N_Pacientes', 'Edad_Media', 'IMC_Medio', 'Presion_Media', '%_Primigestas', 'Embarazos_Previos_Reales', '%_Diabetes', '%_HTA']

print(f"--- EXPERIMENTO: USANDO SOLO NULLIPAROUS (TRUE/FALSE) ---")
print(f"Silhouette Score (Calidad Matemática): {score:.4f}")
print("\nPerfiles Clínicos Resultantes:")
print(agrupado.round(1).to_string())
