import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

print("Cargando datos originales...")
df_full = pd.read_csv("data/v2/clustering_feature_view.csv")
df_features = df_full.drop(columns=["patient_id"])

num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols)
])

X_processed = preprocessor.fit_transform(df_features)

print("Calculando PCA...")
pca_optimo = PCA(n_components=12, random_state=42)
X_cluster = pca_optimo.fit_transform(X_processed)

print("Asignando etiquetas maestras (Ground Truth) usando K-Means (K=4)...")
kmeans_final = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
clusters = kmeans_final.fit_predict(X_cluster)

# Crear dataset listo para Machine Learning (KNN)
df_labeled = pd.DataFrame(X_cluster, columns=[f"PC{i+1}" for i in range(12)])
df_labeled.insert(0, 'patient_id', df_full['patient_id'])
df_labeled['Cluster_Label'] = clusters

output_path = "data/v2/labeled_patients.csv"
df_labeled.to_csv(output_path, index=False)
print(f"Dataset etiquetado guardado exitosamente en: {output_path}")
