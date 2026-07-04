import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import joblib
import os

print("Entrenando el Pipeline Final Definitivo...")
df_full = pd.read_csv("data/v2/clustering_feature_view.csv")
df_features = df_full.drop(columns=["patient_id"])

# 1. Ajustar Preprocesador
num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols)
])

X_processed = preprocessor.fit_transform(df_features)

# 2. Ajustar PCA (5 componentes)
pca_5 = PCA(n_components=5, random_state=42)
X_pca_5 = pca_5.fit_transform(X_processed)

# El Ground Truth maestro lo da el K-Means de 12 componentes
pca_12 = PCA(n_components=12, random_state=42)
X_pca_12 = pca_12.fit_transform(X_processed)
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
y_labels = kmeans.fit_predict(X_pca_12)

# 3. Entrenar KNN Definitivo (5 componentes, K=15, Manhattan) -> Exactitud 99%
knn = KNeighborsClassifier(n_neighbors=15, metric='manhattan')
knn.fit(X_pca_5, y_labels)

# 4. Guardar en disco (Exportación)
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
joblib.dump(pca_5, os.path.join(output_dir, "pca_5.pkl"))
joblib.dump(knn, os.path.join(output_dir, "knn_model.pkl"))

# Para mantener la compatibilidad con el pipeline, guardaremos el orden de las columnas originales
joblib.dump(df_features.columns.tolist(), os.path.join(output_dir, "feature_columns.pkl"))

print(f"Modelos exportados exitosamente a la carpeta '{output_dir}/'")
