import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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

print("=" * 75)
print("  COMPARACION DE INICIALIZACIONES K-MEANS")
print("=" * 75)

# --- K-MEANS: Comparar init ---
configs_kmeans = [
    {"init": "k-means++", "n_init": 10, "desc": "k-means++ (n_init=10) [ACTUAL]"},
    {"init": "k-means++", "n_init": 30, "desc": "k-means++ (n_init=30)"},
    {"init": "k-means++", "n_init": 50, "desc": "k-means++ (n_init=50)"},
    {"init": "random",    "n_init": 10, "desc": "random    (n_init=10)"},
    {"init": "random",    "n_init": 30, "desc": "random    (n_init=30)"},
    {"init": "random",    "n_init": 50, "desc": "random    (n_init=50)"},
]

print(f"\n  {'Configuracion':<35} | {'Inercia':>10} | {'Silhouette':>10} | {'C-H':>10} | {'D-B':>10}")
print("  " + "-" * 85)

best_sil = -1
best_config = ""
all_labels = {}

for cfg in configs_kmeans:
    km = KMeans(n_clusters=4, init=cfg["init"], n_init=cfg["n_init"], random_state=42)
    labels = km.fit_predict(X_pca)
    
    sil = silhouette_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    inercia = km.inertia_
    
    marker = " <-- MEJOR" if sil > best_sil else ""
    if sil > best_sil:
        best_sil = sil
        best_config = cfg["desc"]
    
    all_labels[cfg["desc"]] = labels
    
    print(f"  {cfg['desc']:<35} | {inercia:>10.1f} | {sil:>10.4f} | {ch:>10.1f} | {db:>10.4f}{marker}")

# Comparar si las etiquetas cambian entre configuraciones
print(f"\n  Mejor configuracion K-Means: {best_config}")

print("\n--- ESTABILIDAD: Las etiquetas cambian entre configuraciones? ---")
base_labels = all_labels["k-means++ (n_init=10) [ACTUAL]"]
for desc, labels in all_labels.items():
    if desc == "k-means++ (n_init=10) [ACTUAL]":
        continue
    coincidencias = np.sum(base_labels == labels)
    pct = coincidencias / len(base_labels) * 100
    print(f"  vs {desc:<35}: {coincidencias}/{len(base_labels)} coinciden ({pct:.1f}%)")

# --- KNN: Comparar hiperparametros ---
print("\n" + "=" * 75)
print("  COMPARACION DE HIPERPARAMETROS KNN")
print("=" * 75)

km_base = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
y = km_base.fit_predict(X_pca)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

configs_knn = [
    {"n": 5,  "weights": "uniform",  "metric": "euclidean",  "desc": "K=5,  uniform,  euclidean"},
    {"n": 5,  "weights": "distance", "metric": "euclidean",  "desc": "K=5,  distance, euclidean"},
    {"n": 7,  "weights": "uniform",  "metric": "euclidean",  "desc": "K=7,  uniform,  euclidean"},
    {"n": 7,  "weights": "distance", "metric": "euclidean",  "desc": "K=7,  distance, euclidean"},
    {"n": 10, "weights": "uniform",  "metric": "euclidean",  "desc": "K=10, uniform,  euclidean"},
    {"n": 10, "weights": "distance", "metric": "euclidean",  "desc": "K=10, distance, euclidean"},
    {"n": 15, "weights": "uniform",  "metric": "euclidean",  "desc": "K=15, uniform,  euclidean"},
    {"n": 15, "weights": "distance", "metric": "euclidean",  "desc": "K=15, distance, euclidean"},
    {"n": 5,  "weights": "uniform",  "metric": "manhattan",  "desc": "K=5,  uniform,  manhattan"},
    {"n": 5,  "weights": "distance", "metric": "manhattan",  "desc": "K=5,  distance, manhattan"},
    {"n": 7,  "weights": "uniform",  "metric": "manhattan",  "desc": "K=7,  uniform,  manhattan"},
    {"n": 7,  "weights": "distance", "metric": "manhattan",  "desc": "K=7,  distance, manhattan"},
    {"n": 15, "weights": "distance", "metric": "manhattan",  "desc": "K=15, distance, manhattan [ACTUAL]"},
    {"n": 5,  "weights": "uniform",  "metric": "minkowski",  "desc": "K=5,  uniform,  minkowski(p=3)"},
    {"n": 7,  "weights": "distance", "metric": "minkowski",  "desc": "K=7,  distance, minkowski(p=3)"},
]

print(f"\n  {'Configuracion':<40} | {'Acc Train':>10} | {'Acc Test':>10}")
print("  " + "-" * 65)

best_test = 0
best_knn_desc = ""

for cfg in configs_knn:
    p_val = 3 if cfg["metric"] == "minkowski" else 2
    knn = KNeighborsClassifier(
        n_neighbors=cfg["n"], 
        weights=cfg["weights"], 
        metric=cfg["metric"],
        p=p_val
    )
    knn.fit(X_train, y_train)
    
    acc_train = accuracy_score(y_train, knn.predict(X_train))
    acc_test = accuracy_score(y_test, knn.predict(X_test))
    
    marker = ""
    if acc_test > best_test:
        best_test = acc_test
        best_knn_desc = cfg["desc"]
        marker = " <-- MEJOR"
    elif acc_test == best_test:
        marker = " <-- EMPATE"
    
    print(f"  {cfg['desc']:<40} | {acc_train:>9.2%} | {acc_test:>9.2%}{marker}")

print(f"\n  Mejor configuracion KNN: {best_knn_desc} ({best_test:.2%})")

# Detalle del mejor KNN
print(f"\n--- REPORTE DETALLADO DEL MEJOR KNN ---")
best_cfg = next(c for c in configs_knn if c["desc"] == best_knn_desc)
p_val = 3 if best_cfg["metric"] == "minkowski" else 2
knn_best = KNeighborsClassifier(
    n_neighbors=best_cfg["n"],
    weights=best_cfg["weights"],
    metric=best_cfg["metric"],
    p=p_val
)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)
print(classification_report(y_test, y_pred, target_names=[
    "C0: Primigestas Sanas",
    "C1: Riesgo Hipertensivo",
    "C2: Multiparas Sanas",
    "C3: Riesgo Metabolico"
]))
