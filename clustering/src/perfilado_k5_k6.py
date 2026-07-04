import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
X_pca_12 = pca_12.fit_transform(X_processed)

def profile_clusters(k):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_pca_12)
    
    df_perfilado = df_features.copy()
    df_perfilado['Cluster'] = labels
    
    agrupado = df_perfilado.groupby('Cluster').agg({
        'age_years': ['count', 'mean'],
        'bmi_initial': 'mean',
        'mean_arterial_pressure': 'mean',
        'previous_pregnancies': 'mean',
        'diabetes': lambda x: (x == 1).mean() * 100,
        'chronic_hypertension': lambda x: (x == 1).mean() * 100,
        'previous_preeclampsia': lambda x: (x == 1).mean() * 100,
    })
    
    # Flatten multi-level columns
    agrupado.columns = ['N_Pacientes', 'Edad_Media', 'IMC_Medio', 'Presion_Media', 'Embarazos_Previos', '%_Diabetes', '%_Hipertension', '%_Preeclampsia_Previa']
    
    print(f"\n--- PERFILES CLÍNICOS PARA K={k} ---")
    print(agrupado.round(1).to_string())

profile_clusters(4) # Run for 4 to remember the base
profile_clusters(5)
profile_clusters(6)
