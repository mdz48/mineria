import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os
import matplotlib
matplotlib.use('Agg') # para evitar problemas de GUI en terminal

df = pd.read_csv("data/v2/clustering_feature_view.csv")

if "patient_id" in df.columns:
    df = df.drop(columns=["patient_id"])

# Separa numéricas de categóricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Definimos preprocesamiento: Imputación de nulos (median/mode) + Scaling / OHE
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

X_processed = preprocessor.fit_transform(df)

# Correr PCA con todos los componentes posibles
pca = PCA()
pca.fit(X_processed)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot de la varianza explicada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.80, color='r', linestyle='-', alpha=0.5, label='80% Varianza')
plt.axhline(y=0.90, color='g', linestyle='-', alpha=0.5, label='90% Varianza')
plt.title('Varianza Explicada Acumulada por Componentes Principales (PCA)')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.grid(True)
plt.legend()
plt.tight_layout()

output_dir = r"C:\Users\MAXIMILIANO\.gemini\antigravity\brain\4c77bb59-640a-4763-89f1-8d8a7a3f7184\scratch"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "pca_variance.png")
plt.savefig(plot_path)

# Encontrar codos / puntos de interés
comp_80 = np.argmax(cumulative_variance >= 0.80) + 1
comp_90 = np.argmax(cumulative_variance >= 0.90) + 1
comp_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Total de features originales (tras One-Hot): {X_processed.shape[1]}")
print(f"Componentes necesarios para retener el 80% de la varianza: {comp_80}")
print(f"Componentes necesarios para retener el 90% de la varianza: {comp_90}")
print(f"Componentes necesarios para retener el 95% de la varianza: {comp_95}")
for i in range(15):
    if i < len(cumulative_variance):
        print(f"Comp {i+1}: {cumulative_variance[i]:.2%}")
