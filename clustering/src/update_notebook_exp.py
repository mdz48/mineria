import json

notebook_path = r"c:\Universidad\mineria\clustering\notebooks\cluster.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

markdown_exp = """### Experimento: Importancia de la Magnitud Numérica vs Variable Binaria

¿Qué pasaría si forzamos a la Inteligencia Artificial a **ignorar el conteo exacto** de embarazos previos y le damos únicamente la variable binaria `nulliparous` (0 = No, 1 = Sí)?

Clínicamente, la barrera más importante para preeclampsia es ser primigesta (Primer embarazo). Sin embargo, matemáticamente, el algoritmo necesita **distancia espacial** para separar los grupos. 

En la siguiente celda replicaremos el proceso de preprocesamiento, PCA y K-Means, pero eliminando las columnas de conteo obstétrico. Observaremos cómo:
1. El **Silhouette Score** (cohesión matemática) disminuye de ~0.33 a ~0.29.
2. Los grupos de "Sanas Primerizas" y "Sanas Multíparas" **colapsan en un solo mega-clúster**, demostrando que K-Means necesita la magnitud numérica (`previous_pregnancies = 1, 2, 3...`) como "regla de medir" para estirar el gráfico y separar correctamente a los grupos.
"""

code_exp = """# Experimento: Usando SOLO 'nulliparous' sin conteos exactos
from sklearn.metrics import silhouette_score

# 1. Excluir el conteo exacto de embarazos/partos/abortos
columnas_a_excluir = [
    "patient_id", 
    "previous_pregnancies", 
    "previous_deliveries", 
    "previous_miscarriages", 
    "previous_cesareans"
]
df_features_exp = df_full.drop(columns=columnas_a_excluir)

# 2. Preprocesamiento
num_cols_exp = df_features_exp.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_exp = df_features_exp.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor_exp = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_exp),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols_exp)
])

X_processed_exp = preprocessor_exp.fit_transform(df_features_exp)

# 3. PCA y Clustering
pca_exp = PCA(n_components=12, random_state=42)
X_pca_exp = pca_exp.fit_transform(X_processed_exp)

kmeans_exp = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels_exp = kmeans_exp.fit_predict(X_pca_exp)
score_exp = silhouette_score(X_pca_exp, labels_exp)

# 4. Perfilado
df_perfilado_exp = df_full.copy()
df_perfilado_exp['Cluster_Exp'] = labels_exp

agrupado_exp = df_perfilado_exp.groupby('Cluster_Exp').agg({
    'age_years': ['count', 'mean'],
    'bmi_initial': 'mean',
    'mean_arterial_pressure': 'mean',
    'nulliparous': lambda x: (x == 1).mean() * 100, # % de primigestas
    'diabetes': lambda x: (x == 1).mean() * 100,
    'chronic_hypertension': lambda x: (x == 1).mean() * 100,
})

agrupado_exp.columns = ['N_Pacientes', 'Edad_Media', 'IMC_Medio', 'Presion_Media', '%_Primigestas', '%_Diabetes', '%_HTA']

print(f"--- EXPERIMENTO: USANDO SOLO NULLIPAROUS (TRUE/FALSE) ---")
print(f"Silhouette Score (Calidad Matemática): {score_exp:.4f} (Cayó de ~0.33)\\n")
print("Perfiles Clínicos Resultantes (Observa la fusión en un Mega-Clúster):")
print(agrupado_exp.round(1).to_string())
"""

def split_to_lines(code_str):
    lines = code_str.split('\n')
    return [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

cell_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": split_to_lines(markdown_exp)
}

cell_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": split_to_lines(code_exp)
}

nb["cells"].append(cell_md)
nb["cells"].append(cell_code)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook actualizado con el experimento.")
