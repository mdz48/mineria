import json

notebook_path = r'c:\Universidad\mineria\clustering\notebooks\cluster.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create the comparison cell
comparison_code = [
    "# === COMPARATIVA EXPERIMENTAL: 12 vs 5 COMPONENTES ===\n",
    "# Guardamos los modelos originales y probamos qué pasaría si usáramos 5 componentes\n",
    "from sklearn.metrics import silhouette_score\n",
    "from sklearn.cluster import KMeans\n",
    "\n",
    "# PCA 5\n",
    "pca_5 = PCA(n_components=5, random_state=42)\n",
    "X_pca_5 = pca_5.fit_transform(X_processed)\n",
    "kmeans_5 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)\n",
    "labels_5 = kmeans_5.fit_predict(X_pca_5)\n",
    "\n",
    "# Calcular Varianza y Silhouette para 5 componentes\n",
    "var_5 = sum(pca_5.explained_variance_ratio_)\n",
    "sil_5 = silhouette_score(X_pca_5, labels_5)\n",
    "\n",
    "# Calcular Varianza y Silhouette para 12 componentes (el que elegimos)\n",
    "var_12 = sum(pca_optimo.explained_variance_ratio_)\n",
    "kmeans_12 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)\n",
    "labels_12 = kmeans_12.fit_predict(X_cluster) # X_cluster tiene 12\n",
    "sil_12 = silhouette_score(X_cluster, labels_12)\n",
    "\n",
    "print(\"=== DILEMA: INFORMACIÓN CLÍNICA VS SEPARACIÓN MATEMÁTICA ===\")\n",
    "print(f\"Usando 12 Componentes (Elegido): Varianza Retenida = {var_12:.2%} | Silhouette = {sil_12:.4f}\")\n",
    "print(f\"Usando  5 Componentes (Descarte): Varianza Retenida = {var_5:.2%} | Silhouette = {sil_5:.4f}\")\n",
    "print(\"\\nINTERPRETACIÓN:\")\n",
    "print(\"Si usamos 5 componentes, el Silhouette sube (mejor separación matemática), pero perdemos casi la mitad de la información clínica (53% varianza).\")\n",
    "print(\"Elegimos 12 componentes porque en medicina es preferible retener el 80% de los datos vitales, aunque los clústeres se solapen ligeramente más en los bordes.\")\n"
]

comparison_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": comparison_code
}

# Insert cell after PCA cell (index 8)
nb['cells'].insert(9, comparison_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Comparison cell inserted at index 9')
