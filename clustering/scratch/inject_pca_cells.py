import json
import os

NB_PATH = r"c:\Universidad\mineria\clustering\notebooks\cluster_grande.ipynb"

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Selección del Número de Componentes (Análisis de Varianza Acumulada)\n",
        "Antes de realizar el clustering, evaluamos cuántos componentes principales necesitamos para retener una cantidad de varianza clínica aceptable (al menos 80%)."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import matplotlib.pyplot as plt\n",
        "from sklearn.decomposition import PCA\n",
        "\n",
        "# Calculamos el PCA con todos los componentes posibles para ver la curva completa\n",
        "pca_full = PCA(random_state=42)\n",
        "pca_full.fit(X_processed)\n",
        "\n",
        "cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')\n",
        "plt.axhline(y=0.80, color='r', linestyle='-')\n",
        "plt.axvline(x=12, color='g', linestyle='-')\n",
        "plt.text(12.5, 0.75, '12 Componentes (80.39%)', color='g', fontsize=12)\n",
        "plt.text(1, 0.82, 'Umbral Clínico del 80%', color='r', fontsize=12)\n",
        "\n",
        "plt.title('Varianza Explicada Acumulada por Componentes Principales')\n",
        "plt.xlabel('Número de Componentes')\n",
        "plt.ylabel('Varianza Acumulada')\n",
        "plt.grid(True)\n",
        "plt.show()"
    ]
}

# Find the comparative cell
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and len(cell['source']) > 0 and 'COMPARATIVA EXPERIMENTAL' in cell['source'][0]:
        insert_idx = i
        break

if insert_idx != -1:
    nb['cells'].insert(insert_idx, markdown_cell)
    nb['cells'].insert(insert_idx + 1, code_cell)
    
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Injected PCA cells successfully!")
else:
    print("Could not find the insertion point.")
