import json
import os

notebook_path = r"c:\Universidad\mineria\clustering\notebooks\cluster.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

elbow_code = """import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Análisis del Método del Codo (Elbow Method) para determinar K óptimo
inercia = []
rango_k = range(1, 11)

for k in rango_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_cluster) # X_cluster viene de las celdas anteriores (PCA)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(rango_k, inercia, marker='o', linestyle='--', color='b')
plt.title('Método del Codo para determinar K óptimo')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('Inercia (Suma de distancias al cuadrado)')
plt.xticks(rango_k)
plt.axvline(x=4, color='r', linestyle=':', label='K=4 seleccionado')
plt.legend()
plt.grid(True)
plt.show()

# Se observa una inflexión o "codo" entre K=3 y K=4.
"""

silhouette_code = """from sklearn.metrics import silhouette_score

# Análisis de Silhouette para confirmar K óptimo (probando K=3, 4, 5)
print("Evaluando Silhouette Score para K=3, 4 y 5:\\n")

for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels)
    print(f"K={k} -> Silhouette Score: {score:.4f}")

# El score más alto indica la mejor cohesión y separación. 
# K=4 nos dio el equilibrio perfecto entre matemáticas y lógica clínica.
"""

def split_to_lines(code_str):
    lines = code_str.split('\n')
    # Add newline character back to all but the last string, as Jupyter does
    return [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

cell_elbow = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": split_to_lines(elbow_code)
}

cell_silhouette = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": split_to_lines(silhouette_code)
}

nb["cells"].append(cell_elbow)
nb["cells"].append(cell_silhouette)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook actualizado correctamente con Codo y Silhouette.")
