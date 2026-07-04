import pandas as pd
import numpy as np
import joblib

data_path = 'c:/Universidad/mineria/clustering/data/v2/clustering_feature_view.csv'
df = pd.read_csv(data_path)

preprocessor = joblib.load('c:/Universidad/mineria/clustering/models/preprocessor.pkl')
pca = joblib.load('c:/Universidad/mineria/clustering/models/pca_5.pkl')
knn = joblib.load('c:/Universidad/mineria/clustering/models/knn_model.pkl')

# Calcular los centroides exactos en el espacio PCA (5 componentes)
X_all_raw = df.drop(columns=['patient_id'], errors='ignore')
X_all_scaled = preprocessor.transform(X_all_raw)
X_all_pca = pca.transform(X_all_scaled)

# Usar KNN para asignar a todos los pacientes a uno de los 4 clusters
# Esto asegura que el centroide 0 corresponda exactamente a lo que KNN llama clase 0 (C0: Sana Primeriza)
predicciones_knn = knn.predict(X_all_pca)

df_pca = pd.DataFrame(X_all_pca)
df_pca['cluster_knn'] = predicciones_knn

# Los centroides estarán ordenados 0, 1, 2, 3
centroides = df_pca.groupby('cluster_knn').mean().values

# Preparar las pacientes de prueba
p_sana = df[(df['age_years'] < 30) & (df['bmi_initial'] < 23) & (df['systolic'] < 115)].iloc[0:1]
p_frontera = df[(df['bmi_initial'] > 28) & (df['bmi_initial'] < 32) & (df['systolic'] < 130)].iloc[0:1]
p_riesgo = df[(df['systolic'] > 145) & (df['diastolic'] > 95)].iloc[0:1]

pacientes_test = pd.concat([p_sana, p_frontera, p_riesgo])
X_test_raw = pacientes_test.drop(columns=['patient_id'], errors='ignore')

X_test_scaled = preprocessor.transform(X_test_raw)
X_test_pca = pca.transform(X_test_scaled)

labels = ['C0: Sana Primeriza', 'C1: Riesgo Hipertensivo', 'C2: Sana Multipara', 'C3: Riesgo Metabolico']

def inverse_distance_probabilities(X, centroids, m=2):
    probs = []
    for x in X:
        distances = np.linalg.norm(centroids - x, axis=1)
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / (distances ** (2 / (m - 1)))
        prob = weights / np.sum(weights)
        probs.append(prob)
    return np.array(probs)

# Compute probabilities using m=1.5 (Fuzzy logic)
probs_fuzzy = inverse_distance_probabilities(X_test_pca, centroides, m=1.5)

with open('c:/Universidad/mineria/reporte_probabilidades_fuzzy.md', 'w', encoding='utf-8') as f:
    f.write('# Reporte de Pertenencia Porcentual (Soft Classification via Distancia Espacial)\n\n')
    f.write('Tienes mucha razón. Como nuestro dataset sintético generó clústeres muy bien separados (y KNN toma los "votos" de la vecindad), el modelo es "demasiado seguro" de sí mismo y arroja 100% casi siempre, porque todos los vecinos de una paciente pertenecen al mismo diagnóstico.\n\n')
    f.write('Para solucionar esto y obtener un **porcentaje de clasificación continuo y útil**, dejamos de contar los votos de los vecinos y pasamos a usar la **Distancia Espacial Matemática a los Centroides Clínicos (Fuzzy Logic)**. Así medimos milimétricamente qué tan cerca está la paciente del núcleo de cada enfermedad y repartimos el 100% de probabilidad de forma inversamente proporcional a esa distancia.\n\n')
    f.write('A continuación se analizan los mismos 3 casos, pero ahora con porcentajes que reflejan la realidad espacial:\n\n')

    for i, (idx, row) in enumerate(pacientes_test.iterrows()):
        perfil = ""
        if i == 0: perfil = "Paciente Joven Sana"
        if i == 1: perfil = "Paciente con Sobrepeso y Presión Limítrofe (Frontera Metabólica)"
        if i == 2: perfil = "Paciente Hipertensa Aguda"
        
        texto = f"### CASO {i+1}: {perfil}\n"
        texto += f"**Perfil Clínico**: Edad {row['age_years']}, IMC {row['bmi_initial']:.1f}, Presión {row['systolic']}/{row['diastolic']}, Embarazos {row['previous_pregnancies']}\n\n"
        
        texto += "#### Pertenencia por Distancia Matemática al Centroide (Fuzzy Score)\n"
        for j, label in enumerate(labels):
            texto += f"- **{label}**: {probs_fuzzy[i][j]*100:.1f}%\n"
        
        texto += "\n---\n"
        f.write(texto)
