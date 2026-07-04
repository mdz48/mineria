import json
import os

nb_path = r'c:\Universidad\mineria\clustering\notebooks\cluster.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

md_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## Extracción de Centroides Clínicos (KNN y DBSCAN)\n',
        'A continuación, realizamos la ingeniería inversa para calcular el promedio (centroide) de las variables originales de acuerdo a cómo el modelo de producción (KNN) y el modelo de densidad (DBSCAN) decidieron agrupar a las pacientes.'
    ]
}

code_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import numpy as np\n',
        'import pandas as pd\n',
        '# Aseguramos que tenemos las etiquetas de KNN (ya entrenado)\n',
        'knn_labels = knn_model.predict(X_cluster) # X_cluster es el PCA usado para entrenar\n',
        '\n',
        '# Asignamos temporalmente las etiquetas al dataframe original para agrupar\n',
        'df_features = df.drop(columns=[\'patient_id\', \'true_cluster_name\', \'true_cluster_id\'], errors=\'ignore\')\n',
        'numeric_cols = df_features.select_dtypes(include=[np.number]).columns\n',
        'df_numeric = df_features[numeric_cols].copy()\n',
        'df_numeric[\'knn_cluster\'] = knn_labels\n',
        'df_numeric[\'dbscan_cluster\'] = dbscan_labels\n',
        '\n',
        'key_cols = [\'age_years\', \'bmi_initial\', \'systolic\', \'diastolic\', \'mean_arterial_pressure\', \'weight_kg\', \'previous_pregnancies\']\n',
        '\n',
        'print(\"--- CENTROIDES KNN (PRODUCCIÓN) ---\")\n',
        'knn_centroids = df_numeric.groupby(\'knn_cluster\')[key_cols].mean().round(2)\n',
        'display(knn_centroids)\n',
        '\n',
        'print(\"\\n--- CENTROIDES DBSCAN ---\")\n',
        'dbscan_centroids = df_numeric[df_numeric[\'dbscan_cluster\'] != -1].groupby(\'dbscan_cluster\')[key_cols].mean().round(2)\n',
        'display(dbscan_centroids)\n'
    ]
}

nb['cells'].extend([md_cell, code_cell])

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook actualizado exitosamente.')
