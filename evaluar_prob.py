import pandas as pd
import numpy as np
import joblib

data_path = 'c:/Universidad/mineria/clustering/data/v2/clustering_feature_view.csv'
df = pd.read_csv(data_path)

preprocessor = joblib.load('c:/Universidad/mineria/clustering/models/preprocessor.pkl')
pca = joblib.load('c:/Universidad/mineria/clustering/models/pca_5.pkl')
knn = joblib.load('c:/Universidad/mineria/clustering/models/knn_model.pkl')

p_sana = df[(df['age_years'] < 30) & (df['bmi_initial'] < 23) & (df['systolic'] < 115)].iloc[0:1]
p_frontera = df[(df['bmi_initial'] > 28) & (df['bmi_initial'] < 32) & (df['systolic'] < 130)].iloc[0:1]
p_riesgo = df[(df['systolic'] > 145) & (df['diastolic'] > 95)].iloc[0:1]

pacientes_test = pd.concat([p_sana, p_frontera, p_riesgo])
X_raw = pacientes_test.drop(columns=['patient_id'], errors='ignore')

X_scaled = preprocessor.transform(X_raw)
X_pca = pca.transform(X_scaled)

knn_probs = knn.predict_proba(X_pca)

labels = ['C0: Sana Primeriza', 'C1: Riesgo Hipertensivo', 'C2: Sana Multipara', 'C3: Riesgo Metabolico']

with open('c:/Universidad/mineria/reporte_probabilidades.md', 'w', encoding='utf-8') as f:
    f.write('# Reporte de Pertenencia Porcentual a Clústeres (Soft Classification)\n\n')
    f.write('Al utilizar el método `predict_proba()` de nuestro modelo de producción (KNN), obtenemos un **porcentaje de pertenencia**, lo cual es vital para pacientes en la zona gris (frontera).\n\n')
    f.write('A continuación se analizan 3 casos de prueba:\n\n')

    for i, (idx, row) in enumerate(pacientes_test.iterrows()):
        perfil = ""
        if i == 0: perfil = "Paciente Joven Sana (Esperado: Clúster 0 o 2)"
        if i == 1: perfil = "Paciente con Sobrepeso y Presión Limítrofe (Frontera)"
        if i == 2: perfil = "Paciente Hipertensa Aguda (Esperado: Clúster 1)"
        
        texto = f"### CASO {i+1}: {perfil}\n"
        texto += f"**Perfil Clínico**: Edad {row['age_years']}, IMC {row['bmi_initial']:.1f}, Presión {row['systolic']}/{row['diastolic']}, Embarazos {row['previous_pregnancies']}\n\n"
        
        texto += "#### Veredicto Probabilístico KNN (Basado en la proporción de los 7 vecinos más cercanos)\n"
        for j, label in enumerate(labels):
            texto += f"- **{label}**: {knn_probs[i][j]*100:.1f}%\n"
        
        texto += "\n---\n"
        f.write(texto)
