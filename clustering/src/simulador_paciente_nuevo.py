import pandas as pd
import joblib
import os

print("========================================")
print(" SIMULADOR DE INFERENCIA CLÍNICA (KNN)")
print("========================================")

# 1. Cargar Modelos (La Inteligencia Artificial "Empaquetada")
try:
    preprocessor = joblib.load("models/preprocessor.pkl")
    pca_5 = joblib.load("models/pca_5.pkl")
    knn = joblib.load("models/knn_model.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    print("[OK] Modelos cargados exitosamente en milisegundos.\n")
except FileNotFoundError:
    print("[ERROR] No se encontraron los modelos. Corre 'exportar_modelos.py' primero.")
    exit(1)

# Nuestro diccionario clínico para traducir el número del clúster a un diagnóstico
diagnosticos = {
    0: "[Clúster 0] Jóvenes Primigestas Sanas (Riesgo Bajo)",
    1: "[Clúster 1] Alto Riesgo Hipertensivo / Preeclampsia (Riesgo Crítico)",
    2: "[Clúster 2] Jóvenes Multíparas Sanas (Riesgo Bajo-Medio)",
    3: "[Clúster 3] Riesgo Metabólico / Obesidad (Riesgo Alto)"
}

# 2. Definir Pacientes Ficticias (Como si acabaran de llegar al hospital)
# Le daremos los datos médicos básicos que capturaría un doctor
pacientes_nuevas = [
    {
        "patient_id": "SIM_001",
        "description": "Chica de 25 años, peso normal, sana, su primer embarazo.",
        "age_years": 25.0,
        "bmi_initial": 22.0,
        "mean_arterial_pressure": 80.0,
        "weight_gain": 8.0,
        "previous_pregnancies": 0.0,
        "previous_deliveries": 0.0,
        "previous_miscarriages": 0.0,
        "previous_cesareans": 0.0,
        "diabetes": 0.0,
        "chronic_hypertension": 0.0,
        "previous_preeclampsia": 0.0,
        "active_smoking": 0.0,
        "multiple_pregnancy": 0.0
    },
    {
        "patient_id": "SIM_002",
        "description": "Mujer de 35 años, sobrepeso, presión alta, con preeclampsia en un embarazo anterior.",
        "age_years": 35.0,
        "bmi_initial": 28.5,
        "mean_arterial_pressure": 115.0, # Alta presión
        "weight_gain": 6.0,
        "previous_pregnancies": 2.0,
        "previous_deliveries": 1.0,
        "previous_miscarriages": 0.0,
        "previous_cesareans": 1.0,
        "diabetes": 0.0,
        "chronic_hypertension": 1.0, # Hipertensión
        "previous_preeclampsia": 1.0, # Preeclampsia
        "active_smoking": 0.0,
        "multiple_pregnancy": 0.0
    }
]

df_simulacion = pd.DataFrame(pacientes_nuevas)
pacientes_info = df_simulacion[['patient_id', 'description']].copy()

# Dejar solo las variables médicas y rellenar las que falten con valores por defecto
df_raw = df_simulacion.drop(columns=['patient_id', 'description'])

# Asegurar que la estructura es exactamente la misma que en el entrenamiento
df_full = pd.read_csv("data/v2/clustering_feature_view.csv")
template = df_full[feature_cols].iloc[0:0].copy()
df_raw_aligned = pd.concat([template, df_raw], ignore_index=True)

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df_full[col]):
        df_raw_aligned[col] = df_raw_aligned[col].fillna(df_full[col].median())
    else:
        df_raw_aligned[col] = df_raw_aligned[col].fillna(df_full[col].mode()[0])

df_raw_aligned = df_raw_aligned[feature_cols].astype(df_full[feature_cols].dtypes)

# 3. Pipeline de Inferencia (La Magia)
print("Ejecutando Inferencia...")
# Paso A: Limpiar y estandarizar datos (como lo aprendió el Preprocesador)
X_scaled = preprocessor.transform(df_raw_aligned)

# Paso B: Geometría de 5 dimensiones (como lo aprendió PCA)
X_pca = pca_5.transform(X_scaled)

# Paso C: Votación de las 15 vecinas más cercanas (Distancia Manhattan)
predicciones = knn.predict(X_pca)

# 4. Mostrar Diagnóstico Predictivo
print("-" * 60)
for i in range(len(pacientes_nuevas)):
    print(f"Paciente : {pacientes_info['patient_id'].iloc[i]}")
    print(f"Ingreso  : {pacientes_info['description'].iloc[i]}")
    cluster_predicho = predicciones[i]
    print(f"==> RESULTADO DE LA IA: {diagnosticos[cluster_predicho]}")
    print("-" * 60)
