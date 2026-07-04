import numpy as np

# Perfiles base para Clusters 0-4
CLUSTER_PROFILES_V2 = {
    0: { # Control normal
        "age": (28.0, 4.0), "bmi": (23.0, 2.0), "systolic": (110.0, 5.0), "diastolic": (68.0, 4.0),
        "diabetes": 0.01, "chronic_hypertension": 0.01, "previous_preeclampsia": 0.005,
        "family_history_hypertension": 0.10, "family_history_heart_disease": 0.05,
        "chronic_kidney_disease": 0.001, "multiple_pregnancy": 0.01, "active_smoking": 0.05,
        "nulliparous": 0.35, "education": [0.6, 0.3, 0.1], # superior, secundaria, primaria
        "rural": 0.15,
        "gestational_week": (32.0, 4.0), # T3 mostly
        "weight_gain_coef": 1.0 # Multiplicador sobre ganancia esperada
    },
    1: { # Riesgo cardiometabolico
        "age": (30.0, 5.0), "bmi": (32.0, 3.0), "systolic": (125.0, 6.0), "diastolic": (80.0, 5.0),
        "diabetes": 0.40, "chronic_hypertension": 0.05, "previous_preeclampsia": 0.02,
        "family_history_hypertension": 0.20, "family_history_heart_disease": 0.15,
        "chronic_kidney_disease": 0.01, "multiple_pregnancy": 0.02, "active_smoking": 0.08,
        "nulliparous": 0.35, "education": [0.5, 0.4, 0.1],
        "rural": 0.20,
        "gestational_week": (30.0, 6.0),
        "weight_gain_coef": 1.4 # Sube peso rápido
    },
    2: { # Hipertensivo / Preeclampsia
        "age": (32.0, 6.0), "bmi": (26.0, 3.0), "systolic": (148.0, 8.0), "diastolic": (95.0, 6.0),
        "diabetes": 0.05, "chronic_hypertension": 0.50, "previous_preeclampsia": 0.25,
        "family_history_hypertension": 0.35, "family_history_heart_disease": 0.10,
        "chronic_kidney_disease": 0.02, "multiple_pregnancy": 0.03, "active_smoking": 0.10,
        "nulliparous": 0.45, "education": [0.4, 0.4, 0.2],
        "rural": 0.25,
        "gestational_week": (28.0, 7.0),
        "weight_gain_coef": 0.8 # Baja ganancia por restriccion
    },
    3: { # Deficit antropometrico (incluye tabaquismo frec)
        "age": (24.0, 5.0), "bmi": (17.5, 1.0), "systolic": (105.0, 5.0), "diastolic": (65.0, 5.0),
        "diabetes": 0.01, "chronic_hypertension": 0.01, "previous_preeclampsia": 0.01,
        "family_history_hypertension": 0.05, "family_history_heart_disease": 0.02,
        "chronic_kidney_disease": 0.005, "multiple_pregnancy": 0.005, "active_smoking": 0.25,
        "nulliparous": 0.50, "education": [0.1, 0.4, 0.5],
        "rural": 0.60,
        "gestational_week": (34.0, 4.0),
        "weight_gain_coef": 0.6 # Muy baja ganancia
    },
    4: { # Muy alto riesgo obstetrico
        "age": (38.0, 4.0), "bmi": (28.0, 4.0), "systolic": (135.0, 12.0), "diastolic": (88.0, 8.0),
        "diabetes": 0.20, "chronic_hypertension": 0.20, "previous_preeclampsia": 0.15,
        "family_history_hypertension": 0.25, "family_history_heart_disease": 0.20,
        "chronic_kidney_disease": 0.20, "multiple_pregnancy": 0.45, "active_smoking": 0.15,
        "nulliparous": 0.20, "education": [0.3, 0.4, 0.3],
        "rural": 0.30,
        "gestational_week": (32.0, 5.0),
        "weight_gain_coef": 1.1
    }
}

def asignar_clusters(n: int, proporciones: list[float], rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(proporciones), size=n, p=proporciones)
