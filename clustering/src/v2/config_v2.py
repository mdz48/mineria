import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR_V2 = BASE_DIR / "data" / "v2"

DEFAULT_CONFIG_V2 = {
    "version": "2.0",
    "semilla": 42,
    "n_pacientes": 2000,
    "proporciones_cluster": [0.45, 0.18, 0.12, 0.13, 0.07, 0.05],
    "snapshot": {
        "gestational_week_range": [14, 38],
        "prefer_trimester": [2, 3]
    },
    "outliers": {
        "tasa_fila": 0.05,
        "tasa_extremo": 0.015
    },
    "missingness": {
        "mcar_prob_celda": 0.03,
        "mar_prob_base": 0.08,
        "mnar_prob_base": 0.06
    },
    "export": {
        "include_consultation": False,
        "idioma_columnas": "en",
        "vista_clustering": "clustering_feature_view.csv"
    }
}

CLUSTERING_FEATURES = [
    "age_years", "bmi_initial", "gestational_week", "gestational_trimester",
    "height_cm", "initial_weight", "weight_kg", "weight_gain",
    "systolic", "diastolic", "mean_arterial_pressure",
    "diabetes", "chronic_hypertension", "previous_preeclampsia",
    "family_history_hypertension", "family_history_heart_disease",
    "chronic_kidney_disease", "multiple_pregnancy", "active_smoking",
    "previous_pregnancies", "previous_deliveries", "previous_miscarriages",
    "previous_cesareans", "nulliparous",
    "education_level", "residence", "marital_status"
]
