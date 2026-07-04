from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_CONFIG = {
    "semilla": 42,
    "n_pacientes": 2000,
    "proporciones_cluster": [0.45, 0.18, 0.12, 0.15, 0.07, 0.03],
    "outliers": {
        "tasa_fila": 0.05,
        "tasa_extremo": 0.015,
        "intensidad_perturbacion": 5.0,
        "prob_mecanismo": {
            "medicion": 0.40,
            "registro": 0.25,
            "combinacion_imposible": 0.20,
            "patologia_rara": 0.10,
            "posicion": 0.05,
        },
    },
    "missingness": {
        "mcar_prob_celda": 0.03,
        "mar_prob_base": 0.08,
        "mnar_prob_base": 0.06,
    },
}

FEATURE_COLUMNS = [
    "edad_anios",
    "imc_pregestacional",
    "semanas_gestacion",
    "trimestre",
    "num_embarazos_previos",
    "num_partos_previos",
    "embarazo_multiple",
    "tabaquismo_activo",
    "diabetes_previa",
    "hipertension_cronica",
    "peso_kg",
    "talla_cm",
    "ganancia_peso_kg",
    "presion_sistolica",
    "presion_diastolica",
    "frecuencia_cardiaca",
    "nivel_educacion",
    "area_residencia",
    "suplemento_acido_folico",
    "suplemento_hierro",
]

MCAR_COLUMNS = [
    "edad_anios",
    "imc_pregestacional",
    "semanas_gestacion",
    "peso_kg",
    "talla_cm",
    "ganancia_peso_kg",
    "presion_sistolica",
    "presion_diastolica",
    "frecuencia_cardiaca",
    "num_embarazos_previos",
]

NUMERIC_COLUMNS = [
    "edad_anios",
    "imc_pregestacional",
    "semanas_gestacion",
    "trimestre",
    "num_embarazos_previos",
    "num_partos_previos",
    "embarazo_multiple",
    "tabaquismo_activo",
    "diabetes_previa",
    "hipertension_cronica",
    "peso_kg",
    "talla_cm",
    "ganancia_peso_kg",
    "presion_sistolica",
    "presion_diastolica",
    "frecuencia_cardiaca",
    "suplemento_acido_folico",
    "suplemento_hierro",
]
