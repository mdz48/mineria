import numpy as np

CLUSTER_NAMES = {
    0: "bajo_riesgo",
    1: "riesgo_metabolico",
    2: "riesgo_hipertensivo",
    3: "bajo_peso_nutricional",
    4: "alto_riesgo_obstetrico",
    5: "residual",
}

# Perfiles por cluster: medias y desviaciones para generación condicionada
CLUSTER_PROFILES = {
    0: {
        "edad": (28.0, 4.0),
        "imc": (23.0, 2.0),
        "semanas": (30.0, 8.0),
        "pa_sys": (110.0, 6.0),
        "pa_dia": (68.0, 5.0),
        "fc": (80.0, 6.0),
        "talla": (162.0, 6.0),
        "ganancia_intercept": 0.3,
        "ganancia_coef": 0.35,
        "p_diabetes": 0.02,
        "p_hipertension": 0.03,
        "p_tabaco": 0.03,
        "p_multiple": 0.01,
        "p_emb_previos": 0.35,
        "p_suplemento": 0.85,
        "p_educ_superior": 0.45,
        "p_urbana": 0.70,
    },
    1: {
        "edad": (32.0, 5.0),
        "imc": (32.5, 3.0),
        "semanas": (28.0, 7.0),
        "pa_sys": (125.0, 8.0),
        "pa_dia": (78.0, 6.0),
        "fc": (84.0, 7.0),
        "talla": (165.0, 7.0),
        "ganancia_intercept": 0.5,
        "ganancia_coef": 0.45,
        "p_diabetes": 0.40,
        "p_hipertension": 0.12,
        "p_tabaco": 0.12,
        "p_multiple": 0.02,
        "p_emb_previos": 0.40,
        "p_suplemento": 0.60,
        "p_educ_superior": 0.30,
        "p_urbana": 0.75,
    },
    2: {
        "edad": (36.0, 5.0),
        "imc": (27.0, 4.0),
        "semanas": (32.0, 6.0),
        "pa_sys": (145.0, 10.0),
        "pa_dia": (92.0, 8.0),
        "fc": (86.0, 8.0),
        "talla": (163.0, 6.0),
        "ganancia_intercept": 0.15,
        "ganancia_coef": 0.25,
        "p_diabetes": 0.08,
        "p_hipertension": 0.55,
        "p_tabaco": 0.05,
        "p_multiple": 0.03,
        "p_emb_previos": 0.25,
        "p_suplemento": 0.70,
        "p_educ_superior": 0.35,
        "p_urbana": 0.65,
    },
    3: {
        "edad": (24.0, 5.0),
        "imc": (17.2, 1.0),
        "semanas": (29.0, 8.0),
        "pa_sys": (105.0, 7.0),
        "pa_dia": (65.0, 5.0),
        "fc": (82.0, 7.0),
        "talla": (158.0, 5.0),
        "ganancia_intercept": 0.05,
        "ganancia_coef": 0.18,
        "p_diabetes": 0.02,
        "p_hipertension": 0.04,
        "p_tabaco": 0.06,
        "p_multiple": 0.02,
        "p_emb_previos": 0.30,
        "p_suplemento": 0.35,
        "p_educ_superior": 0.15,
        "p_urbana": 0.40,
    },
    4: {
        "edad": (39.0, 4.0),
        "imc": (26.0, 5.0),
        "semanas": (33.0, 5.0),
        "pa_sys": (130.0, 12.0),
        "pa_dia": (82.0, 8.0),
        "fc": (88.0, 9.0),
        "talla": (164.0, 6.0),
        "ganancia_intercept": 0.35,
        "ganancia_coef": 0.38,
        "p_diabetes": 0.25,
        "p_hipertension": 0.30,
        "p_tabaco": 0.04,
        "p_multiple": 0.45,
        "p_emb_previos": 0.55,
        "p_suplemento": 0.75,
        "p_educ_superior": 0.40,
        "p_urbana": 0.60,
    },
    5: {
        "edad": (30.0, 6.0),
        "imc": (25.0, 4.0),
        "semanas": (27.0, 9.0),
        "pa_sys": (118.0, 12.0),
        "pa_dia": (72.0, 10.0),
        "fc": (83.0, 10.0),
        "talla": (161.0, 7.0),
        "ganancia_intercept": 0.25,
        "ganancia_coef": 0.32,
        "p_diabetes": 0.10,
        "p_hipertension": 0.10,
        "p_tabaco": 0.06,
        "p_multiple": 0.05,
        "p_emb_previos": 0.35,
        "p_suplemento": 0.65,
        "p_educ_superior": 0.30,
        "p_urbana": 0.55,
    },
}


def asignar_clusters(n: int, proporciones: list[float], rng: np.random.Generator) -> np.ndarray:
    proporciones = np.array(proporciones, dtype=float)
    proporciones /= proporciones.sum()
    counts = rng.multinomial(n, proporciones)
    clusters = np.repeat(np.arange(len(proporciones)), counts)
    rng.shuffle(clusters)
    return clusters


def trimestre_desde_semanas(semanas: float) -> int:
    if semanas <= 13:
        return 1
    if semanas <= 27:
        return 2
    return 3


def ganancia_esperada_iom(imc: float, semanas: float, intercept: float, coef: float) -> float:
    if imc < 18.5:
        base = 0.5
    elif imc < 25:
        base = 0.35
    elif imc < 30:
        base = 0.25
    else:
        base = 0.15
    t = trimestre_desde_semanas(semanas)
    factor_t = {1: 0.3, 2: 0.65, 3: 1.0}[t]
    return intercept + base + coef * semanas * factor_t * 0.1
