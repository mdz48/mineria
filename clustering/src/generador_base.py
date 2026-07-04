import numpy as np
import pandas as pd

from .clusters import (
    CLUSTER_PROFILES,
    asignar_clusters,
    ganancia_esperada_iom,
    trimestre_desde_semanas,
)


def _clip(val: float, lo: float, hi: float) -> float:
    return float(np.clip(val, lo, hi))


def _generar_fila(cluster: int, paciente_id: int, rng: np.random.Generator) -> dict:
    p = CLUSTER_PROFILES[cluster]

    edad = _clip(rng.normal(*p["edad"]), 15, 49)
    imc = _clip(rng.normal(*p["imc"]), 14, 55)
    if cluster == 4:
        semanas = _clip(rng.normal(33, 5), 6, 42)
        if rng.random() < 0.35:
            semanas = _clip(rng.normal(32, 4), 24, 36)
    else:
        semanas = _clip(rng.normal(*p["semanas"]), 6, 42)

    trimestre = trimestre_desde_semanas(semanas)
    talla_cm = _clip(rng.normal(*p["talla"]), 140, 185)

    ganancia = ganancia_esperada_iom(
        imc, semanas, p["ganancia_intercept"], p["ganancia_coef"]
    )
    ganancia += rng.normal(0, 1.5 if cluster != 3 else 0.8)
    if cluster == 1:
        ganancia += rng.uniform(2, 6)
    if cluster == 3:
        ganancia -= rng.uniform(1, 4)
    ganancia = _clip(ganancia, -2, 25)

    peso_base = imc * (talla_cm / 100) ** 2
    peso_kg = _clip(peso_base + ganancia + rng.normal(0, 1.5), 40, 150)

    pa_dia = _clip(rng.normal(*p["pa_dia"]), 45, 120)
    delta = rng.uniform(20, 55)
    pa_sys = _clip(pa_dia + delta, 80, 180)
    fc = _clip(rng.normal(*p["fc"]), 55, 110)

    diabetes = int(rng.random() < p["p_diabetes"])
    if diabetes and imc < 27:
        imc = _clip(rng.normal(29, 2), 27, 45)

    hipertension = int(rng.random() < p["p_hipertension"])
    if hipertension:
        pa_sys = _clip(pa_sys + rng.uniform(5, 15), 80, 180)
        pa_dia = _clip(pa_dia + rng.uniform(3, 10), 45, 120)
        if pa_sys <= pa_dia + 15:
            pa_sys = pa_dia + rng.uniform(18, 40)

    tabaco = int(rng.random() < p["p_tabaco"])
    multiple = int(rng.random() < p["p_multiple"])

    if rng.random() < p["p_emb_previos"]:
        num_emb = int(rng.integers(1, 6))
    else:
        num_emb = 0
    num_partos = int(rng.integers(0, num_emb + 1)) if num_emb > 0 else 0

    if rng.random() < p["p_suplemento"]:
        sup_folico, sup_hierro = 1, 1
    else:
        sup_folico = int(rng.random() < 0.5)
        sup_hierro = int(rng.random() < 0.4)

    if rng.random() < p["p_educ_superior"]:
        nivel_educacion = "superior"
    elif rng.random() < 0.55:
        nivel_educacion = "secundaria"
    else:
        nivel_educacion = "primaria"

    area = "urbana" if rng.random() < p["p_urbana"] else "rural"

    if edad > 35:
        if rng.random() < 0.08:
            diabetes = 1
        if rng.random() < 0.10:
            hipertension = 1

    return {
        "paciente_id": paciente_id,
        "cluster_verdadero": cluster,
        "edad_anios": round(edad, 1),
        "imc_pregestacional": round(imc, 2),
        "semanas_gestacion": round(semanas, 1),
        "trimestre": trimestre,
        "num_embarazos_previos": num_emb,
        "num_partos_previos": num_partos,
        "embarazo_multiple": multiple,
        "tabaquismo_activo": tabaco,
        "diabetes_previa": diabetes,
        "hipertension_cronica": hipertension,
        "peso_kg": round(peso_kg, 1),
        "talla_cm": round(talla_cm, 1),
        "ganancia_peso_kg": round(ganancia, 1),
        "presion_sistolica": round(pa_sys, 0),
        "presion_diastolica": round(pa_dia, 0),
        "frecuencia_cardiaca": round(fc, 0),
        "nivel_educacion": nivel_educacion,
        "area_residencia": area,
        "suplemento_acido_folico": sup_folico,
        "suplemento_hierro": sup_hierro,
    }


def aplicar_restricciones(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trimestre"] = df["semanas_gestacion"].apply(trimestre_desde_semanas)
    df["num_partos_previos"] = df.apply(
        lambda r: min(r["num_partos_previos"], r["num_embarazos_previos"]), axis=1
    )

    mask_pa = df["presion_sistolica"] <= df["presion_diastolica"] + 14
    df.loc[mask_pa, "presion_sistolica"] = (
        df.loc[mask_pa, "presion_diastolica"] + 20
    )

    for idx, row in df.iterrows():
        talla_m = row["talla_cm"] / 100
        peso_esperado = row["imc_pregestacional"] * talla_m**2 + row["ganancia_peso_kg"]
        if abs(row["peso_kg"] - peso_esperado) > 8:
            df.at[idx, "peso_kg"] = round(peso_esperado + np.random.default_rng(idx).normal(0, 1), 1)

    return df


def generar_base(n: int, proporciones: list[float], semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    clusters = asignar_clusters(n, proporciones, rng)
    filas = [_generar_fila(int(c), i + 1, rng) for i, c in enumerate(clusters)]
    df = pd.DataFrame(filas)
    return aplicar_restricciones(df)
