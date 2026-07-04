import numpy as np
import pandas as pd

NUMERIC_OUTLIER_COLS = [
    "edad_anios",
    "imc_pregestacional",
    "semanas_gestacion",
    "peso_kg",
    "talla_cm",
    "ganancia_peso_kg",
    "presion_sistolica",
    "presion_diastolica",
    "frecuencia_cardiaca",
]

MECANISMOS = ["medicion", "registro", "combinacion_imposible", "patologia_rara", "posicion"]


def _elegir_mecanismo(prob: dict, rng: np.random.Generator) -> str:
    keys = list(prob.keys())
    vals = np.array([prob[k] for k in keys], dtype=float)
    vals /= vals.sum()
    return str(rng.choice(keys, p=vals))


def _perturbar_medicion(row: pd.Series, rng: np.random.Generator, intensidad: float) -> pd.Series:
    row = row.copy()
    cols = rng.choice(NUMERIC_OUTLIER_COLS, size=rng.integers(1, 4), replace=False)
    for col in cols:
        std = max(abs(row[col]) * 0.15, 5)
        row[col] = row[col] + rng.normal(0, intensidad * std)
        if col in ("presion_sistolica", "presion_diastolica", "frecuencia_cardiaca", "edad_anios"):
            row[col] = round(row[col])
        elif col == "semanas_gestacion":
            row[col] = round(row[col], 1)
        else:
            row[col] = round(row[col], 2 if col == "imc_pregestacional" else 1)
    return row


def _perturbar_registro(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    row = row.copy()
    col = rng.choice(["edad_anios", "peso_kg", "talla_cm", "presion_sistolica"])
    if col == "edad_anios":
        row[col] = rng.choice([340, 99, 5])
    elif col == "peso_kg":
        row[col] = round(row[col] * rng.choice([10, 0.1]), 1)
    elif col == "talla_cm":
        row[col] = round(row[col] * 10, 1)
    else:
        row[col] = round(row[col] * 10, 0)
    return row


def _perturbar_combinacion(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    row = row.copy()
    row["imc_pregestacional"] = round(rng.uniform(33, 40), 2)
    row["semanas_gestacion"] = round(rng.uniform(36, 40), 1)
    row["ganancia_peso_kg"] = round(rng.uniform(-5, -1), 1)
    row["peso_kg"] = round(rng.uniform(42, 50), 1)
    return row


def _perturbar_patologia_rara(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    row = row.copy()
    row["presion_sistolica"] = round(rng.uniform(170, 200), 0)
    row["presion_diastolica"] = round(rng.uniform(105, 120), 0)
    row["embarazo_multiple"] = 1
    row["edad_anios"] = round(rng.uniform(16, 18), 1)
    row["hipertension_cronica"] = 1
    return row


def _perturbar_posicion(row: pd.Series, cluster: int, rng: np.random.Generator) -> pd.Series:
    row = row.copy()
    if cluster == 1:
        row["presion_sistolica"] = round(rng.uniform(105, 115), 0)
        row["presion_diastolica"] = round(rng.uniform(65, 72), 0)
    elif cluster == 2:
        row["imc_pregestacional"] = round(rng.uniform(22, 25), 2)
    else:
        row["imc_pregestacional"] = round(rng.uniform(28, 31), 2)
    return row


def inyectar_outliers(
    df: pd.DataFrame,
    config: dict,
    semilla: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(semilla + 1000)
    cfg = config["outliers"]
    n = len(df)
    n_outliers = int(n * cfg["tasa_fila"])
    n_extremos = int(n * cfg["tasa_extremo"])

    df_out = df.copy()
    meta = df[["paciente_id", "cluster_verdadero"]].copy()
    meta["es_outlier"] = 0
    meta["tipo_outlier"] = "ninguno"

    indices = []
    for k in sorted(df["cluster_verdadero"].unique()):
        idx_k = df.index[df["cluster_verdadero"] == k].tolist()
        n_k = max(1, int(n_outliers * len(idx_k) / n))
        indices.extend(rng.choice(idx_k, size=min(n_k, len(idx_k)), replace=False).tolist())
    indices = list(dict.fromkeys(indices))[:n_outliers]
    extremos = set(rng.choice(indices, size=min(n_extremos, len(indices)), replace=False))

    for idx in indices:
        mecanismo = _elegir_mecanismo(cfg["prob_mecanismo"], rng)
        row = df_out.loc[idx]
        cluster = int(row["cluster_verdadero"])

        if mecanismo == "medicion":
            new_row = _perturbar_medicion(row, rng, cfg["intensidad_perturbacion"])
        elif mecanismo == "registro":
            new_row = _perturbar_registro(row, rng)
        elif mecanismo == "combinacion_imposible":
            new_row = _perturbar_combinacion(row, rng)
        elif mecanismo == "patologia_rara":
            new_row = _perturbar_patologia_rara(row, rng)
        else:
            new_row = _perturbar_posicion(row, cluster, rng)

        if idx in extremos:
            new_row = _perturbar_medicion(new_row, rng, cfg["intensidad_perturbacion"] * 1.2)

        for col in new_row.index:
            if col in df_out.columns:
                df_out.at[idx, col] = new_row[col]

        meta.at[idx, "es_outlier"] = 1
        meta.at[idx, "tipo_outlier"] = mecanismo

    return df_out, meta
