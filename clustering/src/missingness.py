import json

import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def inyectar_missingness(
    df: pd.DataFrame,
    config: dict,
    semilla: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(semilla + 2000)
    cfg = config["missingness"]
    df_miss = df.copy()
    log: list[dict] = []

    mcar_cols = [
        "presion_sistolica",
        "presion_diastolica",
        "frecuencia_cardiaca",
        "peso_kg",
        "talla_cm",
        "ganancia_peso_kg",
        "edad_anios",
        "imc_pregestacional",
        "semanas_gestacion",
        "num_embarazos_previos",
    ]

    for idx in df_miss.index:
        pid = int(df_miss.at[idx, "paciente_id"])
        for col in mcar_cols:
            if rng.random() < cfg["mcar_prob_celda"]:
                df_miss.at[idx, col] = np.nan
                log.append({"paciente_id": pid, "variable": col, "mecanismo": "MCAR"})

    mar_base = cfg["mar_prob_base"]
    for idx, row in df_miss.iterrows():
        pid = int(row["paciente_id"])

        if row["presion_sistolica"] < 120 and not pd.isna(row["presion_diastolica"]):
            if rng.random() < _sigmoid(-1.5 + 2.5 * (1 - row["presion_sistolica"] / 120)):
                df_miss.at[idx, "presion_diastolica"] = np.nan
                log.append({"paciente_id": pid, "variable": "presion_diastolica", "mecanismo": "MAR"})

        if row["trimestre"] == 1 and not pd.isna(row["ganancia_peso_kg"]):
            if rng.random() < mar_base * 1.2:
                df_miss.at[idx, "ganancia_peso_kg"] = np.nan
                log.append({"paciente_id": pid, "variable": "ganancia_peso_kg", "mecanismo": "MAR"})

        if row["num_embarazos_previos"] == 0 and not pd.isna(row["num_partos_previos"]):
            if rng.random() < mar_base:
                df_miss.at[idx, "num_partos_previos"] = np.nan
                log.append({"paciente_id": pid, "variable": "num_partos_previos", "mecanismo": "MAR"})

        if row["cluster_verdadero"] == 0 and row["trimestre"] == 1:
            if not pd.isna(row["suplemento_hierro"]) and rng.random() < mar_base * 0.8:
                df_miss.at[idx, "suplemento_hierro"] = np.nan
                log.append({"paciente_id": pid, "variable": "suplemento_hierro", "mecanismo": "MAR"})

        if row["semanas_gestacion"] < 12 and not pd.isna(row["frecuencia_cardiaca"]):
            if rng.random() < mar_base:
                df_miss.at[idx, "frecuencia_cardiaca"] = np.nan
                log.append({"paciente_id": pid, "variable": "frecuencia_cardiaca", "mecanismo": "MAR"})

    mnar_base = cfg["mnar_prob_base"]
    for idx, row in df.iterrows():
        pid = int(row["paciente_id"])

        if row["tabaquismo_activo"] == 1 and not pd.isna(df_miss.at[idx, "tabaquismo_activo"]):
            if rng.random() < _sigmoid(-2 + 3.5):
                df_miss.at[idx, "tabaquismo_activo"] = np.nan
                log.append({"paciente_id": pid, "variable": "tabaquismo_activo", "mecanismo": "MNAR"})

        if row["imc_pregestacional"] > 35 and not pd.isna(df_miss.at[idx, "peso_kg"]):
            p = _sigmoid(-2 + 0.15 * (row["imc_pregestacional"] - 35))
            if rng.random() < p:
                df_miss.at[idx, "peso_kg"] = np.nan
                log.append({"paciente_id": pid, "variable": "peso_kg", "mecanismo": "MNAR"})

        if row["presion_sistolica"] > 160 and not pd.isna(df_miss.at[idx, "presion_sistolica"]):
            if rng.random() < _sigmoid(-1.5 + 0.03 * (row["presion_sistolica"] - 160)):
                df_miss.at[idx, "presion_sistolica"] = np.nan
                log.append({"paciente_id": pid, "variable": "presion_sistolica", "mecanismo": "MNAR"})

        if row["suplemento_acido_folico"] == 0 and not pd.isna(df_miss.at[idx, "suplemento_acido_folico"]):
            if rng.random() < mnar_base * 1.5:
                df_miss.at[idx, "suplemento_acido_folico"] = np.nan
                log.append({"paciente_id": pid, "variable": "suplemento_acido_folico", "mecanismo": "MNAR"})

    mecanismos_por_paciente: dict[int, dict] = {}
    for entry in log:
        pid = entry["paciente_id"]
        if pid not in mecanismos_por_paciente:
            mecanismos_por_paciente[pid] = {}
        mecanismos_por_paciente[pid][entry["variable"]] = entry["mecanismo"]

    mecanismo_json = [
        json.dumps(mecanismos_por_paciente.get(int(pid), {}), ensure_ascii=False)
        for pid in df_miss["paciente_id"]
    ]

    log_df = pd.DataFrame(log)
    return df_miss, log_df, mecanismo_json
