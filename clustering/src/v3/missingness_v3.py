import json
import numpy as np
import pandas as pd

def inyectar_missingness_v2(df_vista: pd.DataFrame, config: dict, semilla: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rng = np.random.default_rng(semilla + 20)
    df = df_vista.copy()
    
    # Cast numeric columns to float to avoid TypeError when assigning np.nan
    for col in ["systolic", "diastolic", "weight_kg", "height_cm", "initial_weight", "previous_pregnancies", "bmi_initial"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    n = len(df)
    
    p_mcar = config["missingness"]["mcar_prob_celda"]
    p_mar = config["missingness"]["mar_prob_base"]
    p_mnar = config["missingness"]["mnar_prob_base"]
    
    log_records = []
    mecanismo_counts = {"mcar": 0, "mar": 0, "mnar": 0}
    
    cols_mcar = ["systolic", "diastolic", "weight_kg", "height_cm", "initial_weight", "previous_pregnancies"]
    for col in cols_mcar:
        if col not in df.columns: continue
        mask = rng.random(n) < p_mcar
        df.loc[mask, col] = np.nan
        mecanismo_counts["mcar"] += mask.sum()
        for idx in df[mask].index:
            log_records.append({"patient_id": df.loc[idx, "patient_id"], "variable": col, "mecanismo": "mcar"})
            
    # MAR
    cols_mar = ["weight_kg"]
    for col in cols_mar:
        if col not in df.columns: continue
        
        # Rule: older age -> more likely to miss weight_kg
        prob = np.where(df["age_years"] > 35, p_mar * 2, p_mar * 0.5)
        mask = rng.random(n) < prob
        df.loc[mask, col] = np.nan
        mecanismo_counts["mar"] += int(mask.sum())
        for idx in df[mask].index: log_records.append({"patient_id": df.loc[idx, "patient_id"], "variable": col, "mecanismo": "mar"})
        
    # Rule: rural residence -> more likely to miss height_cm
    if "height_cm" in df.columns and "residence" in df.columns:
        prob = np.where(df["residence"] == "rural", p_mar * 2, p_mar * 0.5)
        mask = rng.random(n) < prob
        df.loc[mask, "height_cm"] = np.nan
        mecanismo_counts["mar"] += int(mask.sum())
        for idx in df[mask].index: log_records.append({"patient_id": df.loc[idx, "patient_id"], "variable": "height_cm", "mecanismo": "mar"})
        
    # MNAR
    # Rule: high systolic -> more likely to not report
    if "systolic" in df.columns:
        prob = np.where(df["systolic"] > 140, p_mnar * 3, p_mnar * 0.2)
        mask = rng.random(n) < prob
        # Backup original
        df.loc[mask, "systolic"] = np.nan
        mecanismo_counts["mnar"] += int(mask.sum())
        for idx in df[mask].index: log_records.append({"patient_id": df.loc[idx, "patient_id"], "variable": "systolic", "mecanismo": "mnar"})
        
    df_log = pd.DataFrame(log_records) if log_records else pd.DataFrame(columns=["patient_id", "variable", "mecanismo"])
    mecanismo_counts = {k: int(v) for k, v in mecanismo_counts.items()}
    return df, df_log, json.dumps(mecanismo_counts)
