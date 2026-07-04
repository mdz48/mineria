import numpy as np
import pandas as pd

def inyectar_outliers_v2(df_vista: pd.DataFrame, config: dict, semilla: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(semilla + 10)
    df = df_vista.copy()
    
    # Cast numeric columns to float to avoid LossySetitemError
    for col in ["systolic", "diastolic", "height_cm", "bmi_initial", "weight_gain", "gestational_week"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    n = len(df)
    
    tasa_fila = config["outliers"]["tasa_fila"]
    num_outliers = int(n * tasa_fila)
    
    indices = rng.choice(n, size=num_outliers, replace=False)
    meta_records = []
    
    for idx in indices:
        mecanismo = rng.choice(["medicion", "registro", "combinacion_imposible", "patologia_rara", "posicion"])
        
        if mecanismo == "medicion":
            df.loc[idx, "systolic"] = 210.0 + rng.uniform(0, 20)
        elif mecanismo == "registro":
            df.loc[idx, "height_cm"] = 1700.0
        elif mecanismo == "combinacion_imposible":
            df.loc[idx, "bmi_initial"] = 38.0
            df.loc[idx, "weight_gain"] = -8.0
            df.loc[idx, "gestational_week"] = 39.0
        elif mecanismo == "patologia_rara":
            df.loc[idx, "systolic"] = 185.0
            df.loc[idx, "multiple_pregnancy"] = 1
            df.loc[idx, "age_years"] = 17
        elif mecanismo == "posicion":
            df.loc[idx, "systolic"] = 100.0
            df.loc[idx, "diastolic"] = 60.0
            
        meta_records.append({
            "patient_id": df.loc[idx, "patient_id"],
            "es_outlier": 1,
            "tipo_outlier": mecanismo
        })
        
    df_meta = pd.DataFrame(meta_records) if meta_records else pd.DataFrame(columns=["patient_id", "es_outlier", "tipo_outlier"])
    return df, df_meta
