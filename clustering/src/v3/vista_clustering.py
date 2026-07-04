import pandas as pd
from .config_v3 import CLUSTERING_FEATURES

def materializar_vista_clustering(df_patient: pd.DataFrame, df_mr: pd.DataFrame, df_diary: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(df_patient, df_mr, on="patient_id", how="inner")
    
    # Filter only clustering snapshots
    df_diary_snapshot = df_diary[df_diary["is_clustering_snapshot"] == 1]
    df = pd.merge(df, df_diary_snapshot, on="medical_record_id", how="inner")
    
    # Nulliparous was requested in plan as a derived column (boolean/int)
    df["nulliparous"] = (df["parity_count"] == 0).astype(int)
    
    # Rename columns to match the SQL view aliases from the plan
    df = df.rename(columns={
        "age_at_registration": "age_years",
        "parity_count": "previous_pregnancies",
        "delivery_count": "previous_deliveries",
        "miscarriage_count": "previous_miscarriages",
        "cesarean_count": "previous_cesareans"
    })
    
    # Select final features
    cols_to_select = ["patient_id"] + [c for c in CLUSTERING_FEATURES if c in df.columns]
    
    return df[cols_to_select].copy()
