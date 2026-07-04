import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
import pandas as pd

from src.v2.config_v2 import DEFAULT_CONFIG_V2, DATA_DIR_V2
from src.v2.clusters_v2 import asignar_clusters
from src.v2.generador_entidades import generar_entidades
from src.v2.vista_clustering import materializar_vista_clustering
from src.v2.outliers_v2 import inyectar_outliers_v2
from src.v2.missingness_v2 import inyectar_missingness_v2

def main():
    DATA_DIR_V2.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG_V2.copy()
    
    n = config["n_pacientes"]
    semilla = config["semilla"]
    proporciones = config["proporciones_cluster"]
    
    print(f"Generando {n} pacientes v2 (semilla={semilla})...")
    rng = np.random.default_rng(semilla)
    clusters = asignar_clusters(n, proporciones, rng)
    
    df_patient, df_mr, df_diary = generar_entidades(n, clusters, rng)
    
    df_gt = pd.DataFrame({
        "patient_id": df_patient["patient_id"],
        "cluster_verdadero": clusters
    })
    
    print("Materializando ClusteringFeatureView...")
    df_vista_completa = materializar_vista_clustering(df_patient, df_mr, df_diary)
    
    print("Inyectando outliers y missingness...")
    df_vista_outliers, meta_outliers = inyectar_outliers_v2(df_vista_completa, config, semilla)
    df_vista_sintetica, log_miss, mecanismo_json = inyectar_missingness_v2(df_vista_outliers, config, semilla)
    
    print("Guardando archivos en data/v2/...")
    df_patient.to_csv(DATA_DIR_V2 / "patient.csv", index=False)
    df_mr.to_csv(DATA_DIR_V2 / "medical_record.csv", index=False)
    df_diary.to_csv(DATA_DIR_V2 / "patient_diary.csv", index=False)
    
    df_vista_completa.to_csv(DATA_DIR_V2 / "clustering_feature_view_completo.csv", index=False)
    df_vista_sintetica.to_csv(DATA_DIR_V2 / "clustering_feature_view.csv", index=False)
    
    df_gt = pd.merge(df_gt, meta_outliers, on="patient_id", how="left")
    df_gt["es_outlier"] = df_gt["es_outlier"].fillna(0).astype(int)
    df_gt.to_csv(DATA_DIR_V2 / "ground_truth.csv", index=False)
    
    log_miss.to_csv(DATA_DIR_V2 / "missingness_log.csv", index=False)
    
    with open(DATA_DIR_V2 / "config_generacion_v2.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    print("¡Generación completada con éxito!")
    print(f"Vista clustering shape: {df_vista_sintetica.shape}")

if __name__ == "__main__":
    main()
