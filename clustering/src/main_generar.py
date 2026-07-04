import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, DEFAULT_CONFIG, FEATURE_COLUMNS
from src.generador_base import generar_base
from src.missingness import inyectar_missingness
from src.outliers import inyectar_outliers
from src.validacion import generar_reporte


def main() -> None:
    config = DEFAULT_CONFIG.copy()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    semilla = config["semilla"]
    n = config["n_pacientes"]

    print(f"Generando {n} pacientes (semilla={semilla})...")
    df_completo = generar_base(n, config["proporciones_cluster"], semilla)

    df_con_outliers, meta_outliers = inyectar_outliers(df_completo, config, semilla)
    df_sintetico, log_miss, mecanismo_json = inyectar_missingness(
        df_con_outliers, config, semilla
    )

    meta = meta_outliers.copy()
    meta["mecanismo_nulo"] = mecanismo_json

    cols_sintetico = ["paciente_id"] + FEATURE_COLUMNS
    df_export = df_sintetico[cols_sintetico].copy()
    df_limpio = df_completo[cols_sintetico].copy()

    path_sintetico = DATA_DIR / "datos_embarazo_sintetico.csv"
    path_completo = DATA_DIR / "datos_embarazo_completo.csv"
    path_meta = DATA_DIR / "metadatos_ground_truth.csv"
    path_config = DATA_DIR / "config_generacion.json"
    path_log = DATA_DIR / "missingness_log.csv"
    path_reporte = DATA_DIR / "reporte_calidad.md"

    df_export.to_csv(path_sintetico, index=False)
    df_limpio.to_csv(path_completo, index=False)
    meta.to_csv(path_meta, index=False)
    log_miss.to_csv(path_log, index=False)

    with open(path_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    reporte = generar_reporte(df_completo, df_sintetico, meta, log_miss, config)
    path_reporte.write_text(reporte, encoding="utf-8")

    print(f"  -> {path_sintetico}")
    print(f"  -> {path_completo}")
    print(f"  -> {path_meta}")
    print(f"  -> {path_log}")
    print(f"  -> {path_config}")
    print(f"  -> {path_reporte}")
    print("Listo.")


if __name__ == "__main__":
    main()
