import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

MAPPING_VARS = {
    "paciente_id": "patient_id",
    "edad_anios": "age_years",
    "imc_pregestacional": "bmi_initial",
    "semanas_gestacion": "gestational_week",
    "trimestre": "gestational_trimester",
    "peso_kg": "weight_kg",
    "talla_cm": "height_cm",
    "ganancia_peso_kg": "weight_gain",
    "presion_sistolica": "systolic",
    "presion_diastolica": "diastolic",
    "diabetes_previa": "diabetes",
    "hipertension_cronica": "chronic_hypertension",
    "embarazo_multiple": "multiple_pregnancy",
    "num_embarazos_previos": "previous_pregnancies",
    "num_partos_previos": "previous_deliveries",
    "nivel_educacion": "education_level",
    "area_residencia": "residence",
    "tabaquismo_activo": "active_smoking",
    "frecuencia_cardiaca": "mean_arterial_pressure",
    "suplemento_acido_folico": "previous_preeclampsia",
    "suplemento_hierro": "active_smoking"
}

def migrar_celda(source_lines, is_code):
    new_lines = []
    for line in source_lines:
        # File path replacement
        line = line.replace('"datos_embarazo_sintetico.csv"', '"v2/clustering_feature_view.csv"')
        line = line.replace("'datos_embarazo_sintetico.csv'", "'v2/clustering_feature_view.csv'")
        
        line = line.replace('"metadatos_ground_truth.csv"', '"v2/ground_truth.csv"')
        line = line.replace("'metadatos_ground_truth.csv'", "'v2/ground_truth.csv'")
        
        line = line.replace('"missingness_log.csv"', '"v2/missingness_log.csv"')
        line = line.replace("'missingness_log.csv'", "'v2/missingness_log.csv'")
        
        # Variables replacement (longest keys first to avoid substring overlap)
        for old_var in sorted(MAPPING_VARS.keys(), key=len, reverse=True):
            new_var = MAPPING_VARS[old_var]
            line = line.replace(f'"{old_var}"', f'"{new_var}"')
            line = line.replace(f"'{old_var}'", f"'{new_var}'")
            line = line.replace(f"`{old_var}`", f"`{new_var}`") # for markdown
            line = line.replace(f" {old_var} ", f" {new_var} ")
            
            # Substrings directly attached
            line = line.replace(old_var, new_var)

        new_lines.append(line)
    return new_lines

def migrar_notebook(input_path, output_path):
    if not input_path.exists():
        print(f"No existe el archivo {input_path}")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        is_code = cell.get("cell_type") == "code"
        source = cell.get("source", [])
        
        if isinstance(source, list):
            cell["source"] = migrar_celda(source, is_code)
        elif isinstance(source, str):
            # Sometimes source is a single string instead of a list of lines
            lines = [s + "\n" for s in source.split("\n")]
            if lines: lines[-1] = lines[-1].rstrip("\n") # fixing the last split
            
            new_lines = migrar_celda(lines, is_code)
            cell["source"] = "".join(new_lines)

        # Also empty the outputs to make the notebook clean and smaller for git
        if is_code:
            cell["outputs"] = []
            cell["execution_count"] = None
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook migrado con éxito: {output_path}")

def main():
    print("Iniciando migración de notebooks...")
    migrar_notebook(NOTEBOOKS_DIR / "eda_tablas.ipynb", NOTEBOOKS_DIR / "eda_tablas_v2.ipynb")
    migrar_notebook(NOTEBOOKS_DIR / "eda_graficos.ipynb", NOTEBOOKS_DIR / "eda_graficos_v2.ipynb")
    print("Migración completada.")

if __name__ == "__main__":
    main()
