import json

notebook_path = r'c:\Universidad\mineria\clustering\notebooks\cluster.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "joblib.dump(xgb_model, '../models/xgboost_model.pkl')" in source:
            # Reemplazar la forma estática por una dinámica
            new_source = source.replace(
                "joblib.dump(xgb_model, '../models/xgboost_model.pkl')",
                "import os\n"
                "# Determinar ruta dinámica por culpa del kernel de Jupyter\n"
                "if os.path.exists('../models'):\n"
                "    model_dir = '../models'\n"
                "elif os.path.exists('clustering/models'):\n"
                "    model_dir = 'clustering/models'\n"
                "else:\n"
                "    model_dir = 'models'\n"
                "os.makedirs(model_dir, exist_ok=True)\n"
                "model_path = os.path.join(model_dir, 'xgboost_model.pkl')\n"
                "joblib.dump(xgb_model, model_path)"
            )
            new_source = new_source.replace(
                "print(\"Modelo XGBoost guardado en '../models/xgboost_model.pkl'\")",
                "print(f\"Modelo XGBoost guardado en '{model_path}'\")"
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line] # reconstruct lines
            # keep trailing newlines correct
            if new_source.endswith('\n'):
                cell['source'][-1] = cell['source'][-1]
            else:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Ruta arreglada en el notebook.")
