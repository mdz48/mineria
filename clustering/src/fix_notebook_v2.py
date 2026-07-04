import json

notebook_path = r"c:\Universidad\mineria\clustering\notebooks\cluster.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    # Fix paths
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            line = line.replace('"../data/v2/clustering_feature_view.csv"', '"clustering/data/v2/clustering_feature_view.csv"')
            line = line.replace("'../data/v2/ground_truth.csv'", "'clustering/data/v2/ground_truth.csv'")
            
            # Fix X_pca to X_cluster in DBSCAN cell
            line = line.replace('X_pca', 'X_cluster')
            
            new_source.append(line)
        cell['source'] = new_source

    # Fix Markdown (remove FastAPI mention)
    if cell['cell_type'] == 'markdown':
        new_source = []
        for line in cell['source']:
            line = line.replace("En un entorno de producción (como nuestra API de FastAPI), cuando llega una paciente nueva, ", "En un entorno productivo, cuando llega una paciente nueva, ")
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook fixes applied successfully.")
