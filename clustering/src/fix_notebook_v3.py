import json

notebook_path = r"c:\Universidad\mineria\clustering\notebooks\cluster.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Revertir a ../data/v2/ para que corra desde la carpeta notebooks
            line = line.replace('"clustering/data/v2/clustering_feature_view.csv"', '"../data/v2/clustering_feature_view.csv"')
            line = line.replace("'clustering/data/v2/ground_truth.csv'", "'../data/v2/ground_truth.csv'")
            
            # Asegurarnos de que X_pca se cambió a X_cluster (en caso de que lo vuelva a correr)
            line = line.replace('X_pca', 'X_cluster')
            
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook paths correctly set to ../data/v2/ and X_cluster variable applied.")
