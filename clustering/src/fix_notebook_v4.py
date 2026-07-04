import json

notebook_path = r"c:\Universidad\mineria\clustering\notebooks\cluster.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Revertir todo a clustering/data/v2/
            line = line.replace('"../data/v2/clustering_feature_view.csv"', '"clustering/data/v2/clustering_feature_view.csv"')
            line = line.replace("'../data/v2/ground_truth.csv'", "'clustering/data/v2/ground_truth.csv'")
            
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook paths correctly set to clustering/data/v2/")
