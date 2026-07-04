import json

notebook_path = r"c:\Universidad\mineria\clustering\notebooks\cluster.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Fix data paths
            line = line.replace('"clustering/data/v2/clustering_feature_view.csv"', '"../data/v2/clustering_feature_view.csv"')
            line = line.replace("'clustering/data/v2/clustering_feature_view.csv'", "'../data/v2/clustering_feature_view.csv'")
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Paths fixed in notebook.")
