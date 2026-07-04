import pandas as pd

df_labeled = pd.read_csv("data/v2/labeled_patients.csv")
df_full = pd.read_csv("data/v2/clustering_feature_view.csv")

df_combined = df_labeled.copy()
df_combined['previous_pregnancies'] = df_full['previous_pregnancies']

c0 = df_combined[df_combined['Cluster_Label'] == 0]
print("--- CLUSTER 0 (Verde) ---")
print(c0.groupby('previous_pregnancies')['PC2'].agg(['count', 'mean', 'min', 'max']))

c2 = df_combined[df_combined['Cluster_Label'] == 2]
print("\n--- CLUSTER 2 (Azul) ---")
print(c2.groupby('previous_pregnancies')['PC2'].agg(['count', 'mean', 'min', 'max']))
