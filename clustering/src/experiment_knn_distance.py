import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/v2/labeled_patients.csv")
X_5 = df[[f"PC{i+1}" for i in range(5)]]
y = df['Cluster_Label']

X_train, X_test, y_train, y_test = train_test_split(X_5, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Mejor configuracion global: {grid.best_params_}")
acc = grid.best_estimator_.score(X_test, y_test)
print(f"Accuracy en Prueba Final con mejor configuracion: {acc*100:.2f}%\n")

results = pd.DataFrame(grid.cv_results_)
for metric in ['euclidean', 'manhattan', 'cosine']:
    best_for_metric = results[results['param_metric'] == metric].sort_values('rank_test_score').iloc[0]
    print(f"Mejor resultado usando {metric.upper()}: K={best_for_metric['param_n_neighbors']} -> Score de Entrenamiento CV: {best_for_metric['mean_test_score']*100:.2f}%")
