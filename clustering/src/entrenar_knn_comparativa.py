import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import matplotlib

matplotlib.use('Agg')

# 1. Cargar datos etiquetados
df = pd.read_csv("data/v2/labeled_patients.csv")
y = df['Cluster_Label']

# 2. Definir los dos conjuntos de variables de entrada (X)
X_12 = df[[f"PC{i+1}" for i in range(12)]]
X_5 = df[[f"PC{i+1}" for i in range(5)]]

# 3. Dividir en Entrenamiento (80%) y Prueba (20%) con stratify para mantener la proporción de clases
X_train_12, X_test_12, y_train, y_test = train_test_split(X_12, y, test_size=0.2, random_state=42, stratify=y)
X_train_5, X_test_5, _, _ = train_test_split(X_5, y, test_size=0.2, random_state=42, stratify=y)

print(f"Tamaño de Entrenamiento: {X_train_12.shape[0]} pacientes")
print(f"Tamaño de Prueba: {X_test_12.shape[0]} pacientes")
print("-" * 50)

# 4. Configurar búsqueda en cuadrícula (GridSearchCV) para encontrar el mejor número de vecinos (K)
param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 15]}

# --- Entrenar Modelo de 12 Componentes ---
print("Entrenando Modelo 1 (12 Componentes PCA)...")
knn_12 = KNeighborsClassifier()
grid_12 = GridSearchCV(knn_12, param_grid, cv=5, scoring='accuracy')
grid_12.fit(X_train_12, y_train)

best_knn_12 = grid_12.best_estimator_
y_pred_12 = best_knn_12.predict(X_test_12)
acc_12 = accuracy_score(y_test, y_pred_12)
print(f"Mejor K para 12 Componentes: {grid_12.best_params_['n_neighbors']}")
print(f"Accuracy de Prueba (12 PC): {acc_12 * 100:.2f}%\n")

# --- Entrenar Modelo de 5 Componentes ---
print("Entrenando Modelo 2 (5 Componentes PCA)...")
knn_5 = KNeighborsClassifier()
grid_5 = GridSearchCV(knn_5, param_grid, cv=5, scoring='accuracy')
grid_5.fit(X_train_5, y_train)

best_knn_5 = grid_5.best_estimator_
y_pred_5 = best_knn_5.predict(X_test_5)
acc_5 = accuracy_score(y_test, y_pred_5)
print(f"Mejor K para 5 Componentes: {grid_5.best_params_['n_neighbors']}")
print(f"Accuracy de Prueba (5 PC): {acc_5 * 100:.2f}%\n")

print("-" * 50)
print(f"PERDIDA DE RENDIMIENTO AL USAR 5 COMPONENTES: {(acc_12 - acc_5)*100:.2f}%")
print("-" * 50)

# 5. Generar Matrices de Confusión
cm_12 = confusion_matrix(y_test, y_pred_12)
cm_5 = confusion_matrix(y_test, y_pred_5)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_12, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title(f'Matriz de Confusión (12 Componentes)\nAccuracy: {acc_12*100:.1f}%')
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Clúster Real')

sns.heatmap(cm_5, annot=True, fmt='d', cmap='Reds', ax=axes[1], cbar=False)
axes[1].set_title(f'Matriz de Confusión (5 Componentes)\nAccuracy: {acc_5*100:.1f}%')
axes[1].set_xlabel('Predicción')
axes[1].set_ylabel('Clúster Real')

plt.tight_layout()
output_dir = r"C:\Users\MAXIMILIANO\.gemini\antigravity\brain\4c77bb59-640a-4763-89f1-8d8a7a3f7184\scratch"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "knn_confusion_matrices.png")
plt.savefig(plot_path, dpi=150)
print(f"Matrices de Confusión guardadas en: {plot_path}")

# Guardar reportes
with open(os.path.join(output_dir, "reporte_knn.txt"), "w") as f:
    f.write(f"--- MODELO 12 PCA ---\n")
    f.write(classification_report(y_test, y_pred_12))
    f.write(f"\n--- MODELO 5 PCA ---\n")
    f.write(classification_report(y_test, y_pred_5))
