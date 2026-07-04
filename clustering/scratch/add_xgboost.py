import json

notebook_path = r'c:\Universidad\mineria\clustering\notebooks\cluster.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find where KNN evaluation is
knn_index = -1
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if cell['cell_type'] == 'code' and 'classification_report' in src and 'KNeighborsClassifier' in src:
        knn_index = i
        break
    elif cell['cell_type'] == 'code' and 'classification_report' in src and 'C0:' in src:
        knn_index = i

if knn_index == -1:
    knn_index = 14

xgboost_code = [
    "# 5. Entrenar Clasificador XGBoost Alternativo\n",
    "import xgboost as xgb\n",
    "import joblib\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "\n",
    "# XGBoost requiere que las etiquetas comiencen en 0 y sean enteras.\n",
    "# Nuestras etiquetas de K-Means ya son 0, 1, 2, 3.\n",
    "xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)\n",
    "xgb_model.fit(X_train, y_train)\n",
    "\n",
    "# Evaluar XGBoost\n",
    "y_pred_xgb = xgb_model.predict(X_test)\n",
    "print(\"=== REPORTE DE CLASIFICACIÓN XGBOOST ===\")\n",
    "print(f\"Accuracy en Datos de Prueba: {accuracy_score(y_test, y_pred_xgb):.2%}\\n\")\n",
    "print(classification_report(y_test, y_pred_xgb, target_names=[\n",
    "    \"C0: Primigestas Sanas\",\n",
    "    \"C1: Riesgo Hipertensivo\",\n",
    "    \"C2: Multiparas Sanas\",\n",
    "    \"C3: Riesgo Metabólico\"\n",
    "]))\n",
    "\n",
    "# Guardar el modelo alternativo\n",
    "joblib.dump(xgb_model, '../models/xgboost_model.pkl')\n",
    "print(\"Modelo XGBoost guardado en '../models/xgboost_model.pkl'\")"
]

xgboost_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": xgboost_code
}

# Insert cell after KNN
nb['cells'].insert(knn_index + 1, xgboost_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('XGBoost cell inserted at index', knn_index + 1)
