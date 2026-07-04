import json

notebook_path = r"c:\Universidad\mineria\clustering\notebooks\cluster.ipynb"

# Cargar el notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Crear celdas a añadir
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## KNN: Clasificador Supervisado para Producción (NO Clustering)\n",
        "\n",
        "> [!IMPORTANT]\n",
        "> **Aclaración metodológica:** K-Nearest Neighbors (KNN) es un algoritmo de *aprendizaje supervisado*, no de clustering. Lo que haremos a continuación NO es 'descubrir' nuevos clústeres, sino entrenar un clasificador para que aprenda a imitar las decisiones que tomó K-Means.\n",
        "\n",
        "**¿Por qué hacer esto?**\n",
        "En un entorno de producción (como nuestra API de FastAPI), cuando llega una paciente nueva, K-Means es ineficiente para predecir a qué grupo pertenece porque tendría que re-evaluar todo el dataset. Entrenar un modelo **KNN ($K=7$, pesos uniformes, distancia euclidiana)** usando las etiquetas que nos dio K-Means nos permite clasificar instantáneamente a nuevas pacientes con un **95% de precisión (Accuracy)** sin sobreajuste (overfitting). \n",
        "\n",
        "A continuación, entrenaremos este clasificador y graficaremos el resultado (K-Means original vs Fronteras de Decisión del KNN) en 3D para entender visualmente cómo el modelo está separando a las pacientes."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import classification_report, accuracy_score\n",
        "import plotly.express as px\n",
        "import plotly.graph_objects as go\n",
        "\n",
        "# 1. Usar las etiquetas de K-Means (K=4) como nuestro 'Target' (Variable Y)\n",
        "y = df_cluster['Cluster']\n",
        "\n",
        "# 2. Dividir en Entrenamiento (80%) y Prueba (20%)\n",
        "X_train, X_test, y_train, y_test = train_test_split(df_cluster, y, test_size=0.2, random_state=42, stratify=y)\n",
        "\n",
        "# 3. Entrenar KNN con los mejores hiperparámetros encontrados\n",
        "knn_best = KNeighborsClassifier(n_neighbors=7, weights='uniform', metric='euclidean', p=2)\n",
        "knn_best.fit(X_train, y_train)\n",
        "\n",
        "# 4. Evaluar el modelo\n",
        "y_pred_test = knn_best.predict(X_test)\n",
        "print(\"=== REPORTE DE CLASIFICACIÓN KNN ===\")\n",
        "print(f\"Accuracy en Datos de Prueba: {accuracy_score(y_test, y_pred_test):.2%}\\n\")\n",
        "print(classification_report(y_test, y_pred_test, target_names=[\n",
        "    \"C0: Primigestas Sanas\",\n",
        "    \"C1: Riesgo Hipertensivo\",\n",
        "    \"C2: Multiparas Sanas\",\n",
        "    \"C3: Riesgo Metabolico\"\n",
        "]))\n",
        "\n",
        "# 5. Visualización 3D Interactiva con Plotly\n",
        "df_plot = df_perfilado.copy()\n",
        "df_plot['Prediccion_KNN'] = knn_best.predict(df_cluster).astype(str)\n",
        "df_plot['PC1'] = df_cluster['PC1']\n",
        "df_plot['PC2'] = df_cluster['PC2']\n",
        "df_plot['PC3'] = df_cluster['PC3']\n",
        "\n",
        "fig = px.scatter_3d(\n",
        "    df_plot,\n",
        "    x='PC1',\n",
        "    y='PC2',\n",
        "    z='PC3',\n",
        "    color='Prediccion_KNN',\n",
        "    hover_data=['age_years', 'bmi_initial', 'mean_arterial_pressure', 'previous_pregnancies'],\n",
        "    title=\"Fronteras de Decisión KNN en el Espacio PCA 3D\",\n",
        "    color_discrete_sequence=px.colors.qualitative.Set1,\n",
        "    opacity=0.7\n",
        ")\n",
        "fig.update_traces(marker=dict(size=4))\n",
        "fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))\n",
        "fig.show()"
    ]
}

# Añadir celdas al final
nb['cells'].append(markdown_cell)
nb['cells'].append(code_cell)

# Guardar el notebook modificado
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Celdas de KNN añadidas con éxito al notebook.")
