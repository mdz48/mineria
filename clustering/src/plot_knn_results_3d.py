import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import os

# 1. Cargar datos etiquetados y datos originales (para la metadata del hover)
df_labeled = pd.read_csv("data/v2/labeled_patients.csv")
df_full = pd.read_csv("data/v2/clustering_feature_view.csv")

X_5 = df_labeled[[f"PC{i+1}" for i in range(5)]]
y = df_labeled['Cluster_Label']

# 2. Dividir (igual que en nuestro experimento)
X_train, X_test, y_train, y_test = train_test_split(X_5, y, test_size=0.2, random_state=42, stratify=y)

# 3. Entrenar el modelo absoluto ganador
print("Entrenando modelo KNN (K=15, métrica=Manhattan)...")
knn = KNeighborsClassifier(n_neighbors=15, metric='manhattan')
knn.fit(X_train, y_train)

# 4. Predecir a las 400 pacientes de prueba
y_pred = knn.predict(X_test)

# 5. Armar un dataframe solo con las pacientes de prueba para graficarlas
df_test = X_test.copy()
df_test['Cluster Real'] = 'Clúster ' + y_test.astype(str)
df_test['Predicción del Modelo'] = 'Clúster ' + pd.Series(y_pred).astype(str).values

# Determinar si el modelo acertó o se equivocó
df_test['Estado de Predicción'] = ['✅ Acierto (Correcto)' if real == pred else '❌ ERROR' for real, pred in zip(y_test, y_pred)]

# Agregar datos originales al hover usando el índice
df_test['Edad'] = df_full.loc[df_test.index, 'age_years']
df_test['IMC'] = df_full.loc[df_test.index, 'bmi_initial'].round(2)

# 6. Generar el gráfico interactivo 3D
print("Generando HTML con Plotly...")
fig = px.scatter_3d(
    df_test,
    x='PC1',
    y='PC2',
    z='PC3',
    color='Cluster Real',
    symbol='Estado de Predicción',
    hover_data=['Predicción del Modelo', 'Edad', 'IMC'],
    title="Prueba Final del Modelo KNN (99% Exactitud) - Identificación de Aciertos y Errores",
    color_discrete_sequence=px.colors.qualitative.Set1,
    symbol_sequence=['circle', 'x'] # Círculos para aciertos, X para errores
)

# Ajustar tamaño de los puntos
fig.update_traces(marker=dict(size=7, opacity=0.8))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

output_path = os.path.abspath("knn_resultados_3d.html")
fig.write_html(output_path)
print(f"Gráfico HTML interactivo generado en: {output_path}")
