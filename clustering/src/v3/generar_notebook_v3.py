import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Análisis de Clustering V3: Incorporación de Historial Obstétrico Adverso\n",
    "\n",
    "En esta versión V3, se han añadido las variables `preterm_count` (partos prematuros previos) y `emergency_cesarean_count` (cesáreas de emergencia), manteniendo una regla física estricta para nulíparas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.decomposition import PCA\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# 1. Cargar la vista de clustering v3\n",
    "df_full = pd.read_csv('../data/v3/clustering_feature_view.csv')\n",
    "df_features = df_full.drop(columns=['patient_id'])\n",
    "\n",
    "# 2. Pipeline de preprocesamiento\n",
    "num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()\n",
    "cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()\n",
    "\n",
    "preprocessor = ColumnTransformer(transformers=[\n",
    "    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),\n",
    "    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))]), cat_cols)\n",
    "])\n",
    "\n",
    "X_processed = preprocessor.fit_transform(df_features)\n",
    "print(f\"Dimensiones post-preprocesamiento: {X_processed.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Análisis de Componentes Principales (PCA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "pca = PCA(random_state=42)\n",
    "pca.fit(X_processed)\n",
    "varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)\n",
    "n_components_80 = np.argmax(varianza_acumulada >= 0.80) + 1\n",
    "print(f\"Componentes para retener el 80% de varianza: {n_components_80}\")\n",
    "\n",
    "pca_optimo = PCA(n_components=n_components_80, random_state=42)\n",
    "X_cluster = pca_optimo.fit_transform(X_processed)\n",
    "df_cluster = pd.DataFrame(X_cluster, columns=[f\"PC{i+1}\" for i in range(n_components_80)])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Determinación del Número de Clústeres (K)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_score\n",
    "\n",
    "sse = []\n",
    "sil_scores = []\n",
    "K_range = range(2, 8)\n",
    "for k in K_range:\n",
    "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
    "    labels = km.fit_predict(X_cluster)\n",
    "    sse.append(km.inertia_)\n",
    "    sil_scores.append(silhouette_score(X_cluster, labels))\n",
    "    print(f\"K={k} -> Silhouette Score: {sil_scores[-1]:.4f}\")\n",
    "\n",
    "plt.figure(figsize=(10,4))\n",
    "plt.subplot(1,2,1)\n",
    "plt.plot(K_range, sse, marker='o')\n",
    "plt.title('Método del Codo (Elbow)')\n",
    "plt.subplot(1,2,2)\n",
    "plt.plot(K_range, sil_scores, marker='o', color='orange')\n",
    "plt.title('Silhouette Score')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Modelo Final (K=4) y Perfilado Clínico"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "km_final = KMeans(n_clusters=4, random_state=42, n_init=10)\n",
    "df_full['Cluster'] = km_final.fit_predict(X_cluster)\n",
    "\n",
    "cols_clinicas = [\n",
    "    'age_years', 'bmi_initial', 'mean_arterial_pressure', 'nulliparous',\n",
    "    'previous_pregnancies', 'preterm_count', 'emergency_cesarean_count', \n",
    "    'previous_preeclampsia', 'diabetes', 'chronic_hypertension'\n",
    "]\n",
    "perfilado = df_full.groupby('Cluster')[cols_clinicas].mean().round(3)\n",
    "print(\"PERFIL CLÍNICO V3:\")\n",
    "display(perfilado)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## DBSCAN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "from sklearn.cluster import DBSCAN\n",
    "from sklearn.neighbors import NearestNeighbors\n",
    "\n",
    "dbscan = DBSCAN(eps=3.0, min_samples=10)\n",
    "db_labels = dbscan.fit_predict(X_cluster)\n",
    "\n",
    "n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)\n",
    "n_noise = (db_labels == -1).sum()\n",
    "print(f\"DBSCAN descubrió {n_clusters} clústeres. Ruido: {n_noise} pacientes.\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

os.makedirs(r'c:\Universidad\mineria\clustering\notebooks', exist_ok=True)
with open(r'c:\Universidad\mineria\clustering\notebooks\cluster_v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)
