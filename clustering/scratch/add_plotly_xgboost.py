import json

notebook_path = r'c:\Universidad\mineria\clustering\notebooks\cluster.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find XGBoost cell
xgb_index = -1
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'xgb.XGBClassifier' in src:
        xgb_index = i
        break

if xgb_index != -1:
    plotly_code = [
        "\n",
        "# 4. Visualización 3D Interactiva con Plotly para XGBoost\n",
        "import plotly.express as px\n",
        "\n",
        "df_plot_xgb = df_perfilado.copy()\n",
        "df_plot_xgb['Prediccion_XGB'] = xgb_model.predict(df_cluster.drop(columns=['Cluster'])).astype(str)\n",
        "df_plot_xgb['PC1'] = df_cluster['PC1']\n",
        "df_plot_xgb['PC2'] = df_cluster['PC2']\n",
        "df_plot_xgb['PC3'] = df_cluster['PC3']\n",
        "\n",
        "fig_xgb = px.scatter_3d(\n",
        "    df_plot_xgb,\n",
        "    x='PC1',\n",
        "    y='PC2',\n",
        "    z='PC3',\n",
        "    color='Prediccion_XGB',\n",
        "    hover_data=['age_years', 'bmi_initial', 'mean_arterial_pressure', 'previous_pregnancies'],\n",
        "    title=\"Fronteras de Decisión XGBoost en el Espacio PCA 3D\",\n",
        "    color_discrete_sequence=px.colors.qualitative.Set1,\n",
        "    opacity=0.7\n",
        ")\n",
        "fig_xgb.update_traces(marker=dict(size=4))\n",
        "fig_xgb.update_layout(margin=dict(l=0, r=0, b=0, t=40))\n",
        "fig_xgb.show()\n"
    ]
    nb['cells'][xgb_index]['source'].extend(plotly_code)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('XGBoost plotting code appended successfully.')
else:
    print('XGBoost cell not found!')
