import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

def evaluate_method(y_true, y_pred, method_name):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"--- {method_name} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

def main():
    # 1. Cargar datos
    df_features = pd.read_csv('clustering/data/20k/clustering_feature_view_completo.csv')
    df_gt = pd.read_csv('clustering/data/20k/ground_truth.csv')
    
    # Unir
    df = pd.merge(df_features, df_gt, on='patient_id')
    
    # Variables de interes
    target = 'es_outlier'
    cluster_col = 'cluster_verdadero'
    
    y_true = df[target]
    
    # 3. a) Regla manual:
    # systolic > 1.15 * media_cluster o bmi_initial > 1.20 * media_cluster
    df['pred_manual'] = 0
    for cluster_id in df[cluster_col].unique():
        mask = df[cluster_col] == cluster_id
        mean_systolic = df.loc[mask, 'systolic'].mean()
        mean_bmi = df.loc[mask, 'bmi_initial'].mean()
        
        cond_systolic = df.loc[mask, 'systolic'] > (1.15 * mean_systolic)
        cond_bmi = df.loc[mask, 'bmi_initial'] > (1.20 * mean_bmi)
        
        df.loc[mask, 'pred_manual'] = (cond_systolic | cond_bmi).astype(int)
        
    # 3. b) Z-Scores
    # |Z-Score| > 3.0 respecto al cluster para systolic, bmi_initial, o weight_kg
    df['pred_zscore'] = 0
    vars_zscore = ['systolic', 'bmi_initial', 'weight_kg']
    for cluster_id in df[cluster_col].unique():
        mask = df[cluster_col] == cluster_id
        
        # std ddof=1 is default in pandas, but we can use ddof=0 or just default.
        std_vals = df.loc[mask, vars_zscore].std()
        std_vals = std_vals.replace(0, 1e-9) # Evitar division por cero
        
        z_scores = (df.loc[mask, vars_zscore] - df.loc[mask, vars_zscore].mean()) / std_vals
        is_anomaly = (np.abs(z_scores) > 3.0).any(axis=1)
        
        df.loc[mask, 'pred_zscore'] = is_anomaly.astype(int)
        
    # 3. c) Mahalanobis
    # Usando las variables continuas numéricas
    cont_vars = ['age_years', 'bmi_initial', 'gestational_week', 'height_cm', 
                 'initial_weight', 'weight_kg', 'weight_gain', 'systolic', 
                 'diastolic', 'mean_arterial_pressure']
    
    df['pred_mahalanobis_95'] = 0
    df['pred_mahalanobis_99'] = 0
    
    threshold_95 = chi2.ppf(0.95, df=len(cont_vars))
    threshold_99 = chi2.ppf(0.99, df=len(cont_vars))
    
    for cluster_id in df[cluster_col].unique():
        mask = df[cluster_col] == cluster_id
        X = df.loc[mask, cont_vars].values
        
        if len(X) > len(cont_vars):
            centroid = np.mean(X, axis=0)
            cov_matrix = np.cov(X, rowvar=False)
            try:
                # Agregar un pequeno valor a la diagonal para evitar singularidades
                cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
                inv_cov = np.linalg.inv(cov_matrix)
                
                # Calcular distancias
                distances = [mahalanobis(x, centroid, inv_cov) for x in X]
                squared_distances = np.square(distances)
                
                df.loc[mask, 'pred_mahalanobis_95'] = (squared_distances > threshold_95).astype(int)
                df.loc[mask, 'pred_mahalanobis_99'] = (squared_distances > threshold_99).astype(int)
            except np.linalg.LinAlgError:
                pass
                
    # 4. Medir Precision, Recall, F1
    print("=== RESULTADOS DE DETECCION DE OUTLIERS ===")
    evaluate_method(y_true, df['pred_manual'], "Regla Manual")
    evaluate_method(y_true, df['pred_zscore'], "Z-Scores")
    evaluate_method(y_true, df['pred_mahalanobis_95'], "Mahalanobis (Chi2 0.95)")
    evaluate_method(y_true, df['pred_mahalanobis_99'], "Mahalanobis (Chi2 0.99)")

if __name__ == "__main__":
    main()
