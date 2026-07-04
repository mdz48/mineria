import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _check_restricciones(df: pd.DataFrame, es_outlier: pd.Series) -> list[str]:
    errores = []
    no_out = df[~es_outlier.astype(bool)]

    pa_bad = no_out[no_out["presion_sistolica"] <= no_out["presion_diastolica"] + 14]
    if len(pa_bad):
        errores.append(f"PA incoherente en {len(pa_bad)} filas no-outlier")

    paridad_bad = no_out[no_out["num_partos_previos"] > no_out["num_embarazos_previos"]]
    if len(paridad_bad):
        errores.append(f"Paridad inválida en {len(paridad_bad)} filas")

    trim_bad = no_out[
        no_out["trimestre"] != no_out["semanas_gestacion"].apply(
            lambda s: 1 if s <= 13 else (2 if s <= 27 else 3)
        )
    ]
    if len(trim_bad):
        errores.append(f"Trimestre inconsistente en {len(trim_bad)} filas")

    return errores


def generar_reporte(
    df_completo: pd.DataFrame,
    df_sintetico: pd.DataFrame,
    meta: pd.DataFrame,
    log_miss: pd.DataFrame,
    config: dict,
) -> str:
    n = len(df_completo)
    lines = [
        "# Reporte de calidad — dataset sintético embarazo",
        "",
        f"**Pacientes generados:** {n}",
        f"**Semilla:** {config['semilla']}",
        "",
        "## Distribución de clusters (verdad de referencia)",
        "",
    ]

    dist = df_completo["cluster_verdadero"].value_counts(normalize=True).sort_index()
    for k, pct in dist.items():
        esperado = config["proporciones_cluster"][k] * 100
        lines.append(f"- Cluster {k}: {pct*100:.1f}% (esperado ~{esperado:.1f}%)")

    lines.extend(["", "## Outliers", ""])
    n_out = int(meta["es_outlier"].sum())
    lines.append(f"- Filas con outlier: {n_out} ({100*n_out/n:.1f}%)")
    tipo_counts = meta[meta["es_outlier"] == 1]["tipo_outlier"].value_counts()
    for t, c in tipo_counts.items():
        lines.append(f"  - {t}: {c}")

    lines.extend(["", "## Missingness", ""])
    if len(log_miss):
        total_celdas = n * len([c for c in df_sintetico.columns if c not in ("paciente_id", "cluster_verdadero")])
        lines.append(f"- Celdas con nulo inyectado: {len(log_miss)} ({100*len(log_miss)/total_celdas:.1f}% del total)")
        for mec, cnt in log_miss["mecanismo"].value_counts().items():
            lines.append(f"  - {mec}: {cnt}")
        lines.append("")
        lines.append("### Tasas por variable")
        for var, cnt in log_miss["variable"].value_counts().head(10).items():
            lines.append(f"- {var}: {cnt}")

    lines.extend(["", "## Correlaciones (datos completos, sin outliers)", ""])
    clean = df_completo.merge(meta[["paciente_id", "es_outlier"]], on="paciente_id")
    clean = clean[clean["es_outlier"] == 0]
    pairs = [
        ("imc_pregestacional", "ganancia_peso_kg"),
        ("semanas_gestacion", "ganancia_peso_kg"),
        ("peso_kg", "imc_pregestacional"),
        ("presion_sistolica", "presion_diastolica"),
    ]
    for a, b in pairs:
        r = clean[[a, b]].corr().iloc[0, 1]
        lines.append(f"- {a} ↔ {b}: r = {r:.3f}")

    lines.extend(["", "## Métricas de clustering (KMeans k=6, datos limpios)", ""])
    feature_cols = [
        "edad_anios", "imc_pregestacional", "semanas_gestacion",
        "num_embarazos_previos", "num_partos_previos", "embarazo_multiple",
        "tabaquismo_activo", "diabetes_previa", "hipertension_cronica",
        "peso_kg", "talla_cm", "ganancia_peso_kg",
        "presion_sistolica", "presion_diastolica", "frecuencia_cardiaca",
        "suplemento_acido_folico", "suplemento_hierro",
    ]
    X = clean[feature_cols].copy()
    X["nivel_educacion"] = clean["nivel_educacion"].map({"primaria": 0, "secundaria": 1, "superior": 2})
    X["area_residencia"] = clean["area_residencia"].map({"rural": 0, "urbana": 1})
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(X_scaled)
    y_true = clean["cluster_verdadero"].values
    ari = adjusted_rand_score(y_true, labels)
    sil = silhouette_score(X_scaled, y_true)
    lines.append(f"- ARI (KMeans vs cluster_verdadero): {ari:.3f}")
    lines.append(f"- Silhouette (cluster_verdadero como labels): {sil:.3f}")

    lines.extend(["", "## Validación de restricciones", ""])
    errores = _check_restricciones(df_completo, meta["es_outlier"])
    if errores:
        for e in errores:
            lines.append(f"- ⚠ {e}")
    else:
        lines.append("- ✓ Restricciones duras cumplidas (excl. outliers marcados)")

    lines.extend(["", "## Perfiles medios por cluster", ""])
    for k in sorted(clean["cluster_verdadero"].unique()):
        sub = clean[clean["cluster_verdadero"] == k]
        lines.append(
            f"- C{k}: edad={sub['edad_anios'].mean():.1f}, IMC={sub['imc_pregestacional'].mean():.1f}, "
            f"PA={sub['presion_sistolica'].mean():.0f}/{sub['presion_diastolica'].mean():.0f}, "
            f"ganancia={sub['ganancia_peso_kg'].mean():.1f} kg"
        )

    return "\n".join(lines) + "\n"
