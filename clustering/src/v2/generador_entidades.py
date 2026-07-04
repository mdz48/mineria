import numpy as np
import pandas as pd
from .clusters_v2 import CLUSTER_PROFILES_V2

def _clip(val: float, lo: float, hi: float) -> float:
    return float(np.clip(val, lo, hi))

def generar_entidades(n: int, clusters: np.ndarray, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = []
    medical_records = []
    patient_diaries = []
    
    for i, c in enumerate(clusters):
        patient_id = i + 1
        
        if c == 5: # C5 Frontera (mezcla)
            base_c = int(rng.choice([1, 2, 4]))
        else:
            base_c = int(c)
            
        p = CLUSTER_PROFILES_V2[base_c]
        
        # --- PATIENT ---
        age_years = _clip(rng.normal(*p["age"]), 15, 49)
        bmi_initial = _clip(rng.normal(*p["bmi"]), 14, 55)
        height_cm = _clip(rng.normal(160, 6), 140, 185)
        initial_weight = bmi_initial * (height_cm / 100)**2
        
        educ_choice = rng.choice(["superior", "secundaria", "primaria"], p=p["education"])
        residence = "rural" if rng.random() < p["rural"] else "urbana"
        marital_status = "married" if rng.random() < 0.6 else "single"
        
        patients.append({
            "patient_id": patient_id,
            "age_at_registration": round(age_years),
            "bmi_initial": round(bmi_initial, 2),
            "height_cm": round(height_cm, 1),
            "initial_weight": round(initial_weight, 1),
            "education_level": educ_choice,
            "residence": residence,
            "marital_status": marital_status
        })
        
        # --- MEDICAL RECORD ---
        medical_record_id = patient_id * 10
        
        diabetes = int(rng.random() < p["diabetes"])
        if age_years > 35 and rng.random() < 0.05: diabetes = 1
        
        chronic_hypertension = int(rng.random() < p["chronic_hypertension"])
        previous_preeclampsia = int(rng.random() < p["previous_preeclampsia"])
        
        nulliparous_prob = p["nulliparous"]
        if previous_preeclampsia: nulliparous_prob = 0.0 # R10
        
        nulliparous = int(rng.random() < nulliparous_prob)
        
        if nulliparous:
            parity_count = 0
        else:
            parity_count = int(rng.integers(1, 6))
            
        delivery_count = int(rng.integers(0, parity_count + 1)) if parity_count > 0 else 0
        miscarriage_count = parity_count - delivery_count
        cesarean_count = int(rng.integers(0, delivery_count + 1)) if delivery_count > 0 else 0
        
        medical_records.append({
            "medical_record_id": medical_record_id,
            "patient_id": patient_id,
            "diabetes": diabetes,
            "chronic_hypertension": chronic_hypertension,
            "previous_preeclampsia": previous_preeclampsia,
            "family_history_hypertension": int(rng.random() < p["family_history_hypertension"]),
            "family_history_heart_disease": int(rng.random() < p["family_history_heart_disease"]),
            "chronic_kidney_disease": int(rng.random() < p["chronic_kidney_disease"]),
            "multiple_pregnancy": int(rng.random() < p["multiple_pregnancy"]),
            "active_smoking": int(rng.random() < p["active_smoking"]),
            "parity_count": parity_count,
            "delivery_count": delivery_count,
            "miscarriage_count": miscarriage_count,
            "cesarean_count": cesarean_count
        })
        
        # --- PATIENT DIARY ---
        diary_id = patient_id * 100
        gestational_week = _clip(rng.normal(*p["gestational_week"]), 14, 40)
        
        if gestational_week <= 13: trimester = 1
        elif gestational_week <= 27: trimester = 2
        else: trimester = 3
        
        expected_gain = (gestational_week - 12) * 0.4 if gestational_week > 12 else 1.0
        expected_gain = max(0.5, expected_gain)
        weight_gain = expected_gain * p["weight_gain_coef"] + rng.normal(0, 1.5)
        weight_gain = _clip(weight_gain, -2, 25)
        
        if c == 5: weight_gain += rng.normal(0, 3.0)
            
        weight_kg = initial_weight + weight_gain
        weekly_weight_gain_rate = weight_gain / gestational_week if gestational_week > 0 else 0
        
        pa_sys = _clip(rng.normal(*p["systolic"]), 80, 180)
        pa_dia = _clip(rng.normal(*p["diastolic"]), 45, 120)
        
        if c == 5:
            pa_sys += rng.normal(0, 10)
            pa_dia += rng.normal(0, 8)
            
        if chronic_hypertension:
            pa_sys = max(pa_sys, 130 + rng.uniform(0, 20))
            pa_dia = max(pa_dia, 85 + rng.uniform(0, 15))
            
        if pa_sys <= pa_dia + 15:
            pa_sys = pa_dia + rng.uniform(16, 30)
            
        mean_arterial_pressure = (pa_sys + 2 * pa_dia) / 3
        
        patient_diaries.append({
            "diary_id": diary_id,
            "medical_record_id": medical_record_id,
            "is_clustering_snapshot": 1,
            "gestational_week": round(gestational_week, 1),
            "gestational_trimester": trimester,
            "weight_kg": round(weight_kg, 1),
            "weight_gain": round(weight_gain, 1),
            "weekly_weight_gain_rate": round(weekly_weight_gain_rate, 2),
            "systolic": round(pa_sys),
            "diastolic": round(pa_dia),
            "mean_arterial_pressure": round(mean_arterial_pressure, 1)
        })
        
    return pd.DataFrame(patients), pd.DataFrame(medical_records), pd.DataFrame(patient_diaries)
