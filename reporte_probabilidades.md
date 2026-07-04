# Reporte de Pertenencia Porcentual a Clústeres (Soft Classification)

Al utilizar el método `predict_proba()` de nuestro modelo de producción (KNN), obtenemos un **porcentaje de pertenencia**, lo cual es vital para pacientes en la zona gris (frontera).

A continuación se analizan 3 casos de prueba:

### CASO 1: Paciente Joven Sana (Esperado: Clúster 0 o 2)
**Perfil Clínico**: Edad 19, IMC 18.4, Presión 113.0/58.0, Embarazos 1.0

#### Veredicto Probabilístico KNN (Basado en la proporción de los 7 vecinos más cercanos)
- **C0: Sana Primeriza**: 53.3%
- **C1: Riesgo Hipertensivo**: 0.0%
- **C2: Sana Multipara**: 46.7%
- **C3: Riesgo Metabolico**: 0.0%

---
### CASO 2: Paciente con Sobrepeso y Presión Limítrofe (Frontera)
**Perfil Clínico**: Edad 27, IMC 28.7, Presión 103.0/73.0, Embarazos 0.0

#### Veredicto Probabilístico KNN (Basado en la proporción de los 7 vecinos más cercanos)
- **C0: Sana Primeriza**: 100.0%
- **C1: Riesgo Hipertensivo**: 0.0%
- **C2: Sana Multipara**: 0.0%
- **C3: Riesgo Metabolico**: 0.0%

---
### CASO 3: Paciente Hipertensa Aguda (Esperado: Clúster 1)
**Perfil Clínico**: Edad 34, IMC 21.9, Presión 163.0/107.0, Embarazos 2.0

#### Veredicto Probabilístico KNN (Basado en la proporción de los 7 vecinos más cercanos)
- **C0: Sana Primeriza**: 0.0%
- **C1: Riesgo Hipertensivo**: 100.0%
- **C2: Sana Multipara**: 0.0%
- **C3: Riesgo Metabolico**: 0.0%

---
