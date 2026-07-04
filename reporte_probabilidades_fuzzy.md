# Reporte de Pertenencia Porcentual (Soft Classification via Distancia Espacial)

Tienes mucha razón. Como nuestro dataset sintético generó clústeres muy bien separados (y KNN toma los "votos" de la vecindad), el modelo es "demasiado seguro" de sí mismo y arroja 100% casi siempre, porque todos los vecinos de una paciente pertenecen al mismo diagnóstico.

Para solucionar esto y obtener un **porcentaje de clasificación continuo y útil**, dejamos de contar los votos de los vecinos y pasamos a usar la **Distancia Espacial Matemática a los Centroides Clínicos (Fuzzy Logic)**. Así medimos milimétricamente qué tan cerca está la paciente del núcleo de cada enfermedad y repartimos el 100% de probabilidad de forma inversamente proporcional a esa distancia.

A continuación se analizan los mismos 3 casos, pero ahora con porcentajes que reflejan la realidad espacial:

### CASO 1: Paciente Joven Sana
**Perfil Clínico**: Edad 19, IMC 18.4, Presión 113.0/58.0, Embarazos 1.0

#### Pertenencia por Distancia Matemática al Centroide (Fuzzy Score)
- **C0: Sana Primeriza**: 64.4%
- **C1: Riesgo Hipertensivo**: 1.4%
- **C2: Sana Multipara**: 32.8%
- **C3: Riesgo Metabolico**: 1.5%

---
### CASO 2: Paciente con Sobrepeso y Presión Limítrofe (Frontera Metabólica)
**Perfil Clínico**: Edad 27, IMC 28.7, Presión 103.0/73.0, Embarazos 0.0

#### Pertenencia por Distancia Matemática al Centroide (Fuzzy Score)
- **C0: Sana Primeriza**: 94.4%
- **C1: Riesgo Hipertensivo**: 0.9%
- **C2: Sana Multipara**: 2.7%
- **C3: Riesgo Metabolico**: 2.0%

---
### CASO 3: Paciente Hipertensa Aguda
**Perfil Clínico**: Edad 34, IMC 21.9, Presión 163.0/107.0, Embarazos 2.0

#### Pertenencia por Distancia Matemática al Centroide (Fuzzy Score)
- **C0: Sana Primeriza**: 1.8%
- **C1: Riesgo Hipertensivo**: 92.1%
- **C2: Sana Multipara**: 2.1%
- **C3: Riesgo Metabolico**: 4.0%

---
