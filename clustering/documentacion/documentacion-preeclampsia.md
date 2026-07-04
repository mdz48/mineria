Aquí tienes la información estructurada en un formato claro y estandarizado, diseñado específicamente para que otro agente de IA pueda procesar las características (features) y etiquetas (labels) al construir y agrupar el dataset sintético.

---

## Especificaciones para la Arquitectura del Dataset: Preeclampsia

### 1. Definición del Problema (Target)

* El objetivo del modelo es predecir o clasificar la aparición de preeclampsia, definida como un trastorno multisistémico del embarazo.


* La condición aparece después de las 20 semanas de gestación.


* La incidencia global base (probabilidad a priori) es de 1 en cada 10 gestaciones.



### 2. Variables Objetivo / Etiquetas (Labels)

Para que un registro en el dataset sea etiquetado como "Positivo" para preeclampsia, debe presentar hipertensión acompañada de al menos una de las siguientes condiciones adversas:

| Marcador Clínico | Umbral de Diagnóstico (Condición de Activación) |
| --- | --- |
| **Presión Arterial Sistólica** | Mayor o igual a 140 mmHg 

 |
| **Presión Arterial Diastólica** | Mayor o igual a 90 mmHg 
 |

### 3. Variables Predictoras / Factores de Riesgo (Features)

| Característica (Feature) | Tipo de Dato | Riesgo Relativo (Impacto) | Contexto de Extracción |
| --- | --- | --- | --- |
| **Historia de preeclampsia en embarazo previo** | Booleano | RR: 7.19 

 | Condición crítica si ocurrió antes de las 34 semanas de gestación.

 |
| **Enfermedades autoinmunes** | Booleano | RR: 9.72 

 | Agrupa condiciones severas como lupus o síndrome antifosfolípido.

 |
| **Hipertensión Arterial Sistémica (Crónica)** | Booleano | RR: 3.6 a 5.4 

 | Enfermedad preexistente al embarazo.

 |
| **Diabetes Pre-gestacional / Mellitus** | Booleano | RR: 3.56 a 3.7 

 | Historia familiar o presencia previa a la gestación.

 |
| **Nuliparidad** | Booleano | RR: 2.91 

 | Paciente en su primer embarazo.

 |
| **Primi-paternidad** | Booleano | RR: 2.91 

 | Primer hijo concebido con una nueva pareja.

 |
| **Embarazo múltiple** | Booleano | RR: 2.90 

 | Gestación de más de un feto.

 |
| **Historia familiar de preeclampsia** | Booleano | RR: 2.90 

 | Antecedentes directos en la madre o hermana de la paciente.

 |
| **Síndrome de Anticuerpos Antifosfolípidos** | Booleano | RR: 2.80 

 | Condición autoinmune específica preexistente.

 |
| **Lupus Eritematoso Sistémico (LES)** | Booleano | RR: 2.50 

 | Condición autoinmune específica preexistente.

 |
| **Índice de Masa Corporal (IMC)** | Numérico | RR: 2.47 

 | Riesgo elevado si el IMC es mayor a 30 kg/m2 previo al embarazo.

 |
| **Presión sistólica (Temprana)** | Numérico | RR: 2.40 

 | Lecturas mayores a 130 mmHg antes de la semana 20.

 |
| **Historia temprana de enf. cardiovascular** | Booleano | RR: 2.0 a 3.0 

 | Antecedentes familiares o personales tempranos.

 |
| **Edad mayor o igual a 40 años (Multípara)** | Booleano | RR: 1.96 

 | Riesgo segmentado por edad en pacientes con embarazos previos.

 |
| **Enfermedad Renal Crónica (ERC)** | Booleano | RR: 1.80 

 | Presencia de nefropatías previas.

 |
| **Edad mayor o igual a 40 años (Primípara)** | Booleano | RR: 1.68 

 | Riesgo segmentado por edad en pacientes nulíparas.

 |
| **Técnicas de reproducción asistida** | Booleano | RR: 1.32 a 1.83 

 | Paciente sometida a tratamientos de fertilidad.

 |
| **Presión diastólica (Temprana)** | Numérico | RR: 1.40 

 | Lecturas mayores a 80 mmHg antes de la semana 20.

 |
| **Periodo intergenésico mayor a 10 años** | Booleano | RR: 1.12 

 | Tiempo de separación entre embarazos consecutivos.

 |
| **Etnia de riesgo** | Categórico | RR: 2.0 a 4.0 

 | Poblaciones con mayor propensión como afrocaribeñas y del sur de Asia.

 |
| **Ganancia de peso gestacional** | Numérico | Sin RR definido 

 | Capturar si la paciente gana más de 0.5 kg por semana.

 |