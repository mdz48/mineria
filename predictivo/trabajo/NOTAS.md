# Notas del Proyecto - Dataset Electrico 2107
> Registro vivo de hallazgos. Las conclusiones marcadas con [DERIVADO] se obtuvieron
> exclusivamente desde los datos. Las marcadas con [EXTERNO] vienen de fuentes externas
> y deben tomarse con cautela ya que el docente pudo haber modificado el dataset.

---

## RESUMEN EJECUTIVO (Descripción para el Reporte)

Este dataset es una **serie de tiempo multivariante de alta resolución (5 minutos)** proveniente del sistema SCADA de una planta solar fotovoltaica. El objetivo implícito es predecir la potencia de salida (`ac_power`) basándonos en el comportamiento temporal. Los datos requieren un preprocesamiento estricto mediante eliminación de *outliers* extremos y la imputación de bloques masivos de pérdida de señal antes de aplicar modelos predictivos.

### Radiografía del Dataset y Significado de Variables

**1. La estructura del tiempo (Filas):**
Cada fila es una "fotografía" del estado de la planta tomada cada 5 minutos. El sistema sigue un ciclo diurno perfecto: "despierta" (>0 kW) alrededor de las 7:00 AM con la salida del sol, llega a su pico máximo a las 12:00 del mediodía, y se "duerme" (0 kW) al atardecer.

**2. El significado de los Sensores (Columnas):**
La planta cuenta con 24 inversores idénticos (~30 kW de capacidad cada uno). Cada inversor tiene 5 sensores críticos que documentan la conversión de energía solar a eléctrica:

* **Lado de los Paneles (Energía cruda / Corriente Directa - DC):**
  - **`dc_voltage` (Voltios):** Tensión eléctrica proveniente de los paneles solares (opera a promedios altos, ~668V).
  - **`dc_current` (Amperios):** Flujo de electrones de los paneles. Varía fuertemente según la radiación solar.

* **Lado de la Red Eléctrica (Energía procesada / Corriente Alterna - AC):**
  - **`ac_voltage` (Voltios):** Voltaje al que el inversor inyecta la energía a la red pública (~277V a 290V, indicando conexión comercial/industrial trifásica).
  - **`ac_current` (Amperios):** Corriente alterna inyectada a la red.
  - **`ac_power` (Kilowatts - kW):** La potencia real útil inyectada a la red. **Esta es la variable objetivo (Target)** para el modelado predictivo.

**3. Calidad y Anomalías (Requisitos de Limpieza):**
- **El Apagón de Datos:** Existen 1,728 filas (6 días continuos) donde las lecturas de los 24 inversores son nulas (NaN). Esto indica un fallo de comunicación del SCADA, no de la planta eléctrica.
- **Sensores con Alucinaciones (Outliers):** Los inversores 04 y 07 sufren de fallas esporádicas en sus sensores, registrando valores físicamente imposibles (trillones de voltios). Deben ser filtrados estadísticamente, ya que su capacidad real es idéntica al resto.

---

## LO QUE SABEMOS DEL SISTEMA (solo desde los datos)

### Tipo de sistema [DERIVADO - alta confianza]
- Es un sistema de generacion electrica con ciclo diurno perfecto
- La generacion es 0 de noche y sigue una curva de campana durante el dia
- El pico de generacion ocurre exactamente al mediodia (12:00 hrs)
- Esto es consistente con un sistema solar fotovoltaico de paneles fijos
- No hay evidencia de generacion nocturna (descarta eolico o hidro)

### Capacidad del sistema [DERIVADO]
- Potencia AC maxima absoluta registrada: 721.61 kW
- Potencia media diurna (horas de sol): 330.80 kW
- Factor de capacidad observado: ~46% (721 kW pico vs 330 kW promedio diurno)
- Los inversores 07 y 04 tienen maximos mucho mayores (~103-105 kW) que el resto (~30 kW)
    -> Esto sugiere que  -> Esto sugiere que ES UN ESPEJISMO POR OUTLIERS (verificado en analisis profundo)
  -> Los 22 inversores restantes tienen maximos casi identicos (~30 kW) -> mismo modelo
  -> TODOS LOS INVERSORES SON IGUALES (~30 kW maximo real)

### Voltaje AC [DERIVADO]
- Voltaje AC promedio durante horas de generacion: 276.5 V (mediana ~287-290 V por inversor)
- Voltaje maximo observado: 347 V
- El voltaje promedio de ~276 V con mediana de ~288 V apunta a un sistema de:
  -> 277V / 480V trifasico (comun en sistemas solares comerciales de EE.UU.)
  -> o 240V monofasico en algunos inversores
- El voltaje es estable durante la generacion (std baja una vez que el sistema esta encendido)
- Los valores de 0 V durante horas diurnas corresponden a momentos de arranque o falla temporal

### Patron Laboral vs Fin de Semana [DERIVADO]
- Potencia media dias laborales: 329.84 kW
- Potencia media fin de semana: 333.18 kW
- Diferencia: +1.01% (estadisticamente insignificante)
- CONCLUSION: El sistema no tiene autoconsumo ni responde a demanda humana.
  Toda la energia generada se inyecta a la red electrica de forma continua.
  Es una planta de generacion pura (no un sistema de autoconsumo residencial o fabril).

### Patron estacional y latitud estimada [DERIVADO]
Tabla de horas de generacion por mes:

| Mes | Amanecer | Atardecer | Horas de luz | Pico (kW) |
|-----|----------|-----------|--------------|-----------|
| Ene | 8:00     | 16:00     | 9 hrs        | 403.7     |
| Feb | 7:00     | 17:00     | 11 hrs       | 485.2     |
| Mar | 7:00     | 18:00     | 12 hrs       | 483.9     |
| Abr | 7:00     | 18:00     | 12 hrs       | 571.4     |
| May | 7:00     | 19:00     | 13 hrs       | 608.1     |
| Jun | 6:00     | 19:00     | 14 hrs       | 558.6     |
| Jul | 7:00     | 19:00     | 13 hrs       | 533.6     |
| Ago | 7:00     | 19:00     | 13 hrs       | 549.6     |
| Sep | 7:00     | 18:00     | 12 hrs       | 572.6     |
| Oct | 7:00     | 17:00     | 11 hrs       | 563.9     |
| Nov | 7:00     | 16:00     | 10 hrs       | 455.5     |
| Dic | 7:00     | 16:00     | 10 hrs       | 399.1     |

Observaciones:
- Invierno (Dic-Ene): 9-10 horas de luz, pico ~400 kW
- Verano (May-Jun): 13-14 horas de luz, pico ~560-608 kW
- Diferencia de 5 horas de luz entre invierno y verano
- Este rango de variacion estacional (9 a 14 hrs) es consistente con latitud entre 35 y 42 grados Norte
- El pico de Mayo (608 kW) supera al de Junio (558 kW) -> posiblemente Junio tiene mas dias nublados o calurosos
  (las celdas solares pierden eficiencia con el calor excesivo del verano en zonas aridas)
- Los timestamps parecen estar en hora local (el pico siempre es a las 12:00)

---

## ESTRUCTURA DEL DATASET

| Campo                | Valor                                     |
|----------------------|-------------------------------------------|
| Archivo              | 2107_electrical_data.csv                  |
| Dimensiones          | 632,952 filas x 120 columnas              |
| Columna temporal     | measured_on (indice DatetimeIndex)        |
| Rango temporal       | 2017-11-01 a (fecha cierre pendiente)     |
| Frecuencia           | 1 medicion cada 5 minutos                 |
| Dispositivos         | 24 inversores (inv_01 a inv_24)           |
| Senales por inversor | dc_current, dc_voltage, ac_current, ac_voltage, ac_power |
| Excepcion            | inv_05 NO tiene columna dc_voltage        |
| Tipo de datos        | float32 optimizado + 1 datetime           |
| Uso de RAM           | ~292 MB (con float32)                     |

---

## HALLAZGOS TECNICOS

### Calidad del Dato
- 114 de 119 columnas tienen nulos
- Todas las columnas con nulos tienen exactamente 1,728 nulos (6 dias exactos)
- Conclusion: Fue un corte del sistema SCADA, no una falla electrica
- Sensores con varianza cero: 0 (ninguno completamente muerto)

### Verificacion Fisica
- Eficiencia mediana Inversor 01: 96.78% (rango normal confirmado para inversores solares)
- IMPORTANTE: ac_power esta en kW, dc_voltage * dc_current produce W
  Formula correcta: eficiencia = (ac_power * 1000) / (dc_voltage * dc_current)
- 5.4% de registros con eficiencia invalida -> momentos de arranque/parada del inversor

### Anomalia detectada en inversores
- inv_07 e inv_04 tienen picos anomalos (outliers) de ~103-105 kW
- Los 22 inversores restantes tienen maxima de ~30 kW
- Analizando los percentiles, hasta el 99% todos los inversores generan exactamente lo mismo (~29.95 kW). Los picos de 105 kW son errores de lectura del sensor.
- Para modelado predictivo sera importante separar estos grupos o normalizar por capacidad

---

## HIPOTESIS ACTUALES

| Hipotesis                                            | Estado         | Base             |
|------------------------------------------------------|----------------|------------------|
| Sistema solar fotovoltaico                           | Alta confianza | Patron de datos  |
| Planta de generacion pura (inyeccion a red)          | Confirmada     | Analisis laboral |
| Latitud ~35-42 grados Norte (Hemisferio Norte)       | Alta confianza | Patron estacional|
| Los timestamps estan en hora local                   | Probable       | Pico a las 12:00 |
| Todos los inversores son idénticos    | Confirmada     | Maximos por inv  |
de investigacion encontro que los nombres de columna exactos
(incluyendo los ID de sensor como _inv_149579) coinciden con el dataset publico
PVDAQ #2107 "Farm Solar Array" del programa OEDI/DOE. Si el dataset NO fue modificado:
- Ubicacion: Arbuckle, California (38.99 N, 122.13 W)
- Capacidad instalada: 893 kW DC
- Fuente: 2023 DOE Solar Data Prize / NREL
 ] Analizar el 5.4% de registros con eficiencia aparentemente invalida - podrian ser momentos de arranque/parada del inversor

---

## DEFINICION DEL OBJETIVO PREDICTIVO (MODELADO)

Se ha decidido enfocar el **Aprendizaje Supervisado** (y posterior Forecasting) en predecir la **Energia Total Diaria** generada por la planta (Opcion B), en lugar de predecir la potencia en intervalos de 5 minutos.

**Justificacion:**
1. **Utilidad de Negocio:** Predecir la energia total del dia es mucho mas util para un operador de red, quien necesita planificar cuanta electricidad inyectara al sistema o vendera.
2. **Reduccion de Ruido:** Al agregar los datos a nivel diario, eliminamos la alta volatilidad de los intervalos de 5 minutos (nubes pasajeras, ceros nocturnos masivos).
3. **Manejo de Dimensionalidad:** Permite colapsar las 119 variables altamente colineales en promedios diarios estables (voltaje medio, corriente media) que representen el estado general de la planta, evitando problemas de multicolinealidad severa en la Regresion Lineal Multiple.

---

## RESULTADOS: APRENDIZAJE SUPERVISADO (PREDICCION DE ENERGIA DIARIA)

Se entrenaron dos modelos competidores para predecir la energia total diaria de la planta (kWh),
usando la particion cronologica 80/20 (1,722 dias train / 431 dias test) y las variables agregadas
de planta + features ciclicos temporales (seccion 9 del notebook).

**Modelos Competidores:**
1. **Regresion Lineal Multiple (RLM):** sobre datos escalados (StandardScaler), interpretable via coeficientes.
2. **Random Forest Regressor:** 300 arboles, sin escalado, interpretable via importancias.

**Desempeño en Test (431 dias):**

| Modelo | RMSE (kWh) | MAE (kWh) | R2 |
|---|---|---|---|
| Baseline (media del train) | 1481.4 | 1230.9 | -0.013 |
| Regresion Lineal Multiple | 73.7 | 57.4 | 0.9975 |
| Random Forest | 71.3 | 55.0 | 0.9977 |

**Hallazgos:**
- Ambos modelos reducen el RMSE del baseline en ~95%. Error relativo < 2% (media diaria ~3,966 kWh).
- **Empate tecnico con leve ventaja de Random Forest.** La relacion fisica es casi lineal
  (Energia ~ Corriente x Voltaje), por eso la RLM compite de igual a igual.
- Coeficientes RLM e importancias RF coinciden: `planta_ac_current` domina (importancia 94.6% en RF;
  coeficiente escalado 1472.8 kWh en RLM). Fisicamente coherente: voltaje de red casi constante (~280 V)
  -> potencia proporcional a corriente.
- **Limitacion documentada:** las predictoras son mediciones del mismo dia (nowcasting/diagnostico).
  Util para validar coherencia de planta y detectar dias anomalos, no para pronosticar el futuro.
  El pronostico real queda para la etapa de Forecasting.

---

## RESULTADOS: APRENDIZAJE NO SUPERVISADO (MANTENIMIENTO PREDICTIVO)

Se ejecutó un análisis de clustering/detección de anomalías en los datos de alta resolución (5 minutos) enfocándose en el Inversor 01 durante el año 2018. El objetivo fue identificar fallas sutiles o anomalías operativas (Mantenimiento Predictivo).

**Modelos Competidores:**
1. **Isolation Forest:** Algoritmo basado en árboles que aísla anomalías.
2. **DBSCAN:** Algoritmo basado en densidad (requirió escalado `StandardScaler`).

**Desempeño y Resultados:**
- **Isolation Forest:** Configurado con 1% de contaminación, detectó **407 anomalías**. Demostró una lógica física superior: detectó los puntos donde la relación matemática de conversión (Voltaje/Corriente vs Potencia AC) se rompía, marcando certeramente caídas de eficiencia (ej. alta irradiación/voltaje pero muy poca potencia de salida). Es altamente escalable y rápido en memoria.
- **DBSCAN:** Detectó **102 anomalías**. Su enfoque geométrico (euclidiano) tendió a confundir las transiciones normales (como atardeceres o el paso lento de una nube) como anomalías debido a los cambios de densidad. Adicionalmente, sufrió de la explosión de memoria clásica $O(N^2)$, lo que obligó a reducir la muestra de datos analizados.

**Conclusión:**
**Ganador: Isolation Forest.** Es el modelo ideal para la monitorización industrial en este proyecto, proporcionando "alarmas de mantenimiento" realistas sin saturar el sistema (falsos positivos controlados al 1%) y siendo capaz de procesar años de datos históricos en segundos.
