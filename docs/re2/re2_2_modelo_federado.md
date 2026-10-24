# RE2.2 — Prototipo funcional de detección federada de intrusiones

## 1. Objetivo

Integrar la arquitectura del modelo de detección desarrollado en el RE2.1 con el mecanismo de aprendizaje federado del RE1.3 para entrenar y evaluar un modelo global de clasificación binaria sobre UNSW-NB15.

El prototipo utiliza TensorFlow Federated (TFF), un servidor agregador y tres clientes simulados. Cada cliente entrena con su partición local y el servidor combina sus actualizaciones mediante FedAvg.

La ejecución oficial documentada es:

```text
20260926T183336Z_3fc39f61
```

Este documento describe su configuración, procedimiento, resultados y evidencias. La comparación formal con el modelo centralizado corresponde al RE2.3.

## 2. Alcance experimental

La ejecución comprende:

- Particionado de TRAIN + VALIDATION entre tres clientes.
- Construcción del modelo federado con la arquitectura seleccionada en el RE2.1.
- Entrenamiento durante 20 rondas, con participación de todos los clientes.
- Exportación del modelo global de la última ronda en formato `.keras`.
- Evaluación independiente sobre TEST.
- Conservación de configuraciones, historial, métricas y registros de ejecución.

El modelo federado se inicializa desde cero. Se reutiliza la configuración del modelo centralizado, no sus pesos entrenados.

Los clientes y el servidor se simulan en un mismo entorno de ejecución. Esta implementación no constituye un despliegue entre tres equipos o microempresas reales.

## 3. Organización del código

Todas las rutas de este documento se interpretan desde la raíz del repositorio.

| Archivo | Responsabilidad |
|---|---|
| `src/federated/partition_unsw.py` | Generar y verificar las particiones de UNSW-NB15. |
| `src/federated/model.py` | Cargar la configuración y construir el modelo Keras y su adaptación a TFF. |
| `src/federated/server.py` | Configurar los optimizadores y construir el proceso FedAvg. |
| `src/federated_re1_3/client_data.py` | Proporcionar `ClientPartition` y su conversión a datasets TensorFlow. |
| `src/models/federated/train_config.json` | Definir el escenario de entrenamiento. |
| `src/models/federated/train_federated.py` | Ejecutar las rondas y exportar el modelo global. |
| `src/models/federated/evaluate_federated.py` | Evaluar el modelo guardado sobre TEST y exportar los resultados. |

El entrenamiento actual reutiliza la clase `ClientPartition` del RE1.3 para representar los datos locales. No utiliza el generador de datos sintéticos del prototipo anterior.

## 4. Datos y particionado

### 4.1 Conjuntos utilizados

| Conjunto | Registros | Uso |
|---|---:|---|
| TRAIN + VALIDATION | 175 341 | Entrenamiento federado definitivo. |
| TEST | 82 332 | Evaluación final independiente. |

Cada registro contiene **192 características procesadas**.

Las clases son:

| Etiqueta | Clase | Registros en TEST |
|---|---|---:|
| `0` | Benigno | 37 000 |
| `1` | Malicioso | 45 332 |

Se utiliza TRAIN + VALIDATION para mantener el mismo conjunto de datos disponible para el entrenamiento definitivo del modelo centralizado.

VALIDATION deja de ser un conjunto independiente durante este entrenamiento final. La arquitectura y el umbral proceden de la selección previa; TEST no se utiliza para ajustar esos valores ni para seleccionar una ronda.

El particionado reutiliza los datos ya procesados y no vuelve a ajustar el preprocesador.

### 4.2 Escenario oficial

| Parámetro | Valor |
|---|---|
| Número de clientes | 3 |
| Conjunto repartido | TRAIN + VALIDATION |
| Estrategia | Reparto aleatorio estratificado por clase, denominado IID en el código |
| `NON_IID` | `False` |
| Semilla | 42 |
| Mínimo de registros por cliente | 100 |
| Inclusión de TEST | No |

La estratificación busca mantener proporciones de clases similares entre clientes. No demuestra que todas las características tengan distribuciones idénticas.

Las particiones se almacenan en:

```text
data/processed/unsw_nb15/federated/train_validation_iid_clients_3_seed_42_min_100/
```

| Archivo | Contenido |
|---|---|
| `client_01.npz` | Datos e índices del primer cliente. |
| `client_02.npz` | Datos e índices del segundo cliente. |
| `client_03.npz` | Datos e índices del tercer cliente. |
| `distribution.csv` | Cantidades y porcentajes por clase y cliente. |
| `manifest.json` | Configuración, fuentes, distribución, verificaciones y huellas de los archivos de origen. |

Cada archivo de cliente conserva `X`, `y`, los índices dentro del conjunto combinado y la procedencia de sus filas.

El particionador verifica que todas las filas se asignen exactamente una vez, sin índices repetidos ni faltantes. Esto comprueba la exclusividad de las filas asignadas; no implica una búsqueda de registros con contenido idéntico en el dataset original.

## 5. Modelo y configuración federada

### 5.1 Arquitectura

La configuración seleccionada es **C5_Pyme_Max**.

| Componente | Configuración |
|---|---|
| Entrada | 192 características |
| Primera capa oculta | Dense, 64 neuronas, ReLU |
| Regularización | Dropout de 0,20 después de la primera capa oculta |
| Segunda capa oculta | Dense, 32 neuronas, ReLU |
| Salida | Dense, 1 neurona, sigmoide |
| Parámetros | 14 465 |
| Función de pérdida | Binary cross-entropy |
| Tamaño de lote | 256 |
| Umbral de clasificación | 0,80 |

La salida representa la probabilidad estimada de tráfico malicioso. Durante la evaluación se aplica:

```python
prediction = (probability >= 0.80)
```

El archivo `.keras` produce probabilidades. El umbral se conserva en la configuración de la ejecución y lo aplica el evaluador.

### 5.2 Entrenamiento federado

| Parámetro | Valor |
|---|---|
| Algoritmo | FedAvg |
| Ponderación | Cantidad de ejemplos procesados por cliente (`NUM_EXAMPLES`) |
| Clientes por ronda | Los 3 clientes |
| Rondas | 20 |
| Épocas locales por ronda | 1 |
| Optimizador local | Adam |
| Tasa de aprendizaje local | 0,001 |
| Optimizador del servidor | SGD |
| Tasa de aprendizaje del servidor | 1,0 |
| Momentum del servidor | 0,0 |
| Semilla | 42 |
| Modelo exportado | Modelo global de la última ronda |

El optimizador local actualiza el modelo de cada cliente. El servidor agrega las actualizaciones y utiliza SGD con tasa 1,0 y sin momentum para aplicarlas al modelo global.

No se realiza una búsqueda adicional de hiperparámetros federados en esta ejecución.

Aunque cada registro participa en una época local por ronda, 20 rondas federadas no equivalen a 20 épocas de optimización centralizada: las actualizaciones locales se combinan al finalizar cada ronda.

### 5.3 Archivo de configuración

La configuración de entrenamiento está en:

```text
src/models/federated/train_config.json
```

```json
{
  "partitions_dir": "data/processed/unsw_nb15/federated/train_validation_iid_clients_3_seed_42_min_100",
  "model_config_path": "results/re2/centralized/tuning/best_config.json",
  "metadata_path": "data/processed/unsw_nb15/metadata.json",
  "output_root": "artifacts/runs/federated",
  "num_rounds": 20,
  "local_epochs": 1,
  "server_learning_rate": 1.0,
  "seed": 42
}
```

El número de clientes se obtiene del manifiesto de las particiones.

La configuración efectiva queda copiada junto al modelo. Para evaluar una ejecución se utiliza esa copia, evitando depender de cambios posteriores en `best_config.json`.

Los campos `cv_metrics`, `cv_best_epochs` y otros datos de selección conservados dentro de la configuración describen la búsqueda previa del RE2.1. No son resultados de validación cruzada del modelo federado.

## 6. Procedimiento de ejecución

Los comandos se ejecutan desde la raíz del repositorio, con el entorno del proyecto activado.

### 6.1 Verificar el entorno y los insumos

```bash
python tests/verify_environment.py
```

Este script comprueba la importación de las dependencias principales y muestra sus versiones.

Deben estar disponibles:

- `data/processed/unsw_nb15/train.npz`
- `data/processed/unsw_nb15/validation.npz`
- `data/processed/unsw_nb15/test.npz`
- Los metadatos y artefactos del preprocesamiento correspondiente.
- `results/re2/centralized/tuning/best_config.json`

Para reproducir la ejecución oficial, la configuración debe coincidir con los valores documentados, incluido el umbral de 0,80.

### 6.2 Generar las particiones

Comprobar en `src/federated/partition_unsw.py`:

```python
NUM_CLIENTS = 3
NON_IID = False
SEED = 42
MIN_CLIENT_SAMPLES = 100
INCLUDE_VALIDATION = True
```

Ejecutar:

```bash
python -m src.federated.partition_unsw
```

El script no sobrescribe un escenario existente. Si las particiones oficiales ya están disponibles, se reutilizan.

### 6.3 Entrenar y exportar el modelo global

```bash
python -m src.models.federated.train_federated
```

También puede indicarse explícitamente la configuración:

```bash
python -m src.models.federated.train_federated \
  --config src/models/federated/train_config.json
```

El entrenador:

1. Carga la configuración y las particiones.
2. Verifica dimensiones, etiquetas, distribución e índices.
3. Inicializa el proceso federado.
4. Ejecuta las 20 rondas con todos los clientes.
5. Registra las métricas locales agregadas y el tiempo por ronda.
6. Transfiere los pesos globales finales a un modelo Keras.
7. Guarda el modelo y los metadatos de la ejecución.

Este script no carga TEST ni realiza una evaluación independiente.

Cada entrenamiento genera un identificador nuevo. Una reproducción tendrá su propia carpeta y no debe confundirse con la ejecución oficial documentada.

### 6.4 Evaluar sobre TEST

Para evaluar una nueva ejecución:

```bash
python -m src.models.federated.evaluate_federated \
  --run-dir artifacts/runs/federated/ID_DE_LA_EJECUCION
```

Sustituir `ID_DE_LA_EJECUCION` por el identificador generado durante el entrenamiento.

La ejecución oficial utilizó:

```bash
python -m src.models.federated.evaluate_federated \
  --run-dir artifacts/runs/federated/20260926T183336Z_3fc39f61
```

Sus resultados ya existen. El evaluador cancela la operación si encuentra la carpeta de resultados, para evitar sobrescribirlos.

Antes de la inferencia, el evaluador comprueba:

- Estado de entrenamiento completado.
- Coincidencia entre rondas solicitadas y completadas.
- Declaraciones de exclusión de TEST durante el entrenamiento.
- Huella SHA-256 del modelo.
- Compatibilidad de dimensiones.
- Tamaño y distribución de clases de TEST.
- Huellas de los artefactos de preprocesamiento registrados en el manifiesto.

Posteriormente obtiene las probabilidades y aplica el umbral guardado. No entrena el modelo ni busca otro umbral sobre TEST.

## 7. Evidencias generadas

### 7.1 Entrenamiento

Directorio oficial:

```text
artifacts/runs/federated/20260926T183336Z_3fc39f61/
```

| Archivo | Evidencia |
|---|---|
| `global_model.keras` | Arquitectura y pesos globales de la última ronda. |
| `training_history.csv` | Métricas locales agregadas, ejemplos procesados y duración por ronda. |
| `training_config.json` | Configuración solicitada. |
| `effective_config.json` | Configuración efectiva del modelo y de FedAvg. |
| `partition_manifest.json` | Copia del manifiesto utilizado. |
| `partition_hashes.json` | Huellas de las particiones y del manifiesto. |
| `run_summary.json` | Estado, rondas completadas, versiones y huella del modelo. |

El `.keras` permite realizar inferencia, pero no es un checkpoint completo para reanudar exactamente el estado del proceso federado.

### 7.2 Evaluación

Directorio oficial:

```text
results/re2/federated/20260926T183336Z_3fc39f61/
```

| Archivo | Evidencia |
|---|---|
| `metrics.json` | Métricas, matriz, reporte por clase y trazabilidad. |
| `metrics.csv` | Métricas en formato tabular. |
| `confusion_matrix.csv` | Matriz con filas reales y columnas predichas. |
| `classification_report.txt` | Reporte de clasificación por clase. |
| `effective_config.json` | Configuración asociada al modelo evaluado. |

Las predicciones individuales se guardan separadamente en:

```text
artifacts/predictions/20260926T183336Z_3fc39f61/predictions.csv
```

Sus columnas son `test_row`, `real`, `probability` y `prediction`. `test_row` representa la posición dentro de `test.npz`.

### 7.3 Logs

Los registros publicados de la ejecución son:

```text
logs/re2_2/federated_training.log
logs/re2_2/federated_results.log
```

Los scripts imprimen el progreso en consola. Para conservar logs de nuevas ejecuciones debe capturarse su salida en archivos distintos.

Las carpetas de datos, modelos y predicciones pueden mantenerse localmente sin estar versionadas en Git. Deben conservarse junto con las configuraciones para reproducir y auditar el experimento; clonar el código no sustituye disponer de esos insumos.

## 8. Resultados oficiales sobre TEST

Fuente:

```text
results/re2/federated/20260926T183336Z_3fc39f61/metrics.json
```

| Métrica | Valor | Porcentaje |
|---|---:|---:|
| Accuracy | 0,903258 | 90,33 % |
| Precision | 0,929634 | 92,96 % |
| Recall | 0,891798 | 89,18 % |
| F1-score | 0,910323 | 91,03 % |
| ROC-AUC | 0,973294 | — |
| Tasa de falsos positivos (FPR) | 0,082703 | 8,27 % |
| Especificidad | 0,917297 | 91,73 % |
| Tasa de falsos negativos (FNR) | 0,108202 | 10,82 % |

Precision, recall y F1-score corresponden a la clase maliciosa (`1`). ROC-AUC se calcula utilizando las probabilidades, antes de aplicar el umbral.

### 8.1 Matriz de confusión

Las filas representan las clases reales y las columnas las predicciones.

| Clase real | Predicho benigno (`0`) | Predicho malicioso (`1`) | Total |
|---|---:|---:|---:|
| Benigno (`0`) | 33 940 | 3 060 | 37 000 |
| Malicioso (`1`) | 4 905 | 40 427 | 45 332 |
| Total | 38 845 | 43 487 | 82 332 |

- Verdaderos negativos: **33 940**.
- Falsos positivos: **3 060**.
- Falsos negativos: **4 905**.
- Verdaderos positivos: **40 427**.

El modelo detectó 40 427 de los 45 332 registros maliciosos y clasificó incorrectamente como maliciosos 3 060 registros benignos.

### 8.2 Reporte por clase

| Clase | Precision | Recall | F1-score | Soporte |
|---|---:|---:|---:|---:|
| Benigno | 0,873729 | 0,917297 | 0,894983 | 37 000 |
| Malicioso | 0,929634 | 0,891798 | 0,910323 | 45 332 |
| Promedio macro | 0,901682 | 0,904548 | 0,902653 | 82 332 |
| Promedio ponderado | 0,904510 | 0,903258 | 0,903429 | 82 332 |

El recall malicioso supera el 80 % en **9,18 puntos porcentuales**. Esta comprobación complementa la evaluación, pero no sustituye los indicadores específicos del RE2.2.

## 9. Interpretación del historial de entrenamiento

El log registra una reducción de la pérdida de **0,2477** en la primera ronda a **0,1166** en la última.

Estas métricas se calculan mientras los clientes entrenan y posteriormente se agregan mediante TFF. No representan una evaluación independiente del modelo global final.

Por tanto:

- El historial describe la evolución del entrenamiento.
- `metrics.json` contiene los resultados finales sobre TEST.
- Se conserva el modelo de la ronda 20.
- No se selecciona una “mejor ronda” utilizando TEST.

## 10. Verificación de los indicadores del RE2.2

| Indicador | Evidencia | Resultado |
|---|---|---|
| Integrar el modelo de detección del RE2.1 y el mecanismo federado del RE1.3. | Configuración C5_Pyme_Max, adaptación Keras/TFF, FedAvg y reutilización de `ClientPartition`. | Se integra la arquitectura seleccionada con el proceso federado. |
| Ejecutar al menos 5 rondas con un servidor agregador y un mínimo de 2 clientes, cada uno con su partición de UNSW-NB15. | Configuración y `federated_training.log`. | Se completaron 20 rondas con 3 clientes simulados. |
| Obtener y evaluar al menos un modelo global capaz de producir predicciones benignas y maliciosas. | `global_model.keras`, `metrics.json` y matriz de confusión. | Modelo global evaluado sobre 82 332 registros, con predicciones de ambas clases. |
| Disponer de scripts, configuraciones y logs para reproducir las ejecuciones utilizadas en el análisis. | Código, configuración efectiva, logs y artefactos asociados al identificador oficial. | Ejecución identificada y documentada; deben conservarse los insumos y metadatos locales junto con las evidencias publicadas. |

Las evidencias documentan la integración y ejecución funcional del prototipo. La reproducción completa requiere los datos procesados, las particiones y los archivos de la ejecución, además del código del repositorio.

## 11. Limitaciones

- Los resultados corresponden a una ejecución con tres clientes y particionado estratificado. No permiten generalizar a escenarios Non-IID, más clientes o diferentes semillas.
- La simulación mantiene una separación lógica entre clientes y servidor. El programa coordinador carga las particiones en el mismo entorno; no se ha validado un despliegue físicamente distribuido.
- No se evaluaron privacidad diferencial, agregación segura, ataques de envenenamiento ni resistencia frente a clientes adversarios.
- La semilla y las versiones facilitan la reproducción, pero no garantizan resultados numéricos idénticos entre distintos entornos.
- El FPR de 8,27 % y el FNR de 10,82 % muestran errores relevantes que deben considerarse al interpretar la utilidad operativa del modelo.
- El criterio de FPR utilizado en la selección previa no constituye una garantía de obtener ese mismo valor sobre TEST.
- Este resultado no demuestra superioridad frente al enfoque centralizado. Esa conclusión requiere el análisis comparativo del RE2.3.

## 12. Relación con los demás resultados

- **RE1.3:** aporta el mecanismo federado y los componentes reutilizables del prototipo.
- **RE2.1:** aporta el preprocesamiento y la configuración seleccionada del modelo de referencia.
- **RE2.2:** integra los componentes, entrena el modelo global y obtiene sus métricas finales.
- **RE2.3:** comparará ambos enfoques utilizando el mismo TEST, las mismas métricas y una regla de clasificación consistente.

El modelo global y los resultados de la ejecución `20260926T183336Z_3fc39f61` constituyen la referencia federada documentada para esa comparación.