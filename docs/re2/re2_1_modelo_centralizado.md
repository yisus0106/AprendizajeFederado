# RE2.1 – Modelo de detección de intrusiones basado en tráfico de red

## 1. Propósito y alcance

El Resultado Esperado 2.1 (RE2.1) entrega un modelo funcional de clasificación binaria sobre **UNSW-NB15**, con las clases **benigno (`0`)** y **malicioso (`1`)**. Constituye la línea base centralizada para la integración federada del RE2.2 y la comparación experimental del RE2.3.

Según los indicadores de la tesis, debe evaluarse sobre un conjunto de prueba separado del entrenamiento, reportar **accuracy, precision, recall, F1-score y matriz de confusión**, y alcanzar **recall ≥ 80 % para tráfico malicioso**.

Este documento describe el procedimiento implementado y las evidencias guardadas en la rama `RE2_2`. Las métricas de referencia proceden de `results/re2/centralized/train/metrics.json`; no representan una nueva ejecución realizada durante la revisión documental.

## 2. Archivos y requisitos

| Elemento | Ruta desde la raíz del repositorio |
|---|---|
| Inspección de datos | `src/data/inspect_unsw.py` |
| Preprocesamiento | `src/data/preprocess_unsw.py` |
| Selección por validación cruzada y figuras | `src/models/centralized/tune_centralized.py` |
| Entrenamiento y evaluación final | `src/models/centralized/train_centralized.py` |
| Ejecución del flujo de validación | `scripts/validate_re2_1.sh` |
| Configuración seleccionada | `results/re2/centralized/tuning/best_config.json` |
| Resultados de selección | `results/re2/centralized/tuning/` |
| Resultados del entrenamiento final | `results/re2/centralized/train/` |
| Logs de ejecución | `logs/re2_1/` |
| Modelo final | `artifacts/models/centralized/unsw_nb15_baseline.keras` |
| Predicciones de TEST | `artifacts/predictions/centralized_predictions.csv` |

Se requiere un entorno Python compatible con TensorFlow/Keras, NumPy, pandas, scikit-learn, joblib y Matplotlib. El proyecto conserva referencias de dependencias en `requirements/requirements_tff_cpu.txt` y `requirements/requirements-lock.txt`. No debe suponerse que cualquiera de esos archivos cubra por sí solo todas las importaciones: antes de ejecutar, comprobar también la disponibilidad de **pandas y Matplotlib**. El archivo de dependencias TFF incluye un wheel de JAX específico de CPython 3.11/Linux x86_64; no es una receta universal para otras plataformas.

Los datos originales, datos procesados, modelos `.keras` y `artifacts/` están excluidos mediante `.gitignore`. **Clonar el repositorio no descarga esos archivos**: es necesario disponer del dataset y regenerar o recuperar los artefactos de la ejecución correspondiente.

## 3. Dataset y clases

Se utilizan los archivos oficiales:

- `data/raw/unsw_nb15/UNSW_NB15_training-set.csv`.
- `data/raw/unsw_nb15/UNSW_NB15_testing-set.csv`.

`NUSW-NB15_features.csv` puede conservarse como referencia descriptiva; el preprocesador no lo carga.

| Conjunto oficial | Registros | Benignos (`0`) | Maliciosos (`1`) |
|---|---:|---:|---:|
| Entrenamiento | 175,341 | 56,000 | 119,341 |
| Prueba | 82,332 | 37,000 | 45,332 |

La etiqueta objetivo es `label`. Se excluyen de los predictores `id`, `attack_cat` y la propia etiqueta. Quedan **42 variables**: 39 numéricas y tres categóricas (`proto`, `service`, `state`).

La inspección se realiza con `python src/data/inspect_unsw.py`. Las dimensiones y condiciones de calidad deben comprobarse sobre los archivos disponibles localmente; los CSV originales no están versionados en esta rama.

## 4. Separación de datos y preprocesamiento

`preprocess_unsw.py` divide el entrenamiento oficial con `test_size=0.20`, `random_state=42` y estratificación por `label`:

| Partición persistida | Registros | Benignos | Maliciosos |
|---|---:|---:|---:|
| TRAIN | 140,272 | 44,800 | 95,472 |
| VALIDATION | 35,069 | 11,200 | 23,869 |
| TEST oficial | 82,332 | 37,000 | 45,332 |

El preprocesador se ajusta **solo sobre TRAIN**. Aplica `StandardScaler` a las variables numéricas y `OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)` a las categóricas. Después transforma VALIDATION y TEST sin reajustarse y convierte las entradas a `float32`.

En la ejecución documentada se obtienen **192 características**, conservando su orden. Esta dimensión depende de las categorías aprendidas por el preprocesador y debe comprobarse con `metadata.json`.

Se generan en `data/processed/unsw_nb15/`:

| Archivo | Contenido |
|---|---|
| `train.npz` | Claves `X` e `y`; formas esperadas `(140272, 192)` y `(140272,)` |
| `validation.npz` | Claves `X` e `y`; formas esperadas `(35069, 192)` y `(35069,)` |
| `test.npz` | Claves `X` e `y`; formas esperadas `(82332, 192)` y `(82332,)` |
| `preprocessor.joblib` | Transformador ajustado sobre TRAIN |
| `metadata.json` | Semilla, columnas, tamaños, distribución de clases y nombres de características transformadas |

El preprocesamiento comprueba dimensiones, correspondencia entre entradas y etiquetas y ausencia de valores no finitos. Los scripts centralizados verifican además las etiquetas binarias y la compatibilidad de dimensiones entre particiones.

### 4.1 Uso de TRAIN y VALIDATION en el procedimiento vigente

Las particiones guardadas no equivalen a conjuntos de entrenamiento y validación fijos durante el tuning actual. El selector concatena **TRAIN + VALIDATION**, obteniendo el conjunto de desarrollo de **175,341 registros**, y realiza sobre él validación cruzada estratificada de tres folds.

Una vez seleccionada la configuración, el modelo final se entrena desde cero sobre **todo TRAIN + VALIDATION** durante un número de épocas fijado por CV. VALIDATION, por tanto, **sí participa en el entrenamiento final**. TEST permanece separado de la selección y de las actualizaciones de pesos.

### 4.2 Alcance metodológico de la CV

La CV recibe matrices ya transformadas por un preprocesador ajustado anteriormente sobre TRAIN. **El preprocesamiento no se reajusta dentro de cada fold**. Parte de los registros que quedan en validación de un fold pudo intervenir en el ajuste inicial del escalador y del codificador; por ello, esta CV no constituye una evaluación de todo el pipeline con preprocesamiento independiente por fold.

Esta limitación debe declararse al interpretar las métricas de CV. TEST no interviene en ese ajuste. Una variante con preprocesamiento dentro de cada fold requeriría repetir la selección y generar nuevas evidencias; no debe presentarse como si fuera el procedimiento que produjo los resultados actuales.

## 5. Selección de arquitectura y umbral

`tune_centralized.py` utiliza `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` y los mismos folds para las cinco configuraciones:

| ID | Capas ocultas | Dropout tras la primera capa | Learning rate | Parámetros con 192 entradas |
|---|---|---:|---:|---:|
| C1_Nano | 16 → 8 | 0.10 | 0.001 | 3,233 |
| C2_Micro_Base | 32 → 16 | 0.20 | 0.001 | 6,721 |
| C3_Micro_Robust_FL | 32 → 16 | 0.35 | 0.001 | 6,721 |
| C4_Micro_Slow_FL | 32 → 16 | 0.20 | 0.0005 | 6,721 |
| C5_Pyme_Max | 64 → 32 | 0.20 | 0.001 | 14,465 |

Cada fold utiliza Adam, pérdida `binary_crossentropy`, batch size 256 y hasta 20 épocas, sin `class_weight`. Se aplica `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`. Las semillas de entrenamiento son `42 + número_de_fold`.

Sobre las probabilidades de validación de cada fold se prueban **17 umbrales**, desde 0.10 hasta 0.90 con incrementos de 0.05. Esto produce 85 filas agregadas de arquitectura–umbral, a partir de 15 entrenamientos; no se reentrena para cada umbral.

### 5.1 Regla exacta de selección

1. Filtrar candidatos con **recall medio ≥ 0.80** y **FPR medio + desviación estándar ≤ 0.03**.
2. Solo si no hay candidatos, repetir con límite FPR de 0.05. Si tampoco hay candidatos, detener la selección con error.
3. Calcular el mayor F1 medio entre los candidatos elegibles y conservar los que estén a una distancia máxima de **0.005** de ese valor.
4. Entre estos, preferir menor número de parámetros, después menor FPR medio + desviación, y después mayor F1 medio. Los empates restantes se ordenan por ID de configuración y umbral.

La desviación estándar se calcula entre los tres folds con `ddof=1`. Las bandas de las figuras son descriptivas (± una desviación), **no intervalos de confianza**. Los límites de FPR son criterios de selección en CV; no garantizan el mismo FPR sobre TEST.

### 5.2 Configuración seleccionada y resultados de CV

El `best_config.json` versionado identifica **C5_Pyme_Max**, con **umbral 0.80** y límite de selección `FPR medio + desviación ≤ 0.03`.

| Métrica CV | Media | Desviación entre folds |
|---|---:|---:|
| Accuracy | 0.917099 | 0.003336 |
| Precision | 0.986478 | 0.001113 |
| Recall malicioso | 0.890406 | 0.005593 |
| F1-score | 0.935976 | 0.002789 |
| ROC-AUC | 0.989885 | 0.000185 |
| FPR | 0.026018 | 0.002290 |

El FPR medio + desviación es **0.028308**, inferior a 0.03. Las mejores épocas de la configuración ganadora son **[20, 18, 20]**. El selector utiliza la mediana redondeada mediante `floor(mediana + 0.5)`, dando **20 épocas finales**.

## 6. Modelo centralizado final

| Componente | Configuración congelada |
|---|---|
| Entrada | 192 características |
| Primera capa oculta | Dense(64), ReLU |
| Regularización | Dropout(0.20), solo tras la primera capa oculta |
| Segunda capa oculta | Dense(32), ReLU |
| Salida | Dense(1), Sigmoid |
| Parámetros entrenables | 14,465 |
| Optimizador | Adam, learning rate 0.001 |
| Pérdida | Binary Crossentropy |
| Batch size | 256 |
| Datos de entrenamiento final | TRAIN + VALIDATION: 175,341 registros |
| Épocas finales | 20, fijadas por CV |
| EarlyStopping final | No |
| Class weight | No |
| Semilla final | 42 |
| Umbral de clasificación | 0.80 |

`train_centralized.py` lee la configuración desde `best_config.json`, comprueba sus campos y verifica que el modelo tenga el número de parámetros esperado. Entrena un modelo nuevo; no reutiliza los pesos de los folds.

La salida es la probabilidad estimada de clase maliciosa. La regla final es:

```python
prediction = (probability >= 0.80).astype(int)
```

El historial de entrenamiento contiene `loss`, `accuracy`, `precision`, `recall` y `auc`. Las métricas binarias Keras de ese historial usan su umbral predeterminado **0.50**; no deben confundirse con las métricas finales sobre TEST, calculadas explícitamente con **0.80**. No hay columnas `val_*` en el entrenamiento final.

## 7. Evaluación sobre TEST y resultados registrados

La evaluación utiliza los **82,332 registros** del TEST oficial, con la clase positiva `1 = malicioso`. La configuración y el umbral se mantienen fijos.

| Métrica | Valor | Porcentaje cuando corresponde |
|---|---:|---:|
| Accuracy | 0.906173 | 90.62 % |
| Precision | 0.910011 | 91.00 % |
| Recall malicioso | 0.920630 | 92.06 % |
| F1-score | 0.915290 | 91.53 % |
| ROC-AUC | 0.976246 | — |
| FPR | 0.111541 | 11.15 % |
| Specificity | 0.888459 | 88.85 % |

La matriz de confusión tiene filas de clase real y columnas de clase predicha:

| Clase real / predicción | Benigno (`0`) | Malicioso (`1`) |
|---|---:|---:|
| Benigno (`0`) | TN = 32,873 | FP = 4,127 |
| Malicioso (`1`) | FN = 3,598 | TP = 41,734 |

El recall es `TP / (TP + FN) = 41734 / 45332 = 0.920630`, por lo que supera el mínimo del 80 %. El FPR es `FP / (TN + FP) = 4127 / 37000 = 0.111541`.

**El FPR de TEST (11.15 %) supera los límites usados para seleccionar candidatos en CV.** Se cumple el indicador de recall del RE2.1, pero no corresponde afirmar que el modelo mantenga en TEST una tasa de falsas alarmas inferior al 3 % o al 5 %. Esta diferencia debe conservarse en la discusión de resultados, sin reajustar el umbral a partir de TEST.

## 8. Evidencias y trazabilidad

| Archivo | Información guardada |
|---|---|
| `tuning/full_cv_matrix_search.csv` | Media y desviación de métricas para las 85 combinaciones |
| `tuning/cv_training_epochs.csv` | Mejor época, épocas ejecutadas y mejor pérdida de validación por configuración y fold |
| `tuning/eligible_matrix.csv` | Candidatos que cumplen los filtros de recall y FPR |
| `tuning/top_candidates_cv.csv` | Candidatos dentro de la tolerancia de F1, ordenados para el desempate |
| `tuning/best_config.json` | Arquitectura, umbral, regla de selección, métricas CV y épocas finales |
| `tuning/figures/cv_*.png` y `cv_*.pdf` | Cinco figuras, una por configuración, en ambos formatos |
| `train/metrics.json` | Configuración, muestras, métricas TEST, matriz de confusión y SHA-256 de la configuración |
| `train/training_history.csv` | Métricas de entrenamiento por época |

Las rutas abreviadas de esta tabla son relativas a `results/re2/centralized/`.

El modelo final se guarda en `artifacts/models/centralized/unsw_nb15_baseline.keras` y las predicciones en `artifacts/predictions/centralized_predictions.csv`, con columnas `real`, `probability` y `prediction`. Para inferencia reproducible se requiere también el preprocesador y el umbral: cargar el `.keras` por sí solo no reconstruye toda la preparación de datos ni aplica automáticamente el umbral seleccionado.

En las evidencias revisadas, el SHA-256 de `best_config.json` coincide con `best_config_sha256` de `train/metrics.json`:

```text
a2f9f057d124c3eef8f4f5a5ebaeb5af7141d5af5b42390ebc1067e5334e4be5
```

Este hash vincula los bytes del JSON con el reporte; no sustituye la comprobación del modelo ni de los datos originales.

También existe `tuning/test_metrics.json`, generado por la ruta opcional `EVALUATE_TEST=True` del selector, con resultados coincidentes. Para el flujo documentado se mantiene **EVALUATE_TEST=False** y se utiliza `train/metrics.json` como reporte final. No es necesario volver a evaluar TEST desde el tuning.

El log histórico `logs/re2_1/train_centralized.log` coincide numéricamente con las métricas, pero al final enumera antiguas rutas sin el subdirectorio `train/`. Las rutas vigentes son las de este documento y del código; una nueva ejecución del script de validación actualiza el log.

## 9. Reproducción y validación por el usuario

Ejecutar desde la raíz del repositorio, en Bash (por ejemplo, Linux o WSL).

### 9.1 Preparar el entorno y los datos

Activar el entorno existente, adaptando la ruta si es necesario:

```bash
source env_federado_tff/bin/activate
python -c "import tensorflow, numpy, pandas, sklearn, joblib, matplotlib; print('Importaciones disponibles')"
```

Colocar los dos CSV oficiales en `data/raw/unsw_nb15/`. Después ejecutar:

```bash
python src/data/inspect_unsw.py
python src/data/preprocess_unsw.py
```

Comprobar los tamaños y las 192 características esperadas. El script de validación presupone que ya existen `train.npz`, `validation.npz` y `test.npz`; no descarga ni preprocesa los datos.

### 9.2 Ejecutar `validate_re2_1.sh`

**El usuario puede ejecutar el siguiente script para reproducir el flujo y validar el resultado del RE2.1:**

```bash
bash scripts/validate_re2_1.sh
```

Antes de ejecutarlo, mantener en `src/models/centralized/tune_centralized.py`:

```python
PLOT_ONLY = False
EVALUATE_TEST = False
```

El script comprueba esos valores y la presencia de las tres particiones. Luego:

1. Ejecuta el tuning por CV, selecciona la configuración y genera automáticamente las figuras PNG/PDF.
2. Entrena el baseline final sobre TRAIN + VALIDATION y lo evalúa en TEST.

Guarda la salida en `logs/re2_1/tune_centralized.log` y `logs/re2_1/train_centralized.log`. No requiere cambiar manualmente `PLOT_ONLY` a mitad de la ejecución. Usa `set -euo pipefail` para detenerse ante errores, incluso cuando la salida pasa por `tee`.

**Esta ejecución vuelve a entrenar y sobrescribe las salidas y logs correspondientes.** Antes de reproducir una corrida, conservar sus evidencias si se necesita mantener el resultado anterior. No utilizar sucesivas evaluaciones de TEST para elegir configuraciones.

La finalización del script confirma que los comandos terminaron correctamente; **no comprueba automáticamente el indicador recall ≥ 0.80**. Para validar el resultado, revisar:

- `results/re2/centralized/train/metrics.json`: evaluación sobre `test`, ambas clases en la matriz y presencia de las cuatro métricas requeridas.
- `recall >= 0.80` para la clase maliciosa y matriz cuya suma sea 82,332 en el dataset documentado.
- Configuración, umbral y hash coherentes con el JSON seleccionado.
- Existencia del `.keras`, preprocesador y CSV de predicciones correspondientes a la corrida.

### 9.3 Ejecución manual equivalente

Con ambos flags en `False`, se pueden ejecutar las etapas por separado:

```bash
python src/models/centralized/tune_centralized.py
python src/models/centralized/train_centralized.py
```

Si solo se desea reproducir el entrenamiento final con la configuración ya guardada, ejecutar únicamente el segundo comando, una vez preparados los datos. Esto evita repetir la selección, pero vuelve a generar el modelo y las métricas finales.

Para regenerar exclusivamente las figuras de CV existentes, establecer temporalmente `PLOT_ONLY=True` y ejecutar el selector. Este modo lee los CSV y el JSON guardados, no reentrena ni modifica `best_config.json`. Restablecerlo a `False` antes de usar `validate_re2_1.sh`.

### 9.4 Reproducibilidad numérica

Las semillas, configuración, particiones y preprocesador permiten reconstruir el procedimiento. Las versiones de dependencias, hardware y operaciones numéricas pueden producir variaciones; el código no fuerza determinismo completo de todas las operaciones. Deben conservarse las métricas de cada corrida y comprobar el cumplimiento con sus resultados reales, sin exigir igualdad decimal exacta ni asumir que cualquier variación será necesariamente pequeña.

## 10. Verificación de los indicadores del RE2.1

| Indicador de la tesis | Evidencia registrada | Resultado |
|---|---|---|
| Evaluación sobre prueba separada y exactamente dos clases | Código de selección/entrenamiento; `evaluation_dataset: test`; matriz 2 × 2 | Cumplido en el procedimiento documentado |
| Accuracy, precision, recall y F1-score, además de matriz de confusión | `train/metrics.json` y log de entrenamiento | Cumplido |
| Recall malicioso ≥ 80 % sobre prueba | Recall = 92.06 % | Cumplido |

Los resultados versionados respaldan el cumplimiento de los indicadores. La verificación completa del entregable **modelo entrenado** requiere disponer del `.keras` y de los datos/preprocesador asociados, que no están incluidos en Git. Su presencia no debe darse por comprobada únicamente porque el código contenga una llamada a `model.save`.

## 11. Relación con RE2.2 y comparación del RE2.3

El baseline aporta arquitectura, preprocesamiento, configuración congelada y métricas de referencia. Para comparar los enfoques se debe conservar el mismo TEST, representación y orden de características, codificación de clases, regla de decisión y cálculo de métricas.

**El centralizado documentado entrena con TRAIN + VALIDATION.** Para una comparación con igual disponibilidad de ejemplos, las particiones locales del entrenamiento federado final deben cubrir ese mismo conjunto de desarrollo, sin duplicar registros entre clientes. No corresponde exigir que el federado use únicamente TRAIN y, al mismo tiempo, afirmar igualdad de datos de entrenamiento con este baseline.

Si el procedimiento federado reserva VALIDATION para seleccionar rondas o hiperparámetros, se debe distinguir esa fase de selección del entrenamiento final. Si se compara un modelo federado entrenado solo con TRAIN, hay que explicitar la diferencia o generar un baseline centralizado con el mismo presupuesto de datos. TEST nunca debe incorporarse a los clientes ni usarse para ajustar la configuración.

Utilizar la misma arquitectura y el umbral **0.80** permite controlar esas variables, pero no basta por sí solo para demostrar equivalencia experimental: deben documentarse también los datos usados, particionado, rondas, épocas locales, semillas y criterio de selección del modelo global. Este documento no certifica por sí mismo el cumplimiento del RE2.2 o RE2.3.

## 12. Estado y límites de la revisión

**RE2.1: implementado, con indicadores satisfechos según las métricas versionadas.**

La revisión contrastó documentación, scripts centralizados, preprocesamiento, configuración, tablas de CV, métricas, historial y log. Se corrigieron las rutas base de los scripts alojados en `src/models/centralized/` y se actualizó el script de validación para usar la generación automática de figuras.

No se repitió el entrenamiento completo durante esta revisión: el repositorio no incluye los datos originales/procesados ni el modelo final. Los resultados numéricos aquí presentados son los de la corrida guardada. Permanecen explícitas la limitación del preprocesamiento previo a CV y la diferencia de FPR entre CV y TEST.
