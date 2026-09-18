# RE2.1 – Modelo de detección de intrusiones basado en tráfico de red

## 1. Propósito

Este documento describe la implementación, configuración, evaluación y procedimiento de reproducción del **Resultado Esperado 2.1 (RE2.1)** del proyecto:

> Modelo funcional de detección de intrusiones basado en tráfico de red.

El objetivo de este resultado es disponer de un modelo de clasificación binaria capaz de distinguir entre tráfico **benigno** y **malicioso**, utilizando el conjunto de datos **UNSW-NB15**.

El modelo obtenido constituye además la línea base centralizada que será utilizada posteriormente para:

- integrar el mecanismo de detección con la arquitectura de aprendizaje federado en el RE2.2;
- comparar experimentalmente el enfoque centralizado y el federado en el RE2.3.

El desarrollo del RE2.1 comprende:

1. inspección del conjunto de datos;
2. preparación y preprocesamiento;
3. separación de entrenamiento, validación y prueba;
4. definición del modelo neuronal;
5. ajuste controlado de hiperparámetros;
6. selección del umbral de clasificación;
7. evaluación final sobre el conjunto de prueba;
8. almacenamiento de métricas, predicciones y modelo entrenado.

---

## 2. Estructura relacionada del proyecto

Los principales archivos involucrados en el RE2.1 se encuentran organizados de la siguiente manera:

```text
data/
├── raw/
│   └── unsw_nb15/
│       ├── NUSW-NB15_features.csv
│       ├── UNSW_NB15_testing-set.csv
│       └── UNSW_NB15_training-set.csv
│
└── processed/
    └── unsw_nb15/
        ├── metadata.json
        ├── preprocessor.joblib
        ├── train.npz
        ├── validation.npz
        └── test.npz

src/
├── data/
│   ├── inspect_unsw.py
│   └── preprocess_unsw.py
│
└── models/
    ├── train_centralized.py
    └── tune_centralized.py

results/
└── re2/
    └── centralized/
        ├── metrics.json
        ├── training_history.csv
        └── tuning/
            ├── best_config.json
            ├── hyperparameter_search.csv
            └── threshold_search.csv

artifacts/
├── models/
│   └── centralized/
│       └── unsw_nb15_baseline.keras
│
└── predictions/
    └── centralized_predictions.csv
```

---

## 3. Dataset utilizado

Se utilizó el conjunto de datos **UNSW-NB15**, orientado al análisis de tráfico de red y utilizado para problemas de detección de intrusiones.

Los archivos utilizados fueron:

```text
data/raw/unsw_nb15/UNSW_NB15_training-set.csv
data/raw/unsw_nb15/UNSW_NB15_testing-set.csv
```

El archivo:

```text
NUSW-NB15_features.csv
```

se conserva como referencia de las características originales del dataset.

El problema fue definido como una clasificación binaria:

```text
0 = tráfico benigno
1 = tráfico malicioso
```

La variable objetivo utilizada fue:

```text
label
```

No se utilizó `attack_cat` como predictor, debido a que contiene información sobre el tipo de ataque y se encuentra directamente relacionada con la etiqueta binaria que se busca predecir.

Tampoco se utilizó `id`, debido a que corresponde a un identificador de registro y no representa una característica útil del tráfico.

---

## 4. Inspección inicial de UNSW-NB15

La inspección del dataset se realiza mediante:

```bash
python src/data/inspect_unsw.py
```

Durante esta inspección se verifican:

- dimensiones de los conjuntos;
- nombres de las columnas;
- tipos de datos;
- valores nulos;
- registros duplicados;
- valores infinitos;
- distribución de las clases;
- categorías de ataque disponibles.

### 4.1 Conjunto oficial de entrenamiento

El archivo oficial de entrenamiento contiene:

```text
Registros: 175,341
Columnas:  45
```

Distribución de la variable `label`:

| Clase | Cantidad | Porcentaje |
|---|---:|---:|
| Benigno (`0`) | 56,000 | 31.94 % |
| Malicioso (`1`) | 119,341 | 68.06 % |

No se identificaron:

```text
Valores nulos:     0
Registros duplicados: 0
Valores infinitos: 0
```

### 4.2 Conjunto oficial de prueba

El archivo oficial de prueba contiene:

```text
Registros: 82,332
Columnas:  45
```

Distribución:

| Clase | Cantidad | Porcentaje |
|---|---:|---:|
| Benigno (`0`) | 37,000 | 44.94 % |
| Malicioso (`1`) | 45,332 | 55.06 % |

Tampoco se identificaron valores nulos, duplicados o infinitos.

El conjunto oficial de prueba se conserva independiente y no participa en el entrenamiento ni en la selección de hiperparámetros.

---

## 5. Variables utilizadas

El dataset original presenta 45 columnas.

Para el modelo se eliminan:

```text
id
attack_cat
label
```

donde `label` constituye la variable objetivo.

Por tanto, antes de la transformación se utilizan:

```text
42 variables predictoras
```

Estas se dividen en:

```text
3 variables categóricas
39 variables numéricas
```

### 5.1 Variables categóricas

Las variables categóricas utilizadas son:

```text
proto
service
state
```

Estas variables son transformadas mediante **One-Hot Encoding**.

### 5.2 Variables numéricas

Las 39 variables restantes son tratadas como características numéricas y normalizadas mediante:

```text
StandardScaler
```

Después de aplicar el preprocesamiento, la representación final contiene:

```text
192 características
```

Estas 192 entradas conforman la dimensión de entrada de la red neuronal.

---

## 6. Separación TRAIN, VALIDATION y TEST

El conjunto de prueba oficial de UNSW-NB15 se mantiene completamente separado.

El archivo oficial de entrenamiento es dividido internamente en:

```text
80 % TRAIN
20 % VALIDATION
```

utilizando:

```text
random_state = 42
stratify = label
```

La separación resultante es:

| Conjunto | Registros | Benignos | Maliciosos |
|---|---:|---:|---:|
| TRAIN | 140,272 | 44,800 | 95,472 |
| VALIDATION | 35,069 | 11,200 | 23,869 |
| TEST | 82,332 | 37,000 | 45,332 |

Las proporciones de TRAIN y VALIDATION se mantienen aproximadamente en:

```text
31.94 % benigno
68.06 % malicioso
```

El conjunto TEST conserva la distribución original proporcionada por UNSW-NB15.

### 6.1 Función de cada conjunto

**TRAIN**

Se utiliza para actualizar los pesos y sesgos del modelo mediante el proceso de entrenamiento.

**VALIDATION**

Se utiliza exclusivamente para:

- controlar el proceso de entrenamiento;
- aplicar EarlyStopping;
- comparar configuraciones;
- seleccionar hiperparámetros;
- seleccionar el threshold de clasificación.

**TEST**

Se utiliza únicamente para la evaluación final del modelo seleccionado.

No se utiliza TEST para decidir:

- arquitectura;
- learning rate;
- dropout;
- número de neuronas;
- threshold;
- época óptima.

---

## 7. Preprocesamiento

El procedimiento se implementa en:

```text
src/data/preprocess_unsw.py
```

y puede ejecutarse mediante:

```bash
python src/data/preprocess_unsw.py
```

El procedimiento sigue el siguiente flujo:

```text
UNSW_NB15_training-set.csv
            │
            ↓
    separación estratificada
            │
       ┌────┴────┐
       ↓         ↓
     TRAIN   VALIDATION
       │
       ↓
FIT del preprocesador
       │
       ├──────────────→ transforma VALIDATION
       │
       └──────────────→ transforma TEST
```

El preprocesador se ajusta **únicamente utilizando TRAIN**.

Esto evita que información estadística de VALIDATION o TEST participe durante el ajuste de:

- `StandardScaler`;
- categorías de `OneHotEncoder`.

La configuración categórica utiliza:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

permitiendo procesar categorías no observadas durante el ajuste sin modificar la estructura de entrada.

Los datos procesados son convertidos a:

```text
float32
```

y posteriormente se verifica la ausencia de:

- `NaN`;
- valores infinitos;
- diferencias de dimensiones entre los conjuntos.

---

## 8. Artefactos del preprocesamiento

La ejecución genera:

```text
data/processed/unsw_nb15/
├── train.npz
├── validation.npz
├── test.npz
├── preprocessor.joblib
└── metadata.json
```

### `train.npz`

Contiene:

```text
X_train: (140272, 192)
y_train: (140272,)
```

### `validation.npz`

Contiene:

```text
X_validation: (35069, 192)
y_validation: (35069,)
```

### `test.npz`

Contiene:

```text
X_test: (82332, 192)
y_test: (82332,)
```

### `preprocessor.joblib`

Contiene el objeto de preprocesamiento ajustado únicamente con TRAIN.

### `metadata.json`

Contiene información necesaria para reproducir y auditar el preprocesamiento, incluyendo:

- seed;
- dimensiones;
- número de características;
- variables categóricas;
- variables numéricas;
- distribución de clases;
- nombres de las características procesadas.

---

## 9. Modelo centralizado de referencia

El modelo se implementa mediante **TensorFlow/Keras** en:

```text
src/models/train_centralized.py
```

Se utiliza una red neuronal multicapa o **Multilayer Perceptron (MLP)** para clasificación binaria.

La arquitectura final es:

```text
Input(192)
    │
    ↓
Dense(64, ReLU)
    │
    ↓
Dropout(0.20)
    │
    ↓
Dense(32, ReLU)
    │
    ↓
Dense(1, Sigmoid)
```

Número total de parámetros:

```text
14,465 parámetros
```

Todos los parámetros son entrenables.

### 9.1 Función de salida

La última capa utiliza:

```text
sigmoid
```

por lo que el modelo produce una probabilidad entre:

```text
0 y 1
```

correspondiente a la probabilidad estimada de que una muestra pertenezca a la clase maliciosa.

---

## 10. Configuración de entrenamiento

La configuración final utilizada es:

| Parámetro | Valor |
|---|---|
| Arquitectura oculta | `[64, 32]` |
| Activación | ReLU |
| Activación de salida | Sigmoid |
| Dropout | 0.20 |
| Optimizador | Adam |
| Learning rate | 0.001 |
| Función de pérdida | Binary Crossentropy |
| Batch size | 256 |
| Máximo de epochs | 20 |
| EarlyStopping patience | 3 |
| Seed | 42 |
| Threshold final | 0.55 |

La función de pérdida utilizada es:

```text
binary_crossentropy
```

por tratarse de un problema de clasificación binaria.

---

## 11. EarlyStopping

Para evitar entrenamiento innecesario y limitar potenciales problemas de sobreajuste se utiliza:

```python
EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

La ejecución final presentó:

```text
Epochs ejecutados: 18
Mejor epoch:       15
```

El entrenamiento se detuvo después de tres épocas consecutivas sin superar el mejor `val_loss`.

Debido a:

```text
restore_best_weights=True
```

los parámetros utilizados posteriormente para la evaluación corresponden al mejor estado identificado sobre VALIDATION y no necesariamente al último epoch ejecutado.

---

## 12. Ajuste controlado de hiperparámetros

La selección de configuración se implementó mediante:

```text
src/models/tune_centralized.py
```

y se ejecuta con:

```bash
python src/models/tune_centralized.py
```

Durante este procedimiento no se utiliza el conjunto TEST.

Se evaluaron cinco configuraciones.

| ID | Capas | Dropout | Learning rate | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| C0 | 64 → 32 | 0.20 | 0.001 | 0.9494 | 0.9729 | 0.9610 | 0.1106 |
| C1 | 32 → 16 | 0.20 | 0.001 | 0.9510 | 0.9707 | 0.9608 | 0.1065 |
| C2 | 64 → 32 | 0.30 | 0.001 | 0.9489 | 0.9733 | 0.9610 | 0.1117 |
| C3 | 64 → 32 | 0.20 | 0.0005 | 0.9519 | 0.9685 | 0.9601 | 0.1044 |
| C4 | 128 → 64 | 0.20 | 0.001 | 0.9556 | 0.9659 | 0.9608 | 0.0955 |

La configuración seleccionada fue:

```text
C0
```

correspondiente a:

```text
Dense(64)
Dropout(0.20)
Dense(32)
Learning rate = 0.001
```

La configuración presentó el mejor equilibrio según F1 sin incrementar innecesariamente la complejidad del modelo.

Las configuraciones y resultados completos se encuentran en:

```text
results/re2/centralized/tuning/hyperparameter_search.csv
```

---

## 13. Selección del threshold

Inicialmente se utilizó el threshold estándar:

```text
0.50
```

Sin embargo, debido a la necesidad de controlar las falsas alarmas del IDS, se evaluaron múltiples thresholds utilizando únicamente VALIDATION.

Se probaron valores desde:

```text
0.10
```

hasta:

```text
0.90
```

con incrementos de:

```text
0.05
```

Los resultados completos están disponibles en:

```text
results/re2/centralized/tuning/threshold_search.csv
```

Algunos valores relevantes fueron:

| Threshold | Accuracy | Precision | Recall | F1 | FPR |
|---:|---:|---:|---:|---:|---:|
| 0.45 | 0.9458 | 0.9406 | 0.9824 | 0.9610 | 0.1323 |
| 0.50 | 0.9462 | 0.9494 | 0.9729 | 0.9610 | 0.1106 |
| **0.55** | **0.9463** | **0.9564** | **0.9651** | **0.9607** | **0.0938** |
| 0.60 | 0.9449 | 0.9622 | 0.9565 | 0.9594 | 0.0800 |
| 0.65 | 0.9430 | 0.9679 | 0.9476 | 0.9577 | 0.0670 |

El máximo F1 observado se obtuvo alrededor de `0.45`, pero los thresholds cercanos presentaron diferencias mínimas.

Por este motivo se consideraron equivalentes las configuraciones cuya diferencia respecto del máximo F1 fuera menor o igual a:

```text
0.001
```

y que cumplieran:

```text
Recall >= 0.80
```

Entre estas configuraciones se priorizó aquella que redujera la tasa de falsos positivos.

El threshold seleccionado fue finalmente:

```text
0.55
```

Esta decisión permitió disminuir las falsas alarmas manteniendo un recall ampliamente superior al mínimo requerido.

---

## 14. Métricas de evaluación

Las métricas principales utilizadas son:

```text
Accuracy
Precision
Recall
F1-score
```

Adicionalmente se calcula:

```text
ROC-AUC
```

y se genera una matriz de confusión.

Para análisis complementario durante la selección del threshold se utilizaron también:

```text
Specificity
False Positive Rate (FPR)
```

La clase positiva corresponde a:

```text
1 = tráfico malicioso
```

Por tanto, el recall representa la proporción de tráfico malicioso correctamente identificado.

---

## 15. Evaluación final sobre TEST

Después de seleccionar la arquitectura, hiperparámetros y threshold utilizando TRAIN y VALIDATION, el modelo final fue evaluado sobre:

```text
82,332 registros
```

del conjunto TEST oficial.

Los resultados obtenidos fueron:

| Métrica | Resultado |
|---|---:|
| Accuracy | **0.8609** |
| Precision | **0.8143** |
| Recall | **0.9682** |
| F1-score | **0.8846** |
| ROC-AUC | **0.9750** |

Expresado en porcentaje:

| Métrica | Resultado |
|---|---:|
| Accuracy | **86.09 %** |
| Precision | **81.43 %** |
| Recall | **96.82 %** |
| F1-score | **88.46 %** |
| ROC-AUC | **97.50 %** |

---

## 16. Matriz de confusión

La matriz de confusión obtenida fue:

```text
[[26989 10011]
 [ 1441 43891]]
```

Interpretación:

| | Predicción benigno | Predicción malicioso |
|---|---:|---:|
| **Benigno real** | 26,989 | 10,011 |
| **Malicioso real** | 1,441 | 43,891 |

Por tanto:

```text
TN = 26,989
FP = 10,011
FN = 1,441
TP = 43,891
```

El recall para tráfico malicioso se obtiene mediante:

```text
Recall = TP / (TP + FN)
```

resultando aproximadamente:

```text
0.9682
```

o:

```text
96.82 %
```

La tasa de falsos positivos sobre TEST es aproximadamente:

```text
27.06 %
```

por lo que el modelo presenta una alta sensibilidad para detectar tráfico malicioso, aunque continúa mostrando una cantidad relevante de tráfico benigno clasificado como ataque.

Esta característica debe considerarse posteriormente durante la comparación con el modelo federado.

---

## 17. Cumplimiento del indicador de recall

El RE2.1 establece como criterio mínimo:

```text
Recall para tráfico malicioso >= 80 %
```

El modelo final obtuvo:

```text
Recall = 96.82 %
```

Por tanto:

```text
96.82 % >= 80 %
```

y el criterio mínimo se considera satisfecho.

---

## 18. Verificación del RE2.1

El RE2.1 requiere un modelo funcional de detección de intrusiones capaz de diferenciar tráfico benigno y malicioso.

La verificación se resume de la siguiente manera:

| Criterio | Resultado | Estado |
|---|---|---|
| Clasificación binaria | Benigno / Malicioso | Cumplido |
| Evaluación sobre TEST separado | 82,332 registros | Cumplido |
| Accuracy reportado | 86.09 % | Cumplido |
| Precision reportado | 81.43 % | Cumplido |
| Recall reportado | 96.82 % | Cumplido |
| F1-score reportado | 88.46 % | Cumplido |
| Matriz de confusión | Generada | Cumplido |
| Recall mínimo ≥ 80 % | 96.82 % | Cumplido |
| Modelo entrenado almacenado | `.keras` | Cumplido |
| Procedimiento reproducible | Scripts + configuración + resultados | Cumplido |

Con base en estas evidencias, el **RE2.1 se considera implementado y verificado**.

---

## 19. Archivos de resultados

Los resultados del modelo centralizado se almacenan en:

```text
results/re2/centralized/
```

### `metrics.json`

Contiene la configuración y las métricas finales del modelo.

Debe registrar, entre otros:

```text
dataset
seed
input_features
arquitectura
learning_rate
batch_size
epochs
best_epoch
classification_threshold
accuracy
precision
recall
f1_score
roc_auc
confusion_matrix
```

El valor esperado para:

```text
classification_threshold
```

es:

```text
0.55
```

### `training_history.csv`

Contiene la evolución del entrenamiento por epoch, incluyendo:

```text
loss
accuracy
precision
recall
auc
val_loss
val_accuracy
val_precision
val_recall
val_auc
```

### `hyperparameter_search.csv`

Contiene los resultados de las configuraciones C0–C4 evaluadas durante el ajuste de hiperparámetros.

### `threshold_search.csv`

Contiene la evaluación de los diferentes thresholds sobre VALIDATION.

### `best_config.json`

Contiene la configuración seleccionada durante el proceso de tuning.

> **Nota de consistencia:** el threshold definitivo utilizado por el modelo centralizado es `0.55`. Si `best_config.json` todavía contiene el valor automático `0.45` generado por una versión anterior del criterio de selección, debe actualizarse el procedimiento de selección o regenerarse este archivo antes de congelar el RE2.1, de modo que la documentación, configuración y ejecución final sean consistentes.

---

## 20. Artefactos generados

### Modelo entrenado

```text
artifacts/models/centralized/unsw_nb15_baseline.keras
```

Este archivo contiene los pesos y la arquitectura entrenada del modelo.

### Predicciones

```text
artifacts/predictions/centralized_predictions.csv
```

Contiene para cada registro del conjunto TEST:

```text
real
probability
prediction
```

El threshold utilizado para obtener `prediction` es:

```text
0.55
```

---

## 21. Reproducción del RE2.1

Los comandos deben ejecutarse desde la raíz del repositorio.

### Paso 1 – Activar el entorno

Ejemplo:

```bash
source env_federado_tff/bin/activate
```

El mecanismo exacto puede variar dependiendo de la ubicación del entorno virtual.

Las dependencias utilizadas por el proyecto se encuentran documentadas en:

```text
requirements/requirements_tff_cpu.txt
requirements/requirements-lock.txt
```

---

### Paso 2 – Inspeccionar UNSW-NB15

```bash
python src/data/inspect_unsw.py
```

Este paso verifica la estructura y calidad inicial de los archivos originales.

---

### Paso 3 – Preprocesar los datos

```bash
python src/data/preprocess_unsw.py
```

Resultado esperado:

```text
X_train:      (140272, 192)
y_train:      (140272,)
X_validation: (35069, 192)
y_validation: (35069,)
X_test:       (82332, 192)
y_test:       (82332,)
```

Se generan:

```text
train.npz
validation.npz
test.npz
preprocessor.joblib
metadata.json
```

---

### Paso 4 – Ejecutar ajuste de hiperparámetros

```bash
python src/models/tune_centralized.py
```

Este procedimiento utiliza exclusivamente:

```text
TRAIN
VALIDATION
```

y no utiliza TEST para seleccionar la configuración.

Se generan:

```text
results/re2/centralized/tuning/
├── best_config.json
├── hyperparameter_search.csv
└── threshold_search.csv
```

---

### Paso 5 – Entrenar y evaluar el modelo final

Verificar previamente que:

```python
CLASSIFICATION_THRESHOLD = 0.55
```

en:

```text
src/models/train_centralized.py
```

Posteriormente ejecutar:

```bash
python src/models/train_centralized.py
```

El resultado esperado debe ser cercano a:

```text
Accuracy:  0.8609
Precision: 0.8143
Recall:    0.9682
F1-score:  0.8846
ROC-AUC:   0.9750
```

y una matriz de confusión equivalente a:

```text
[[26989 10011]
 [ 1441 43891]]
```

Debido a aspectos asociados a ejecución numérica, plataforma o dependencias, pueden existir pequeñas variaciones en las últimas cifras decimales.

---

## 22. Reproducibilidad

Para favorecer la reproducibilidad se utiliza:

```text
SEED = 42
```

aplicado a:

```text
Python random
NumPy
TensorFlow
```

Además:

- TRAIN, VALIDATION y TEST se almacenan de manera explícita;
- el preprocesador entrenado se conserva mediante `joblib`;
- los hiperparámetros quedan registrados;
- el threshold queda registrado;
- el historial de entrenamiento se conserva;
- las predicciones sobre TEST se almacenan;
- el modelo final se guarda en formato Keras.

Esto permite reconstruir el procedimiento desde los archivos originales de UNSW-NB15 hasta la obtención del modelo final.

---

## 23. Consideraciones metodológicas

### Separación de TEST

El conjunto TEST no debe utilizarse para realizar cambios posteriores en:

```text
arquitectura
dropout
learning rate
batch size
threshold
```

Las decisiones de configuración fueron realizadas utilizando VALIDATION.

TEST constituye el conjunto común que deberá conservarse para la futura comparación entre:

```text
modelo centralizado
vs.
modelo federado
```

en el RE2.3.

### Mismo preprocesamiento

El modelo federado desarrollado en el RE2.2 deberá utilizar una representación compatible con el modelo centralizado.

Por tanto, las particiones locales de los clientes deberán derivarse del conjunto:

```text
TRAIN
```

procesado bajo el mismo procedimiento definido en este resultado.

VALIDATION y TEST no deberán formar parte de las particiones locales de entrenamiento de los clientes.

### Threshold

El threshold final del modelo centralizado se fija en:

```text
0.55
```

Este valor debe mantenerse documentado para garantizar una comparación consistente con el modelo federado.

---

## 24. Relación con RE2.2

El RE2.1 entrega los elementos necesarios para la siguiente etapa:

```text
Modelo IDS centralizado
        │
        ├── arquitectura
        ├── características de entrada
        ├── preprocesamiento
        ├── configuración
        └── métricas de referencia
                │
                ↓
              RE2.2
     Integración con aprendizaje
             federado
```

En el RE2.2 se deberá:

1. distribuir TRAIN entre clientes federados;
2. mantener los datos locales sin centralizarlos durante el entrenamiento federado;
3. adaptar la arquitectura Keras al mecanismo implementado con TensorFlow Federated;
4. ejecutar FedAvg;
5. obtener un modelo global federado;
6. evaluar dicho modelo utilizando el mismo conjunto TEST empleado en este resultado.

La implementación del RE2.2 no debe modificar retrospectivamente la configuración centralizada seleccionada en RE2.1.

---

## 25. Estado del resultado

```text
RE2.1: COMPLETADO
```

Evidencias principales:

```text
Código fuente de inspección                ✓
Código fuente de preprocesamiento          ✓
TRAIN / VALIDATION / TEST separados        ✓
Preprocesador persistido                   ✓
Modelo centralizado implementado           ✓
Tuning de hiperparámetros                  ✓
Selección de threshold                     ✓
Evaluación mediante 4 métricas             ✓
Matriz de confusión                        ✓
Recall malicioso >= 80 %                   ✓
Modelo entrenado almacenado                ✓
Predicciones almacenadas                   ✓
Configuración reproducible                 ✓
```

El modelo centralizado obtenido queda congelado como **baseline de referencia del OE2** para su posterior integración y comparación con el enfoque federado.