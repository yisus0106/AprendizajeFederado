# RE1.3 — Prototipo funcional del mecanismo de aprendizaje federado

## 1. Propósito

Este documento describe la implementación, configuración, ejecución y verificación técnica del **RE1.3 — Prototipo funcional del mecanismo de aprendizaje federado**.

El prototipo implementa un mecanismo base de aprendizaje federado bajo una arquitectura cliente-servidor. Su finalidad es comprobar que múltiples clientes simulados pueden participar en rondas de entrenamiento colaborativo y contribuir a la generación sucesiva de un modelo global sin centralizar lógicamente sus datos locales de entrenamiento.

Este resultado corresponde al mecanismo federado base del proyecto. Por tanto, en esta etapa no se evalúa todavía un sistema de detección de intrusiones ni se utilizan datos reales de tráfico de red para medir desempeño predictivo.

---

## 2. Alcance del prototipo

El prototipo desarrollado para RE1.3 permite:

- inicializar un estado global del modelo;
- simular múltiples clientes federados;
- asignar una partición local independiente a cada cliente;
- convertir cada partición en un `tf.data.Dataset`;
- ejecutar entrenamiento local;
- registrar la cantidad de ejemplos utilizados;
- coordinar el proceso mediante TensorFlow Federated;
- agregar las actualizaciones mediante Federated Averaging (FedAvg);
- generar un nuevo estado global después de cada ronda;
- utilizar el estado actualizado como entrada de la ronda siguiente;
- ejecutar al menos cinco rondas consecutivas;
- registrar métricas y estados de ejecución;
- verificar la localidad lógica de los datos mediante las ubicaciones federadas `CLIENTS` y `SERVER`.

El prototipo representa un escenario experimental controlado. Los clientes son participantes simulados dentro de TensorFlow Federated y no dispositivos físicos independientes.

---

## 3. Arquitectura implementada

El mecanismo utiliza una arquitectura centralizada de aprendizaje federado bajo un esquema cliente-servidor.

```text
                         MODELO GLOBAL
                              │
                              ▼
                    ┌───────────────────┐
                    │ Servidor agregador│
                    │                   │
                    │      FedAvg       │
                    └─────────┬─────────┘
                              │
                    parámetros globales
                   ┌──────────┴──────────┐
                   │                     │
                   ▼                     ▼
             ┌────────────┐        ┌────────────┐
             │ Cliente 01 │        │ Cliente 02 │
             │            │        │            │
             │ Datos      │        │ Datos      │
             │ locales    │        │ locales    │
             │            │        │            │
             │ Entrenam.  │        │ Entrenam.  │
             │ local      │        │ local      │
             └─────┬──────┘        └─────┬──────┘
                   │                     │
                   └──── actualizaciones ┘
                              │
                              ▼
                            FedAvg
                              │
                              ▼
                     Nuevo modelo global
```

Las responsabilidades se distribuyen de la siguiente forma:

| Componente | Responsabilidad |
|---|---|
| Cliente federado | Participar en las rondas federadas |
| Gestor de datos locales | Mantener y proporcionar exclusivamente la partición asignada |
| Entrenamiento local | Utilizar el estado global y entrenar sobre datos locales |
| Servidor agregador | Mantener y coordinar el estado global |
| FedAvg | Agregar las actualizaciones de los clientes |
| Modelo global | Representar el estado consolidado de cada ronda |

---

## 4. Entorno utilizado

El prototipo fue desarrollado sobre WSL2 utilizando Ubuntu como entorno Linux.

La configuración validada es:

| Componente | Versión |
|---|---|
| Ubuntu | 22.04.3 LTS |
| Plataforma | WSL2 |
| Python | 3.11.15 |
| TensorFlow | 2.14.1 |
| TensorFlow Federated | 0.87.0 |
| NumPy | 1.25.2 |
| SciPy | 1.9.3 |

El desarrollo y depuración del código se realiza principalmente mediante Visual Studio Code conectado al entorno WSL2.

---

## 5. Dependencias

Las dependencias utilizadas por el entorno se encuentran registradas en:

```text
requirements/
├── requirements-lock.txt
└── requirements_tff_cpu.txt
```

`requirements-lock.txt` registra las versiones del entorno funcional utilizado durante el desarrollo.

`requirements_tff_cpu.txt` mantiene la configuración principal asociada al entorno de TensorFlow Federated para ejecución sobre CPU.

> **Nota de reproducibilidad:** el archivo de dependencias registra el entorno actualmente funcional. Antes del cierre final del proyecto deberá verificarse también la instalación completa desde un entorno limpio para confirmar que todas las dependencias pueden reconstruirse únicamente a partir de los archivos registrados.

---

## 6. Preparación del entorno

### 6.1 Crear el entorno virtual

Desde Linux/WSL2:

```bash
python3 -m venv env_federado_tff
```

Activar:

```bash
source env_federado_tff/bin/activate
```

La terminal debe indicar que el entorno está activo:

```text
(env_federado_tff)
```

### 6.2 Instalar dependencias

```bash
pip install -r requirements/requirements-lock.txt
```

### 6.3 Verificar el entorno

Ejecutar:

```bash
python tests/verify_environment.py
```

Una ejecución validada produjo:

```text
=== VERIFICACION DEL ENTORNO FEDERADO ===
Python: 3.11.15
Sistema: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
TensorFlow: 2.14.1
TensorFlow Federated: 0.87.0
NumPy: 1.25.2
SciPy: 1.9.3
Estado: ENTORNO OPERATIVO
```

El registro correspondiente se encuentra en:

```text
logs/environment_verification.log
```

---

## 7. Organización de RE1.3

Los principales componentes utilizados por RE1.3 se encuentran organizados de la siguiente manera:

```text
AprendizajeFederado/
│
├── docs/
│   └── re1_3_prototipo_federado.md
│
├── requirements/
│   ├── requirements-lock.txt
│   └── requirements_tff_cpu.txt
│
├── src/
│   └── federated/
│       ├── __init__.py
│       ├── client_data.py
│       ├── model.py
│       ├── server.py
│       └── training.py
│
├── tests/
│   ├── __init__.py
│   ├── verify_environment.py
│   ├── test_client_data.py
│   ├── test_server.py
│   ├── test_federated_round.py
│   ├── test_federated_training.py
│   └── test_data_locality.py
│
├── scripts/
│   └── validate_re1_3.sh
│
└── logs/
    ├── environment_verification.log
    ├── client_data_verification.log
    ├── server_verification.log
    ├── federated_training.log
    ├── data_locality_verification.log
    └── re1_3_validation.log
```

---

## 8. Implementación de los clientes federados

La gestión de los datos locales se encuentra implementada en:

```text
src/federated/client_data.py
```

El componente principal es `ClientPartition`, encargado de representar la partición perteneciente a un cliente federado simulado.

Cada partición contiene:

```text
client_id
features
labels
num_examples
```

La implementación valida que:

- las características tengan una estructura bidimensional;
- las etiquetas tengan una estructura compatible;
- la partición no se encuentre vacía;
- características y etiquetas tengan la misma cantidad de ejemplos;
- no existan valores numéricos no finitos.

Cada cliente puede convertir posteriormente su partición en un `tf.data.Dataset` mediante el método:

```python
to_tf_dataset(...)
```

Este dataset es el utilizado durante el entrenamiento local.

---

## 9. Datos utilizados para RE1.3

RE1.3 utiliza datos sintéticos pequeños creados exclusivamente para comprobar el funcionamiento del mecanismo federado.

La configuración utilizada es:

```text
Número de clientes:       2
Ejemplos por cliente:    12
Ejemplos totales:        24
Características:          2
Etiqueta:                 1
Problema: clasificación binaria sintética
```

La distribución es:

```text
24 ejemplos
│
├── client_01 → 12 ejemplos
│
└── client_02 → 12 ejemplos
```

Los registros de ambos clientes se mantienen como particiones diferenciadas.

Estos datos no representan tráfico benigno o malicioso y sus métricas no deben interpretarse como resultados de detección de intrusiones.

---

## 10. Prueba de los clientes federados

La prueba se ejecuta mediante:

```bash
python -m tests.test_client_data
```

La prueba verifica:

- creación de exactamente dos clientes;
- disponibilidad de doce ejemplos por cliente;
- lectura completa de cada partición;
- estructura correcta de características y etiquetas;
- rechazo controlado de una partición inválida.

Salida de referencia:

```text
=== PRUEBA DE CLIENTES FEDERADOS ===
Clientes creados: 2
client_01: 12 ejemplos locales - OK
client_02: 12 ejemplos locales - OK
Entrada inválida rechazada correctamente.
Error controlado: invalid_client: features y labels deben tener la misma cantidad de registros.
Estado: CLIENTES LOCALES OPERATIVOS
```

La evidencia puede almacenarse mediante:

```bash
python -m tests.test_client_data \
  | tee logs/client_data_verification.log
```

---

## 11. Modelo mínimo utilizado

El modelo utilizado para validar RE1.3 se encuentra en:

```text
src/federated/model.py
```

Su arquitectura es deliberadamente mínima:

```text
2 características
       │
       ▼
Dense(1, activation="sigmoid")
       │
       ▼
salida binaria
```

El modelo dispone de dos conjuntos de parámetros entrenables:

```text
Pesos → shape (2, 1)
Bias  → shape (1,)
```

Este modelo no constituye el modelo definitivo de detección de intrusiones.

Su finalidad es únicamente disponer de una estructura entrenable que permita comprobar:

- inicialización global;
- distribución a los clientes;
- entrenamiento local;
- modificación de parámetros;
- agregación global;
- generación de estados consecutivos.

---

## 12. Servidor agregador

La construcción del proceso federado se encuentra en:

```text
src/federated/server.py
```

El proceso utiliza:

```python
tff.learning.algorithms.build_weighted_fed_avg(...)
```

La configuración incluye:

- función de construcción del modelo;
- optimizador del cliente;
- optimizador del servidor;
- proceso de agregación ponderada mediante FedAvg.

El servidor mantiene el estado global utilizado como punto de partida de las rondas federadas.

---

## 13. Prueba del servidor y FedAvg

La prueba se ejecuta mediante:

```bash
python -m tests.test_server
```

Una ejecución validada produjo:

```text
=== PRUEBA DEL SERVIDOR FEDERADO ===
Proceso FedAvg construido correctamente.
Estado global inicializado correctamente.
Variables entrenables del modelo global: 2
Parametro global 1: shape=(2, 1) - OK
Parametro global 2: shape=(1,) - OK
Estado: SERVIDOR Y FEDAVG OPERATIVOS
```

Esta prueba comprueba la construcción del proceso y la inicialización de un estado global válido.

La ejecución se puede registrar mediante:

```bash
python -m tests.test_server \
  | tee logs/server_verification.log
```

---

## 14. Integración del mecanismo federado

La integración entre clientes y servidor se encuentra implementada en:

```text
src/federated/training.py
```

El flujo de una ronda corresponde a:

```text
Estado global wr
       │
       ▼
Distribución a clientes
       │
   ┌───┴───┐
   ▼       ▼
Cliente 1  Cliente 2
   │       │
   ▼       ▼
Entrenamiento local
   │       │
   └───┬───┘
       ▼
Actualizaciones
       │
       ▼
     FedAvg
       │
       ▼
Estado global wr+1
```

El estado `wr+1` obtenido al finalizar una ronda pasa a convertirse en el estado de entrada de la ronda siguiente.

---

## 15. Prueba de una ronda federada

La integración básica se verifica mediante:

```bash
python -m tests.test_federated_round
```

Una ejecución produjo:

```text
=== PRUEBA DE UNA RONDA FEDERADA ===
Clientes participantes: 2
client_01: 12 ejemplos locales
client_02: 12 ejemplos locales
Parametro 1: actualizado=True
Parametro 2: actualizado=True
Parametros modificados: 2/2
Metricas de la ronda:
OrderedDict([
    ...
])
Ronda federada completada correctamente.
Estado: INTEGRACION FEDERADA OPERATIVA
```

Durante esta prueba participaron los dos clientes y ambos parámetros entrenables del estado global presentaron modificaciones después de la ronda.

---

## 16. Ejecución de cinco rondas consecutivas

La ejecución principal de RE1.3 se verifica mediante:

```bash
python -m tests.test_federated_training
```

La configuración utilizada es:

| Parámetro | Valor |
|---|---:|
| Clientes | 2 |
| Ejemplos por cliente | 12 |
| Ejemplos procesados por ronda | 24 |
| Batch size | 4 |
| Épocas locales | 1 |
| Rondas | 5 |

Una ejecución de referencia produjo:

| Ronda | Loss | Accuracy | Ejemplos | Parámetros actualizados |
|---:|---:|---:|---:|---:|
| 1 | 0.766523 | 0.708333 | 24 | 2/2 |
| 2 | 0.724411 | 0.708333 | 24 | 2/2 |
| 3 | 0.686370 | 0.708333 | 24 | 2/2 |
| 4 | 0.652014 | 0.708333 | 24 | 2/2 |
| 5 | 0.620976 | 0.750000 | 24 | 2/2 |

Salida:

```text
Las 5 rondas produjeron una actualizacion del modelo global.
Estado: ENTRENAMIENTO FEDERADO OPERATIVO
```

La evidencia se registra mediante:

```bash
python -m tests.test_federated_training \
  | tee logs/federated_training.log
```

Los valores de `loss` y `accuracy` corresponden a una ejecución de referencia del modelo sintético. Su finalidad es demostrar que el proceso de entrenamiento se ejecuta y genera métricas; no representan desempeño de detección de intrusiones.

Las condiciones funcionales principales que deben reproducirse son:

```text
2 clientes participantes
5 rondas completadas
24 ejemplos utilizados por ronda
parámetros globales modificados en cada ronda
```

---

## 17. Verificación de localidad lógica de datos

La localidad lógica se verifica mediante:

```bash
python -m tests.test_data_locality
```

La prueba comprueba primero la existencia de dos clientes independientes y valida la estructura de sus particiones:

```text
client_01: 12 ejemplos locales | X=(12, 2) | y=(12, 1)
client_02: 12 ejemplos locales | X=(12, 2) | y=(12, 1)
```

Posteriormente verifica que no existan ejemplos compartidos entre ambas particiones:

```text
Ejemplos compartidos entre clientes: 0
Ejemplos únicos totales: 24
```

La prueba analiza además la firma del proceso de TensorFlow Federated.

La firma incluye conceptualmente:

```text
global_model_weights ... @SERVER
client_data ... @CLIENTS
```

Esto permite verificar la siguiente separación lógica:

```text
Estado global
     │
     ▼
   SERVER


Datos de entrenamiento
     │
     ▼
   CLIENTS
```

Una ejecución validada produjo:

```text
=== VERIFICACION DE LOCALIDAD LOGICA DE DATOS ===
Clientes verificados: 2
client_01: 12 ejemplos locales | X=(12, 2) | y=(12, 1)
client_02: 12 ejemplos locales | X=(12, 2) | y=(12, 1)

Separacion de particiones:
Ejemplos compartidos entre clientes: 0 - OK
Ejemplos unicos totales: 24 - OK

Semantica federada:
Estado global ubicado en SERVER - OK
Datos de entrenamiento ubicados en CLIENTS - OK

Resumen:
Clientes independientes: 2
Ejemplos totales: 24
Ejemplos compartidos: 0
Estado global: SERVER
Datos de entrenamiento: CLIENTS

Estado: LOCALIDAD LOGICA DE DATOS FEDERADOS VERIFICADA
```

Esta comprobación corresponde a la separación lógica establecida por TensorFlow Federated dentro del entorno simulado.

No constituye una captura o inspección de tráfico de red entre máquinas físicamente independientes.

---

## 18. Validación integral de RE1.3

Las principales comprobaciones del resultado pueden ejecutarse mediante:

```bash
./scripts/validate_re1_3.sh
```

El script ejecuta secuencialmente las verificaciones de:

```text
Entorno
   ↓
Clientes
   ↓
Servidor y FedAvg
   ↓
Entrenamiento federado
   ↓
Localidad lógica
```

Para almacenar la evidencia:

```bash
./scripts/validate_re1_3.sh \
  | tee logs/re1_3_validation.log
```

Una ejecución satisfactoria debe finalizar sin errores de aserción ni excepciones.

---

## 19. Evidencias

Los registros principales de RE1.3 se encuentran en:

```text
logs/
├── environment_verification.log
├── client_data_verification.log
├── server_verification.log
├── federated_training.log
├── data_locality_verification.log
└── re1_3_validation.log
```

Estos archivos constituyen evidencia reproducible de las principales verificaciones realizadas durante la construcción incremental del prototipo.

---

## 20. Verificación del resultado

Los resultados obtenidos se contrastan con los indicadores definidos para RE1.3.

| Condición de verificación | Resultado |
|---|---|
| Servidor agregador operativo | 1 servidor |
| Clientes federados operativos | 2 clientes |
| Rondas consecutivas | 5/5 |
| Clientes que utilizan su partición local | 2/2 |
| Ejemplos compartidos entre particiones de prueba | 0 |
| Estado global | `SERVER` |
| Datos de entrenamiento | `CLIENTS` |
| Rondas que actualizaron el modelo global | 5/5 = 100 % |

Por tanto, el prototipo demuestra funcionalmente la coordinación de múltiples clientes simulados, la ejecución de rondas de entrenamiento federado, la agregación global mediante FedAvg y el mantenimiento lógico de los datos de entrenamiento en los clientes.

---

## 21. Relación con el desarrollo iterativo-incremental

La implementación de RE1.3 fue realizada mediante incrementos funcionales sucesivos.

```text
Incremento 1
Configuración y verificación del entorno
              ↓
Incremento 2
Construcción de clientes locales
              ↓
Incremento 3
Servidor y configuración FedAvg
              ↓
Incremento 4
Integración de una ronda federada
              ↓
Incremento 5
Cinco rondas consecutivas
              ↓
Incremento 6
Verificación de localidad lógica
```

Cada incremento fue verificado antes de incorporar el siguiente componente, evitando integrar simultáneamente elementos cuya operación individual no hubiera sido comprobada previamente.

---

## 22. Limitaciones

El prototipo presenta las siguientes limitaciones:

- los clientes son simulados dentro de TensorFlow Federated;
- no se utilizan máquinas físicamente independientes;
- la verificación de localidad corresponde a la semántica lógica `CLIENTS`/`SERVER`;
- no se inspecciona tráfico de red mediante captura de paquetes;
- se utilizan datos sintéticos exclusivamente para validar el mecanismo;
- el modelo utilizado es deliberadamente mínimo;
- no se evalúa desempeño real de detección;
- no se estudian fallos o desconexiones de participantes;
- no se evalúan todavía escenarios heterogéneos o non-IID;
- no se implementan mecanismos avanzados de agregación;
- no se incorporan mecanismos criptográficos adicionales de privacidad o robustez.

Estas limitaciones corresponden al alcance del prototipo base y no invalidan su propósito principal: demostrar el funcionamiento del mecanismo de aprendizaje federado antes de integrarlo con componentes posteriores del proyecto.

---

## 23. Resultado final

RE1.3 proporciona un mecanismo federado funcional y modular compuesto por:

```text
Clientes simulados
        +
Datos locales independientes
        +
Modelo entrenable
        +
Servidor agregador
        +
FedAvg
        +
Rondas federadas
        +
Logs y pruebas
```

Este mecanismo constituye la base técnica reutilizable para los siguientes incrementos del proyecto.