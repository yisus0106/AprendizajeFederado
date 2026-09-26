# Arquitectura de Aprendizaje Federado para Detección Colaborativa de Intrusiones

Repositorio de desarrollo de la tesis:

**“Diseño y prototipo de una arquitectura de aprendizaje federado para la detección colaborativa de intrusiones mediante el análisis de tráfico de red en microempresas”.**

Proyecto desarrollado en la especialidad de **Ingeniería Informática de la Pontificia Universidad Católica del Perú (PUCP)**.

## Descripción

Este proyecto explora el aprendizaje federado como mecanismo para construir colaborativamente un modelo de detección de intrusiones, manteniendo los datos de entrenamiento asociados a cada participante.

La propuesta utiliza una arquitectura cliente-servidor: los clientes entrenan sobre sus particiones locales y un servidor agregador combina sus actualizaciones para obtener un modelo global mediante **Federated Averaging (FedAvg)**.

El desarrollo comprende la implementación del mecanismo federado, la construcción de un modelo de detección sobre **UNSW-NB15**, su integración y evaluación experimental, y la elaboración de lineamientos técnicos para orientar su despliegue en microempresas.

## Objetivo general

Diseñar e implementar un prototipo basado en una arquitectura de aprendizaje federado para la detección colaborativa de intrusiones mediante el análisis de tráfico de red en microempresas, permitiendo aprovechar el aprendizaje distribuido sin centralizar los datos locales de entrenamiento.

## Líneas de trabajo

El proyecto se organiza en tres líneas principales:

1. **Arquitectura y aprendizaje federado:** especificación de requisitos, diseño de componentes e implementación del mecanismo de entrenamiento colaborativo.
2. **Detección de intrusiones y evaluación:** preparación de datos, construcción de una línea base centralizada, integración federada y comparación experimental.
3. **Lineamientos técnicos:** definición de condiciones mínimas de configuración, despliegue y gestión de accesos para el contexto de microempresas.

## Estado del proyecto

| Componente | Estado | Avance |
|---|---|---|
| Requisitos y diseño de la arquitectura | Documentados en la tesis | Definición del alcance, componentes y criterios de verificación |
| Mecanismo base de aprendizaje federado | Implementado y verificado | Clientes simulados, servidor agregador, rondas de entrenamiento y FedAvg |
| Preparación de UNSW-NB15 | Implementada | Inspección, transformación y almacenamiento de particiones |
| Modelo centralizado de detección | Implementado y evaluado | Selección de configuración, entrenamiento y evaluación sobre prueba |
| Integración del modelo con el mecanismo federado | En desarrollo | Consolidación del entrenamiento y evaluación del modelo global |
| Comparación centralizado–federado | Pendiente de consolidación | Evaluación comparable y análisis de diferencias |
| Lineamientos técnicos de despliegue y accesos | Pendientes | Recomendaciones para el contexto de aplicación |

Esta tabla refleja el avance del proyecto y se actualizará conforme se consoliden sus componentes y evidencias.

La presencia de archivos experimentales, pruebas o resultados de etapas posteriores no implica que esas etapas estén finalizadas. Su estado se determina a partir de la implementación vigente, su documentación y sus verificaciones.

## Arquitectura general

La arquitectura contempla los siguientes componentes:

| Componente | Responsabilidad |
|---|---|
| Clientes federados | Mantener sus particiones locales y ejecutar el entrenamiento local |
| Servidor agregador | Coordinar las rondas y actualizar el modelo global mediante FedAvg |
| Modelo de detección | Clasificar registros de tráfico como benignos o maliciosos |
| Preparación de datos | Producir entradas compatibles con los modelos y conservar el preprocesamiento |
| Evaluación experimental | Calcular métricas y comparar los modelos bajo condiciones documentadas |

En cada ronda, los clientes reciben el estado del modelo global, realizan entrenamiento local y producen actualizaciones. El servidor agrega esas actualizaciones para construir el siguiente estado global.

La implementación actual utiliza **clientes simulados en un entorno controlado**. La localidad de los datos se representa mediante las ubicaciones lógicas `CLIENTS` y `SERVER` de TensorFlow Federated.

Esta separación lógica no equivale por sí sola a un despliegue en equipos independientes ni acredita mecanismos adicionales como privacidad diferencial o agregación segura.

## Datos y enfoque de evaluación

El conjunto de datos utilizado para la detección de intrusiones es **UNSW-NB15**.

El problema se plantea como clasificación binaria:

| Etiqueta | Clase |
|---|---|
| `0` | Tráfico benigno |
| `1` | Tráfico malicioso |

El flujo experimental contempla:

1. Inspección y preparación de los datos.
2. Selección de la configuración del modelo utilizando datos de desarrollo.
3. Entrenamiento del modelo centralizado de referencia.
4. Integración y entrenamiento del modelo federado.
5. Evaluación sobre un conjunto de prueba separado del entrenamiento.
6. Comparación e interpretación de resultados.

Las métricas principales son **accuracy, precision, recall y F1-score**, acompañadas de una matriz de confusión. Los análisis complementarios y las condiciones exactas de cada experimento se documentan por separado.

Las comparaciones deben mantener un conjunto de prueba común y declarar los datos de entrenamiento, el preprocesamiento, la configuración y los criterios de selección utilizados.

## Tecnologías principales

| Tecnología | Uso |
|---|---|
| Python | Implementación y automatización |
| TensorFlow / Keras | Construcción, entrenamiento e inferencia de modelos |
| TensorFlow Federated | Simulación y coordinación del aprendizaje federado |
| NumPy y pandas | Procesamiento y análisis de datos |
| scikit-learn | Preprocesamiento, particionado y métricas |
| Matplotlib | Generación de figuras |
| joblib | Persistencia del preprocesador |
| Bash | Ejecución de procedimientos de validación |
| Git | Control de versiones |

El desarrollo se realiza en un entorno Linux sobre WSL2.

## Organización del repositorio

| Directorio | Contenido |
|---|---|
| `src/` | Código fuente de preparación de datos, modelos y entrenamiento |
| `tests/` | Pruebas y verificaciones de componentes |
| `scripts/` | Automatización de procedimientos de ejecución y validación |
| `docs/` | Documentación técnica de los resultados desarrollados |
| `requirements/` | Referencias de dependencias del entorno |
| `results/` | Métricas, historiales, tablas y figuras |
| `logs/` | Registros de ejecución |
| `data/` | Datos originales y procesados disponibles localmente |
| `artifacts/` | Modelos entrenados, predicciones y otros artefactos generados |

Dentro del código actualmente documentado:

- `src/data/` contiene la inspección y el preprocesamiento de UNSW-NB15.
- `src/federated_re1_3/` contiene el prototipo base del mecanismo federado.
- `src/models/centralized/` contiene la selección de configuración y el entrenamiento centralizado.

Los directorios de datos y determinados artefactos se generan o preparan localmente y están excluidos del control de versiones. Por ello, pueden no aparecer inmediatamente después de clonar el repositorio.

## Preparación del entorno

Los comandos siguientes se ejecutan desde la raíz del repositorio en Bash, por ejemplo, en Linux o WSL2.

### 1. Obtener el proyecto

```bash
git clone https://github.com/yisus0106/AprendizajeFederado.git
cd AprendizajeFederado
```

Utilizar la rama o revisión que contenga el componente que se desea reproducir. La rama predeterminada puede no incluir los avances más recientes.

### 2. Crear y activar un entorno virtual

El entorno registrado utiliza Python 3.11.

```bash
python3.11 -m venv env_federado_tff
source env_federado_tff/bin/activate
```

Si el entorno ya existe, basta con activarlo utilizando su ruta correspondiente.

### 3. Instalar las dependencias

La referencia principal de dependencias se encuentra en:

```text
requirements/requirements-lock.txt
```

Para intentar reconstruir el entorno registrado:

```bash
python -m pip install -r requirements/requirements-lock.txt
```

El proyecto también conserva `requirements/requirements_tff_cpu.txt` como referencia del entorno federado.

Los archivos incluyen dependencias específicas de plataforma, entre ellas un wheel de JAX para CPython 3.11/Linux x86_64. La instalación en otras plataformas puede requerir ajustes. La reconstrucción completa desde un entorno limpio debe verificarse antes de considerar reproducida una ejecución.

### 4. Verificar las importaciones y versiones

```bash
python tests/verify_environment.py
```

El script muestra las versiones principales y, si las importaciones finalizan correctamente, imprime:

```text
Estado: ENTORNO OPERATIVO
```

Para comprobar además las bibliotecas utilizadas en el procesamiento de datos y las figuras:

```bash
python -c "import pandas, sklearn, joblib, matplotlib; print('Importaciones disponibles')"
```

Estas comprobaciones verifican la disponibilidad de las bibliotecas; las pruebas de cada componente validan su funcionamiento.

## Preparación de los datos

Para ejecutar los experimentos de detección, colocar los archivos originales en:

```text
data/raw/unsw_nb15/UNSW_NB15_training-set.csv
data/raw/unsw_nb15/UNSW_NB15_testing-set.csv
```

Después ejecutar:

```bash
python src/data/inspect_unsw.py
python src/data/preprocess_unsw.py
```

El preprocesamiento genera las particiones, el transformador y los metadatos en:

```text
data/processed/unsw_nb15/
```

Los datos originales y procesados no se descargan al clonar el repositorio. Los requisitos y las transformaciones aplicadas se describen en la documentación técnica del modelo de detección.

## Ejecución y validación

Los procedimientos disponibles se organizan por componente:

| Componente | Comando | Requisito previo |
|---|---|---|
| Entorno federado | `python tests/verify_environment.py` | Entorno activado y dependencias instaladas |
| Mecanismo federado base | `bash scripts/validate_re1_3.sh` | Entorno federado operativo |
| Modelo centralizado de detección | `bash scripts/validate_re2_1.sh` | Entorno operativo y datos preprocesados |

El script del mecanismo federado base ejecuta verificaciones de clientes, servidor, entrenamiento y localidad lógica de los datos.

El script del modelo centralizado ejecuta la selección de configuración, genera las figuras y realiza el entrenamiento y la evaluación final. Sus opciones y criterios de revisión se explican en la documentación correspondiente.

**Los procedimientos de entrenamiento pueden regenerar y sobrescribir resultados, modelos y logs.** Conservar las evidencias de una corrida antes de repetirla cuando sea necesario mantener su trazabilidad.

Las pruebas de componentes todavía en desarrollo pueden depender de módulos que estén siendo reorganizados. Para reproducir los componentes consolidados, utilizar los procedimientos descritos en sus documentos técnicos.

## Documentación técnica

| Documento | Contenido |
|---|---|
| [Prototipo del mecanismo federado](docs/re1/re1_3_prototipo_federado.md) | Implementación, ejecución y verificación del mecanismo base |
| [Modelo centralizado de detección](docs/re2/re2_1_modelo_centralizado.md) | Dataset, preprocesamiento, selección, entrenamiento, evaluación y reproducción |
| [Dependencias](requirements/) | Referencias del entorno de ejecución |
| [Resultados](results/) | Métricas, tablas, historiales y figuras almacenadas |
| [Logs](logs/) | Evidencias de ejecución |

La documentación de integración federada, comparación experimental y lineamientos técnicos se incorporará conforme se consoliden esas etapas.

Este README mantiene la visión general del proyecto. Las arquitecturas concretas de los modelos, hiperparámetros, métricas numéricas y verificaciones de cada resultado se desarrollan en `docs/`.

## Reproducibilidad y trazabilidad

Cada resultado experimental debe poder relacionarse con:

- La revisión del código utilizada.
- El entorno y las versiones de dependencias.
- Los datos y el preprocesamiento aplicado.
- La configuración y las semillas de ejecución.
- El procedimiento de entrenamiento y evaluación.
- Los modelos, predicciones, métricas y logs generados.

El repositorio conserva código, documentación y determinadas evidencias. Los datos y modelos excluidos mediante `.gitignore` deben recuperarse o regenerarse para reproducir completamente una ejecución.

Las semillas favorecen la reproducibilidad, pero no garantizan igualdad numérica exacta entre plataformas. Cualquier nueva ejecución debe evaluarse a partir de sus propios resultados.

## Alcance y limitaciones

El proyecto tiene carácter académico y experimental:

- Utiliza clientes simulados y un dataset de referencia.
- La evaluación actual se realiza en un entorno controlado.
- No representa todavía un sistema de detección desplegado en producción.
- Los resultados obtenidos sobre UNSW-NB15 no demuestran por sí solos desempeño equivalente sobre tráfico real de microempresas.
- Las condiciones de comunicación, seguridad y operación deben analizarse antes de trasladar el prototipo a una infraestructura distribuida real.

Las limitaciones específicas de cada experimento se registran en su documentación técnica.

## Próximos hitos

- Consolidar la integración entre el modelo de detección y el mecanismo federado.
- Documentar el entrenamiento y la evaluación del modelo global.
- Completar la comparación con la línea base centralizada.
- Elaborar los lineamientos técnicos de despliegue y gestión de accesos.
- Verificar la reproducción de los componentes desde un entorno limpio.

## Autor

**Jesús Alonso Ysla Quispe**  
Ingeniería Informática  
Pontificia Universidad Católica del Perú