# Arquitectura de Aprendizaje Federado para Detección Colaborativa de Intrusiones

Repositorio de desarrollo de la tesis:

**“Diseño y prototipo de una arquitectura de aprendizaje federado para la detección colaborativa de intrusiones mediante el análisis de tráfico de red en microempresas”.**

Proyecto desarrollado en la especialidad de **Ingeniería Informática de la Pontificia Universidad Católica del Perú (PUCP)**.

---

## Descripción

El proyecto investiga el uso de aprendizaje federado como mecanismo de colaboración entre múltiples participantes que requieren construir conocimiento compartido sin centralizar directamente sus datos locales de entrenamiento.

La propuesta utiliza una arquitectura cliente-servidor en la que los clientes realizan operaciones de entrenamiento sobre sus propios datos, mientras que un servidor agregador coordina el proceso y mantiene el estado del modelo global.

El mecanismo de agregación base utilizado es **Federated Averaging (FedAvg)**.

---

## Objetivo general

Diseñar e implementar un prototipo basado en una arquitectura de aprendizaje federado para la detección colaborativa de intrusiones mediante el análisis de tráfico de red en microempresas, permitiendo aprovechar el aprendizaje distribuido sin centralizar los datos locales.

---

## Arquitectura general

La arquitectura propuesta sigue un esquema cliente-servidor.

```text
                    ┌─────────────────────┐
                    │  Servidor agregador │
                    │                     │
                    │    Modelo global    │
                    │         +           │
                    │       FedAvg        │
                    └──────────┬──────────┘
                               │
                     estado del modelo
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
        ┌──────────────┐              ┌──────────────┐
        │   Cliente A  │              │   Cliente B  │
        │              │              │              │
        │ Datos locales│              │ Datos locales│
        │ Entrenamiento│              │ Entrenamiento│
        │    local     │              │    local     │
        └──────┬───────┘              └──────┬───────┘
               │                             │
               └───── actualizaciones ───────┘
                               │
                               ▼
                             FedAvg
                               │
                               ▼
                      Nuevo modelo global
```

Los datos utilizados durante el entrenamiento permanecen asociados a cada participante, mientras el servidor mantiene y actualiza el estado global del modelo.

---

## Estado del proyecto

| Etapa | Estado |
|---|---|
| Especificación de requisitos del entorno federado | Completada |
| Diseño de la arquitectura federada | Completada |
| Prototipo base del mecanismo federado | Completado |
| Integración con componentes posteriores del proyecto | Pendiente |
| Evaluación experimental final | Pendiente |
| Lineamientos técnicos de despliegue | Pendiente |

El estado de esta tabla se actualizará conforme avance el desarrollo de la tesis.

---

## Implementación disponible

Actualmente el repositorio contiene una implementación funcional del mecanismo base de aprendizaje federado.

El prototipo permite:

- simular clientes federados;
- mantener particiones locales independientes;
- ejecutar entrenamiento local;
- inicializar y mantener un modelo global;
- coordinar rondas mediante TensorFlow Federated;
- agregar actualizaciones mediante FedAvg;
- ejecutar rondas federadas consecutivas;
- registrar métricas y logs;
- verificar la localidad lógica de los datos mediante las ubicaciones `CLIENTS` y `SERVER`.

La documentación técnica completa de este resultado se encuentra en:

```text
docs/re1_3_prototipo_federado.md
```

---

## Tecnologías principales

El proyecto utiliza principalmente:

| Tecnología | Uso |
|---|---|
| Python | Lenguaje principal |
| TensorFlow | Construcción y entrenamiento de modelos |
| TensorFlow Federated | Coordinación del aprendizaje federado |
| NumPy | Operaciones numéricas |
| WSL2 | Entorno Linux sobre Windows |
| Ubuntu | Entorno de ejecución |
| Visual Studio Code | Desarrollo y depuración |
| Git | Control de versiones |

Otras herramientas serán incorporadas y documentadas conforme avance la implementación de los distintos resultados del proyecto.

---

## Entorno actualmente validado

| Componente | Versión |
|---|---|
| Ubuntu | 22.04.3 LTS |
| WSL | WSL2 |
| Python | 3.11.15 |
| TensorFlow | 2.14.1 |
| TensorFlow Federated | 0.87.0 |
| NumPy | 1.25.2 |
| SciPy | 1.9.3 |

Las dependencias completas se encuentran en:

```text
requirements/
```

---

## Estructura principal

La estructura general del repositorio es:

```text
AprendizajeFederado/
│
├── README.md
│
├── docs/
│   └── re1_3_prototipo_federado.md
│
├── data/
│
├── logs/
│
├── notebooks/
│
├── practica/
│
├── requirements/
│   ├── requirements-lock.txt
│   └── requirements_tff_cpu.txt
│
├── scripts/
│   └── validate_re1_3.sh
│
├── src/
│   ├── data/
│   ├── federated/
│   │   ├── client_data.py
│   │   ├── model.py
│   │   ├── server.py
│   │   └── training.py
│   │
│   ├── models/
│   ├── privacy/
│   ├── results/
│   └── utils/
│
└── tests/
    ├── verify_environment.py
    ├── test_client_data.py
    ├── test_server.py
    ├── test_federated_round.py
    ├── test_federated_training.py
    └── test_data_locality.py
```

Algunos directorios se encuentran reservados para componentes correspondientes a etapas posteriores del proyecto y serán documentados cuando formen parte de resultados formalmente implementados.

---

## Organización del código

### `src/federated/`

Contiene el mecanismo base de aprendizaje federado.

```text
client_data.py
```

Gestiona las particiones locales de los clientes y su conversión a datasets de TensorFlow.

```text
model.py
```

Define el modelo utilizado por el proceso federado.

```text
server.py
```

Construye el proceso de agregación y mantiene la configuración de FedAvg.

```text
training.py
```

Integra clientes, servidor y rondas de entrenamiento.

### `tests/`

Contiene las pruebas funcionales utilizadas para comprobar progresivamente los componentes.

### `logs/`

Contiene las evidencias generadas durante las ejecuciones.

### `scripts/`

Contiene scripts de validación y automatización del proyecto.

### `requirements/`

Contiene las dependencias requeridas para reproducir el entorno.

### `docs/`

Contiene documentación técnica específica de los resultados implementados.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/yisus0106/AprendizajeFederado.git
cd AprendizajeFederado
```

### 2. Crear un entorno virtual

```bash
python3 -m venv env_federado_tff
```

### 3. Activar el entorno

Linux / WSL2:

```bash
source env_federado_tff/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements/requirements-lock.txt
```

---

## Verificación del entorno

Ejecutar:

```bash
python tests/verify_environment.py
```

Una ejecución correcta debe finalizar con:

```text
Estado: ENTORNO OPERATIVO
```

---

## Ejecución del mecanismo federado

Para ejecutar la prueba principal de entrenamiento federado:

```bash
python -m tests.test_federated_training
```

La configuración actualmente utilizada para la validación funcional es:

```text
Clientes:            2
Ejemplos por cliente: 12
Rondas federadas:     5
```

La finalidad de esta prueba es comprobar el funcionamiento del mecanismo federado y no evaluar todavía el desempeño final de la aplicación de ciberseguridad.

---

## Validación completa del prototipo base

Para ejecutar secuencialmente las principales verificaciones:

```bash
./scripts/validate_re1_3.sh
```

La evidencia generada se encuentra en:

```text
logs/
```

La documentación detallada de las pruebas está disponible en:

```text
docs/re1_3_prototipo_federado.md
```

---

## Documentación

| Documento | Descripción |
|---|---|
| `README.md` | Visión general y estado del proyecto |
| `docs/re1_3_prototipo_federado.md` | Implementación, reproducción y validación técnica de RE1.3 |
| `requirements/` | Dependencias del entorno |
| `logs/` | Evidencias de ejecución |

Nuevos documentos técnicos serán añadidos a `docs/` conforme se implementen los siguientes resultados de la tesis.

---

## Reproducibilidad

El proyecto busca mantener trazabilidad entre:

```text
Código fuente
     +
Dependencias
     +
Pruebas
     +
Logs
     +
Documentación
```

Las versiones utilizadas se registran en `requirements/`, mientras que las principales pruebas y resultados de ejecución se conservan en `tests/` y `logs/`.

La instalación completa desde un entorno limpio deberá ser verificada nuevamente antes de publicar la versión final del proyecto.

---

## Alcance actual

El repositorio corresponde a un proyecto académico y experimental.

El prototipo actual:

- utiliza clientes simulados;
- se ejecuta dentro de un entorno controlado;
- no representa todavía un despliegue productivo;
- no supone infraestructura instalada en microempresas reales;
- utiliza el aprendizaje federado como base para etapas posteriores del proyecto.

Las capacidades adicionales se incorporarán progresivamente conforme se desarrollen los siguientes resultados definidos en la tesis.

---

## Autor

**Jesús Alonso Ysla Quispe**  
Ingeniería Informática  
Pontificia Universidad Católica del Perú
