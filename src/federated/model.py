from copy import deepcopy
from pathlib import Path
import json
import math

import tensorflow as tf
import tensorflow_federated as tff


BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    BASE_DIR
    / "results"
    / "re2"
    / "centralized"
    / "tuning"
    / "best_config.json"
)

DEFAULT_METADATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "unsw_nb15"
    / "metadata.json"
)


def resolve_path(path):
    """Interpreta las rutas relativas desde la raíz del proyecto."""
    path = Path(path)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path


def read_json(path):
    path = resolve_path(path)

    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"{path}: se esperaba un objeto JSON.")

    return content


def validate_positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} debe ser un entero positivo.")

    return value


def validate_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} debe ser un número finito.")

    return float(value)


def load_model_config(
    config_path=DEFAULT_CONFIG_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
):
    """
    Lee una configuración seleccionada y valida sus campos.

    No carga datos de entrenamiento ni el conjunto TEST.
    """
    raw = read_json(config_path)

    required = (
        "hidden_layers",
        "dropout",
        "learning_rate",
        "classification_threshold",
    )

    missing = [name for name in required if name not in raw]

    if missing:
        raise ValueError(
            f"Faltan campos en best_config.json: {missing}"
        )

    # Dimensión de entrada.
    input_features = raw.get("input_features")
    metadata_features = None

    if metadata_path is not None:
        metadata_path = resolve_path(metadata_path)

        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            metadata_features = validate_positive_integer(
                metadata["processed_features"],
                "metadata.processed_features",
            )

    if input_features is None:
        input_features = metadata_features

    input_features = validate_positive_integer(
        input_features, "input_features"
    )

    if (
        metadata_features is not None
        and input_features != metadata_features
    ):
        raise ValueError(
            "La dimensión de best_config.json no coincide "
            "con metadata.json."
        )

    # Arquitectura.
    hidden_layers = raw["hidden_layers"]

    if not isinstance(hidden_layers, list) or not hidden_layers:
        raise ValueError(
            "hidden_layers debe ser una lista no vacía."
        )

    hidden_layers = [
        validate_positive_integer(units, f"hidden_layers[{i}]")
        for i, units in enumerate(hidden_layers)
    ]

    dropout = validate_number(raw["dropout"], "dropout")
    learning_rate = validate_number(
        raw["learning_rate"], "learning_rate"
    )
    threshold = validate_number(
        raw["classification_threshold"],
        "classification_threshold",
    )

    if not 0 <= dropout < 1:
        raise ValueError("dropout debe estar entre 0 y 1, excluyendo 1.")

    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser mayor que cero.")

    if not 0 < threshold < 1:
        raise ValueError(
            "classification_threshold debe estar entre 0 y 1."
        )

    # Los JSON anteriores no incluían optimizer.
    # Adam es el optimizador del modelo centralizado existente.
    optimizer = raw.get("optimizer", "Adam")

    if not isinstance(optimizer, str):
        raise ValueError("optimizer debe ser un texto.")

    optimizer = optimizer.lower()

    if optimizer not in ("adam", "sgd"):
        raise ValueError("optimizer debe ser Adam o SGD.")

    # Conserva los campos adicionales para trazabilidad.
    config = deepcopy(raw)
    config.update({
        "input_features": input_features,
        "hidden_layers": hidden_layers,
        "dropout": dropout,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "classification_threshold": threshold,
    })

    return config


def create_input_spec(config):
    """
    Compatible con ClientPartition.to_tf_dataset():
    X: float32, forma (batch, input_features)
    y: float32, forma (batch, 1)
    """
    return (
        tf.TensorSpec(
            shape=[None, config["input_features"]],
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=[None, 1],
            dtype=tf.float32,
        ),
    )


def create_keras_model(config):
    """
    Construye un modelo nuevo, sin compilar ni cargar pesos.

    Mantiene la arquitectura del centralizado:
    ReLU en capas ocultas, Dropout después de la primera
    y una salida sigmoide.
    """
    layers = [
        tf.keras.layers.Input(
            shape=(config["input_features"],),
            dtype=tf.float32,
        )
    ]

    for index, units in enumerate(config["hidden_layers"]):
        layers.append(
            tf.keras.layers.Dense(
                units,
                activation="relu",
                name=f"hidden_{index + 1}",
            )
        )

        if index == 0 and config["dropout"] > 0:
            layers.append(
                tf.keras.layers.Dropout(
                    config["dropout"],
                    name="dropout",
                )
            )

    layers.append(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="malicious_probability",
        )
    )

    return tf.keras.Sequential(layers, name="federated_ids")


def model_fn(config):
    """Construye una instancia nueva del modelo adaptado a TFF."""
    threshold = config["classification_threshold"]

    return tff.learning.models.from_keras_model(
        keras_model=create_keras_model(config),
        input_spec=create_input_spec(config),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
                threshold=threshold,
            ),
            tf.keras.metrics.Precision(
                name="precision",
                thresholds=threshold,
            ),
            tf.keras.metrics.Recall(
                name="recall",
                thresholds=threshold,
            ),
        ],
    )


def make_model_fn(config):
    """
    Devuelve la función sin argumentos que necesita TFF.

    Captura únicamente una copia de la configuración,
    no un modelo ya construido ni variables TensorFlow.
    """
    config_snapshot = deepcopy(config)

    def factory():
        return model_fn(config_snapshot)

    return factory