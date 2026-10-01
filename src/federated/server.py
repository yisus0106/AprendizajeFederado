import math

import tensorflow_federated as tff

from src.federated.model import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_METADATA_PATH,
    load_model_config,
    make_model_fn,
)


def build_client_optimizer(config):
    """Construye el optimizador local indicado por el JSON."""
    learning_rate = config["learning_rate"]
    optimizer = config["optimizer"]

    if optimizer == "adam":
        return tff.learning.optimizers.build_adam(
            learning_rate=learning_rate,
        )

    if optimizer == "sgd":
        return tff.learning.optimizers.build_sgdm(
            learning_rate=learning_rate,
            momentum=0.0,
        )

    raise ValueError(f"Optimizador no admitido: {optimizer}")


def build_federated_process(
    config_path=DEFAULT_CONFIG_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
    server_learning_rate=1.0,
):
    """
    Construye FedAvg y devuelve su configuración efectiva.

    No carga clientes, no inicializa el estado global
    y no ejecuta rondas de entrenamiento.
    """
    if (
        isinstance(server_learning_rate, bool)
        or not isinstance(server_learning_rate, (int, float))
        or not math.isfinite(server_learning_rate)
        or server_learning_rate <= 0
    ):
        raise ValueError(
            "server_learning_rate debe ser finito y positivo."
        )

    config = load_model_config(
        config_path=config_path,
        metadata_path=metadata_path,
    )

    client_optimizer = build_client_optimizer(config)

    server_optimizer = tff.learning.optimizers.build_sgdm(
        learning_rate=float(server_learning_rate),
        momentum=0.0,
    )

    process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=make_model_fn(config),
        client_optimizer_fn=client_optimizer,
        server_optimizer_fn=server_optimizer,
        client_weighting=tff.learning.ClientWeighting.NUM_EXAMPLES,
    )

    effective_config = {
        "model": config,
        "federated": {
            "aggregation": "FedAvg",
            "client_weighting": "NUM_EXAMPLES",
            "client_optimizer": config["optimizer"],
            "client_learning_rate": config["learning_rate"],
            "server_optimizer": "sgd",
            "server_learning_rate": float(server_learning_rate),
            "server_momentum": 0.0,
        },
    }

    return process, effective_config