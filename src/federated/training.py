import numpy as np

from src.federated.client_data import create_synthetic_client_partitions
from src.federated.server import build_federated_process


def prepare_federated_clients(
    num_clients=2,
    samples_per_client=12,
    batch_size=4,
    local_epochs=1,
    seed=42,
):
    """
    Crea las particiones locales y genera los datasets
    utilizados por los clientes federados simulados.
    """

    # Se crean las particiones de datos sintéticos para cada cliente,
    # asegurando que cada uno tenga un conjunto de datos local independiente.
    partitions = create_synthetic_client_partitions(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        seed=seed,
    )

    federated_datasets = []

    for client in partitions.values():
        # Se construye el tf.data.Dataset para cada cliente, que será utilizado
        # exclusivamente para su entrenamiento local.
        dataset = client.to_tf_dataset(
            batch_size=batch_size,
            local_epochs=local_epochs,
            shuffle=True,
            seed=seed,
        )

        federated_datasets.append(dataset)

    return partitions, federated_datasets


def execute_single_round():
    """
    Ejecuta una ronda completa del mecanismo federado
    utilizando dos clientes simulados.
    """

    # Construcción del proceso federado que implementa FedAvg.
    process = build_federated_process()

    # Estado global antes del entrenamiento.
    state = process.initialize()

    # Obtención de los pesos iniciales del modelo antes de la ronda federada.
    initial_weights = process.get_model_weights(state)

    initial_trainable = [
        np.array(weight)
        for weight in initial_weights.trainable
    ]

    # Preparación independiente de los clientes.
    partitions, federated_datasets = prepare_federated_clients()

    # Una ronda federada real.
    output = process.next(
        state,
        federated_datasets,
    )

    new_state = output.state
    metrics = output.metrics

    updated_weights = process.get_model_weights(new_state)

    updated_trainable = [
        np.array(weight)
        for weight in updated_weights.trainable
    ]

    return {
        "process": process,
        "partitions": partitions,
        "initial_state": state,
        "new_state": new_state,
        "initial_weights": initial_trainable,
        "updated_weights": updated_trainable,
        "metrics": metrics,
    }
    
def execute_federated_training(
    num_rounds=5,
    num_clients=2,
    samples_per_client=12,
    batch_size=4,
    local_epochs=1,
    seed=42,
):
    """
    Ejecuta múltiples rondas consecutivas de entrenamiento federado.

    En cada ronda:
    1. Se utiliza el estado global vigente.
    2. Los clientes entrenan únicamente con sus datasets locales.
    3. FedAvg agrega las actualizaciones.
    4. Se obtiene un nuevo estado global.
    """

    if num_rounds <= 0:
        raise ValueError(
            "El número de rondas debe ser mayor que cero."
        )

    process = build_federated_process()

    state = process.initialize()

    partitions, federated_datasets = prepare_federated_clients(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        batch_size=batch_size,
        local_epochs=local_epochs,
        seed=seed,
    )

    history = []

    for round_number in range(1, num_rounds + 1):

        weights_before = process.get_model_weights(state)

        trainable_before = [
            np.array(weight)
            for weight in weights_before.trainable
        ]

        output = process.next(
            state,
            federated_datasets,
        )

        state = output.state

        weights_after = process.get_model_weights(state)

        trainable_after = [
            np.array(weight)
            for weight in weights_after.trainable
        ]

        changed_parameters = sum(
            not np.allclose(before, after)
            for before, after in zip(
                trainable_before,
                trainable_after,
            )
        )

        train_metrics = (
            output.metrics["client_work"]["train"]
        )

        history.append(
            {
                "round": round_number,
                "loss": float(train_metrics["loss"]),
                "accuracy": float(
                    train_metrics["accuracy"]
                ),
                "num_examples": int(
                    train_metrics["num_examples"]
                ),
                "num_batches": int(
                    train_metrics["num_batches"]
                ),
                "changed_parameters": changed_parameters,
            }
        )

    final_weights = process.get_model_weights(state)

    return {
        "process": process,
        "partitions": partitions,
        "final_state": state,
        "final_weights": final_weights,
        "history": history,
    }