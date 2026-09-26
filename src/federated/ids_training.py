from pathlib import Path
import csv
import random
import time

import numpy as np
import tensorflow as tf

from src.federated.ids_model import (
    create_ids_keras_model,
)

from src.federated.ids_server import (
    build_ids_federated_process,
)

from src.federated.unsw_client_data import (
    create_unsw_client_partitions,
)


BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "re2"
    / "federated"
)

MODEL_DIR = (
    BASE_DIR
    / "artifacts"
    / "models"
    / "federated"
)


def prepare_ids_clients(
    num_clients=2,
    batch_size=256,
    local_epochs=1,
    seed=42,
):
    """
    Prepara las particiones locales UNSW-NB15
    como datasets compatibles con TFF.
    """

    partitions = create_unsw_client_partitions(
        num_clients=num_clients,
        seed=seed,
    )

    federated_datasets = []

    for partition in partitions.values():

        dataset = partition.to_tf_dataset(
            batch_size=batch_size,
            local_epochs=local_epochs,
            shuffle=True,
            seed=seed,
        )

        federated_datasets.append(
            dataset
        )

    return partitions, federated_datasets


def execute_ids_federated_training(
    num_rounds=1,
    num_clients=2,
    batch_size=256,
    local_epochs=1,
    seed=42,
):
    """
    Ejecuta el entrenamiento federado del modelo IDS
    utilizando particiones locales de UNSW-NB15.
    """

    if num_rounds <= 0:
        raise ValueError(
            "num_rounds debe ser mayor que cero."
        )

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    process = build_ids_federated_process()

    state = process.initialize()

    partitions, federated_datasets = (
        prepare_ids_clients(
            num_clients=num_clients,
            batch_size=batch_size,
            local_epochs=local_epochs,
            seed=seed,
        )
    )

    history = []

    for round_number in range(
        1,
        num_rounds + 1,
    ):

        print(
            f"\nIniciando ronda "
            f"{round_number}/{num_rounds}..."
        )

        weights_before = (
            process.get_model_weights(
                state
            )
        )

        trainable_before = [
            np.asarray(weight).copy()
            for weight
            in weights_before.trainable
        ]

        start_time = time.perf_counter()

        output = process.next(
            state,
            federated_datasets,
        )

        round_time = (
            time.perf_counter()
            - start_time
        )

        state = output.state

        weights_after = (
            process.get_model_weights(
                state
            )
        )

        trainable_after = [
            np.asarray(weight)
            for weight
            in weights_after.trainable
        ]

        changed_parameters = sum(
            not np.allclose(
                before,
                after,
            )
            for before, after
            in zip(
                trainable_before,
                trainable_after,
            )
        )

        train_metrics = (
            output.metrics[
                "client_work"
            ][
                "train"
            ]
        )

        round_record = {
            "round": round_number,

            "loss": float(
                train_metrics["loss"]
            ),

            "accuracy": float(
                train_metrics["accuracy"]
            ),

            "num_examples": int(
                train_metrics[
                    "num_examples"
                ]
            ),

            "num_batches": int(
                train_metrics[
                    "num_batches"
                ]
            ),

            "changed_parameters":
                changed_parameters,

            "round_time_seconds":
                float(round_time),
        }

        history.append(
            round_record
        )

        print(
            f"Ronda {round_number}: "
            f"loss={round_record['loss']:.6f}, "
            f"accuracy="
            f"{round_record['accuracy']:.6f}, "
            f"ejemplos="
            f"{round_record['num_examples']}, "
            f"tiempo="
            f"{round_record['round_time_seconds']:.2f}s"
        )

    final_weights = (
        process.get_model_weights(
            state
        )
    )

    # --------------------------------------------------------
    # Convertir el modelo global TFF a un modelo Keras
    # --------------------------------------------------------

    global_model = (
        create_ids_keras_model()
    )

    keras_weights = [
        np.asarray(weight)
        for weight
        in final_weights.trainable
    ]

    keras_weights += [
        np.asarray(weight)
        for weight
        in final_weights.non_trainable
    ]

    global_model.set_weights(
        keras_weights
    )

    # --------------------------------------------------------
    # Guardar resultados nuevos de RE2.2
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        RESULTS_DIR
        / "training_history.csv"
    )

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)

    model_path = (
        MODEL_DIR
        / "federated_ids.keras"
    )

    global_model.save(
        model_path
    )

    return {
        "process": process,
        "partitions": partitions,
        "final_state": state,
        "final_weights": final_weights,
        "global_model": global_model,
        "history": history,
        "history_path": history_path,
        "model_path": model_path,
    }