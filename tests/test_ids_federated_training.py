import numpy as np

from src.federated.ids_training import (
    execute_ids_federated_training,
)


NUM_ROUNDS = 5
NUM_CLIENTS = 2


def verify_ids_federated_training():

    print(
        "=== ENTRENAMIENTO FEDERADO IDS ==="
    )

    result = (
        execute_ids_federated_training(
            num_rounds=NUM_ROUNDS,
            num_clients=NUM_CLIENTS,
            batch_size=256,
            local_epochs=1,
            seed=42,
        )
    )

    partitions = result["partitions"]
    history = result["history"]

    assert len(partitions) == NUM_CLIENTS
    assert len(history) == NUM_ROUNDS

    total_examples = sum(
        partition.num_examples
        for partition
        in partitions.values()
    )

    print(
        f"\nClientes participantes: "
        f"{len(partitions)}"
    )

    for client_id, partition in (
        partitions.items()
    ):
        print(
            f"{client_id}: "
            f"{partition.num_examples} ejemplos"
        )

    for record in history:

        assert np.isfinite(
            record["loss"]
        )

        assert np.isfinite(
            record["accuracy"]
        )

        assert (
            record["num_examples"]
            == total_examples
        )

        assert (
            record["changed_parameters"]
            > 0
        )

    assert result[
        "history_path"
    ].exists()

    assert result[
        "model_path"
    ].exists()

    print(
        f"\nRondas completadas: "
        f"{len(history)}"
    )

    print(
        f"Modelo global: "
        f"{result['model_path']}"
    )

    print(
        "Estado: ENTRENAMIENTO IDS OPERATIVO"
    )


if __name__ == "__main__":
    verify_ids_federated_training()