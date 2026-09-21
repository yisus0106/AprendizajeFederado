import numpy as np

from src.federated.unsw_client_data import (
    EXPECTED_NUM_FEATURES,
    build_stratified_client_indices,
    create_unsw_client_partitions,
    load_common_test_set,
    load_unsw_split,
)


def verify_unsw_client_data():
    """
    Verifica la carga y distribución de UNSW-NB15
    entre los clientes federados.
    """

    print(
        "=== PRUEBA DE PARTICIONES UNSW-NB15 ==="
    )

    # ========================================================
    # Cargar TRAIN
    # ========================================================

    X_train, y_train = load_unsw_split(
        "train"
    )

    assert len(X_train) == len(y_train)

    assert (
        X_train.shape[1]
        == EXPECTED_NUM_FEATURES
    )

    assert set(
        np.unique(y_train).tolist()
    ).issubset(
        {0, 1}
    )

    print(
        f"TRAIN total: "
        f"{len(y_train)} ejemplos"
    )

    print(
        f"Características: "
        f"{X_train.shape[1]}"
    )

    # ========================================================
    # Verificar índices estratificados
    # ========================================================

    client_indices = (
        build_stratified_client_indices(
            labels=y_train,
            num_clients=2,
            seed=42,
        )
    )

    assert len(client_indices) == 2

    all_indices = np.concatenate(
        client_indices
    )

    # Todos los registros deben estar asignados.
    assert (
        len(all_indices)
        == len(y_train)
    )

    # No deben existir registros duplicados.
    assert (
        len(np.unique(all_indices))
        == len(y_train)
    )

    # Los índices deben cubrir exactamente TRAIN.
    assert np.array_equal(
        np.sort(all_indices),
        np.arange(len(y_train)),
    )

    print(
        "Cobertura completa de TRAIN: OK"
    )

    print(
        "Registros duplicados: 0"
    )

    # ========================================================
    # Verificar reproducibilidad
    # ========================================================

    repeated_indices = (
        build_stratified_client_indices(
            labels=y_train,
            num_clients=2,
            seed=42,
        )
    )

    for first, repeated in zip(
        client_indices,
        repeated_indices,
    ):
        assert np.array_equal(
            first,
            repeated,
        )

    print(
        "Particionado reproducible: OK"
    )

    # ========================================================
    # Crear las particiones locales
    # ========================================================

    partitions = (
        create_unsw_client_partitions(
            num_clients=2,
            seed=42,
        )
    )

    assert len(partitions) == 2

    total_client_examples = 0

    for client_id, partition in (
        partitions.items()
    ):

        total_client_examples += (
            partition.num_examples
        )

        assert (
            partition.features.ndim
            == 2
        )

        assert (
            partition.features.shape[1]
            == EXPECTED_NUM_FEATURES
        )

        assert (
            partition.labels.ndim
            == 2
        )

        assert (
            partition.labels.shape[1]
            == 1
        )

        assert (
            len(partition.features)
            == len(partition.labels)
        )

        labels, counts = np.unique(
            partition.labels,
            return_counts=True,
        )

        distribution = {
            int(label): int(count)
            for label, count
            in zip(labels, counts)
        }

        # Cada cliente debe recibir ambas clases.
        assert set(
            distribution.keys()
        ) == {
            0,
            1,
        }

        print(
            f"{client_id}: "
            f"{partition.num_examples} ejemplos "
            f"| clases={distribution}"
        )

    assert (
        total_client_examples
        == len(y_train)
    )

    print(
        "Suma de particiones igual a TRAIN: OK"
    )

    # ========================================================
    # Verificar TEST común
    # ========================================================

    X_test, y_test = (
        load_common_test_set()
    )

    assert len(X_test) == len(y_test)

    assert (
        X_test.shape[1]
        == EXPECTED_NUM_FEATURES
    )

    assert set(
        np.unique(y_test).tolist()
    ).issubset(
        {0, 1}
    )

    print(
        f"TEST común: "
        f"{len(y_test)} ejemplos"
    )

    print(
        "TEST reservado para evaluación: OK"
    )

    print(
        "\nEstado: PARTICIONES "
        "UNSW-NB15 OPERATIVAS"
    )


if __name__ == "__main__":
    verify_unsw_client_data()