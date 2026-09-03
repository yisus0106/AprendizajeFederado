import numpy as np

from src.federated.client_data import (
    ClientPartition,
    create_synthetic_client_partitions,
)


def main():
    print("=== PRUEBA DE CLIENTES FEDERADOS ===")

    # -----------------------------------------------------
    # Prueba 1: crear dos clientes federados
    # -----------------------------------------------------

    clients = create_synthetic_client_partitions(
        num_clients=2,
        samples_per_client=12,
        seed=42,
    )

    assert len(clients) == 2

    print(f"Clientes creados: {len(clients)}")

    # -----------------------------------------------------
    # Prueba 2: comprobar las particiones locales
    # -----------------------------------------------------

    for client_id, client in clients.items():
        dataset = client.to_tf_dataset(
            batch_size=4,
            local_epochs=1,
            shuffle=False,
        )

        examples_read = 0

        for batch_x, batch_y in dataset:
            examples_read += int(batch_x.shape[0])

            assert batch_x.shape[1] == 2
            assert batch_y.shape[1] == 1

        assert examples_read == client.num_examples

        print(
            f"{client_id}: "
            f"{client.num_examples} ejemplos locales - OK"
        )

    # -----------------------------------------------------
    # Prueba 3: verificar rechazo de entrada inválida
    # -----------------------------------------------------

    invalid_features = np.zeros((5, 2), dtype=np.float32)
    invalid_labels = np.zeros((4, 1), dtype=np.float32)

    try:
        ClientPartition(
            client_id="invalid_client",
            features=invalid_features,
            labels=invalid_labels,
        )

        raise AssertionError(
            "Una partición inválida fue aceptada."
        )

    except ValueError as error:
        print("Entrada inválida rechazada correctamente.")
        print(f"Error controlado: {error}")

    print("Estado: CLIENTES LOCALES OPERATIVOS")


if __name__ == "__main__":
    main()