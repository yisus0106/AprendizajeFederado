import numpy as np

from src.federated.client_data import (
    create_synthetic_client_partitions,
)
from src.federated.server import build_federated_process


def main():

    print("=== VERIFICACION DE LOCALIDAD LOGICA DE DATOS ===")

    # -------------------------------------------------
    # 1. Verificar existencia de clientes independientes
    # -------------------------------------------------

    partitions = create_synthetic_client_partitions(
        num_clients=2,
        samples_per_client=12,
        seed=42,
    )

    assert len(partitions) == 2, (
        "Se esperaban exactamente dos clientes."
    )

    print(f"Clientes verificados: {len(partitions)}")

    client_ids = list(partitions.keys())

    assert client_ids[0] != client_ids[1], (
        "Los clientes deben tener identificadores distintos."
    )

    # -------------------------------------------------
    # 2. Verificar estructura de las particiones locales
    # -------------------------------------------------

    total_examples = 0

    for client_id, client in partitions.items():

        assert client.num_examples == 12, (
            f"{client_id} no contiene los 12 ejemplos esperados."
        )

        assert client.features.shape == (12, 2), (
            f"Estructura inválida de features en {client_id}."
        )

        assert client.labels.shape == (12, 1), (
            f"Estructura inválida de labels en {client_id}."
        )

        assert len(client.features) == len(client.labels), (
            f"features y labels no coinciden en {client_id}."
        )

        total_examples += client.num_examples

        print(
            f"{client_id}: "
            f"{client.num_examples} ejemplos locales "
            f"| X={client.features.shape} "
            f"| y={client.labels.shape}"
        )

    assert total_examples == 24, (
        "La cantidad total de ejemplos no coincide."
    )

    # -------------------------------------------------
    # 3. Verificar que las particiones no se solapen
    # -------------------------------------------------

    client_1 = partitions[client_ids[0]]
    client_2 = partitions[client_ids[1]]

    assert not np.shares_memory(
        client_1.features,
        client_2.features,
    ), "Los clientes comparten memoria para features."

    assert not np.shares_memory(
        client_1.labels,
        client_2.labels,
    ), "Los clientes comparten memoria para labels."

    # Se representa cada ejemplo mediante:
    # (variable_1, variable_2, etiqueta)
    samples_client_1 = {
        tuple(row)
        for row in np.concatenate(
            [client_1.features, client_1.labels],
            axis=1,
        )
    }

    samples_client_2 = {
        tuple(row)
        for row in np.concatenate(
            [client_2.features, client_2.labels],
            axis=1,
        )
    }

    overlapping_samples = (
        samples_client_1.intersection(samples_client_2)
    )

    assert len(overlapping_samples) == 0, (
        "Se detectaron ejemplos compartidos entre clientes."
    )

    unique_samples = samples_client_1.union(
        samples_client_2
    )

    assert len(unique_samples) == 24, (
        "Las particiones no contienen 24 ejemplos distintos."
    )

    
    print("\nSeparacion de particiones:")
    print("Ejemplos compartidos entre clientes: 0 - OK")
    print("Ejemplos unicos totales: 24 - OK")
    print("Memoria compartida en features: NO - OK")
    print("Memoria compartida en labels: NO - OK")
    
    # -------------------------------------------------
    # 4. Verificar semántica federada de TFF
    # -------------------------------------------------

    process = build_federated_process()

    signature = str(process.next.type_signature)

    print("\nFirma federada del proceso:")
    print(signature)

    assert "@SERVER" in signature, (
        "No se encontro estado federado ubicado en SERVER."
    )

    assert "@CLIENTS" in signature, (
        "No se encontraron datos ubicados en CLIENTS."
    )

    assert "client_data=" in signature, (
        "La interfaz federada no declara datos de clientes."
    )

    print("\nSemantica federada:")
    print("Estado global ubicado en SERVER - OK")
    print("Datos de entrenamiento ubicados en CLIENTS - OK")

    # -------------------------------------------------
    # 5. Resultado de la verificación
    # -------------------------------------------------

    print("\nResumen:")
    print(f"Clientes independientes: {len(partitions)}")
    print(f"Ejemplos totales: {total_examples}")
    print(
        f"Ejemplos compartidos: "
        f"{len(overlapping_samples)}"
    )
    print("Estado global: SERVER")
    print("Datos de entrenamiento: CLIENTS")

    print(
        "\nEstado: LOCALIDAD LOGICA DE DATOS "
        "FEDERADOS VERIFICADA"
    )


if __name__ == "__main__":
    main()