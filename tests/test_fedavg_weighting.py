import numpy as np

from src.federated.client_data import (
    ClientPartition,
    create_synthetic_client_partitions,
)

from src.federated.server import (
    build_federated_process,
)

from src.federated.training import (
    execute_federated_training,
)


# =========================================================
# CONFIGURACION DE LA PRUEBA
# =========================================================

# Cantidad de ejemplos que tendra cada cliente.
#
# Ejemplos:
# [80, 20]
# [50, 50]
# [60, 30, 10]
# [100, 50, 25, 25]
CLIENT_SIZES = [250, 300, 150, 80]

NUM_ROUNDS = 500

BATCH_SIZE = 4

LOCAL_EPOCHS = 1

SEED = 42


def get_bias(process, state):
    """
    Obtiene el bias de la capa Dense del modelo global.
    """

    weights = process.get_model_weights(state)

    return float(
        np.asarray(
            weights.trainable[-1]
        ).reshape(-1)[0]
    )


def verify_client_distribution():
    """
    Verifica que cada cliente reciba exactamente
    la cantidad de ejemplos indicada en CLIENT_SIZES.
    """

    print()
    print("=== VERIFICACION DE PARTICIONES ===")

    num_clients = len(CLIENT_SIZES)

    partitions = create_synthetic_client_partitions(
        num_clients=num_clients,
        samples_per_client=CLIENT_SIZES,
        seed=SEED,
    )

    assert len(partitions) == num_clients

    total_examples = sum(CLIENT_SIZES)

    print(
        f"Total de ejemplos: {total_examples}"
    )

    print(
        f"Clientes participantes: {num_clients}"
    )

    print()

    for index, (
        client_id,
        client,
    ) in enumerate(partitions.items()):

        expected_examples = CLIENT_SIZES[index]

        assert (
            client.num_examples
            == expected_examples
        )

        weight = (
            client.num_examples
            / total_examples
        )

        print(
            f"{client_id}: "
            f"{client.num_examples} ejemplos "
            f"-> peso FedAvg esperado = "
            f"{weight:.4f} "
            f"({weight * 100:.2f} %)"
        )

    print()

    print(
        "Distribucion de clientes "
        "verificada correctamente."
    )

    return partitions


def verify_fedavg_weighting():
    """
    Comprueba experimentalmente que FedAvg
    pondera las actualizaciones de los clientes
    segun su cantidad de ejemplos.

    La prueba funciona con cualquier cantidad
    de clientes definida en CLIENT_SIZES.
    """

    print()
    print("=== VERIFICACION DE PONDERACION FEDAVG ===")

    num_clients = len(CLIENT_SIZES)

    if num_clients < 2:
        raise ValueError(
            "La prueba requiere al menos "
            "dos clientes."
        )

    total_examples = sum(CLIENT_SIZES)

    # -----------------------------------------------------
    # Crear clientes controlados
    # -----------------------------------------------------
    #
    # Se utilizan features [0, 0] para aislar el efecto
    # sobre el bias.
    #
    # Los clientes alternan etiquetas:
    #
    # cliente 1 -> etiqueta 1
    # cliente 2 -> etiqueta 0
    # cliente 3 -> etiqueta 1
    # cliente 4 -> etiqueta 0
    # ...
    #
    # Así las actualizaciones locales no son iguales.
    # -----------------------------------------------------

    controlled_clients = []

    for index, client_size in enumerate(
        CLIENT_SIZES
    ):

        label_value = (
            1.0
            if index % 2 == 0
            else 0.0
        )

        client = ClientPartition(
            client_id=f"controlled_client_{index + 1:02d}",
            features=np.zeros(
                (client_size, 2),
                dtype=np.float32,
            ),
            labels=np.full(
                (client_size, 1),
                label_value,
                dtype=np.float32,
            ),
        )

        controlled_clients.append(
            client
        )

    # -----------------------------------------------------
    # Crear datasets.
    #
    # Cada cliente utiliza un solo batch para que
    # la comparación matemática sea directa.
    # -----------------------------------------------------

    controlled_datasets = []

    for client in controlled_clients:

        dataset = client.to_tf_dataset(
            batch_size=client.num_examples,
            local_epochs=1,
            shuffle=False,
        )

        controlled_datasets.append(
            dataset
        )

    # -----------------------------------------------------
    # Construir FedAvg
    # -----------------------------------------------------

    process = build_federated_process(
        client_learning_rate=0.1,
        server_learning_rate=1.0,
    )

    initial_state = process.initialize()

    initial_bias = get_bias(
        process,
        initial_state,
    )

    print(
        f"Bias global inicial: "
        f"{initial_bias:.8f}"
    )

    print()

    # -----------------------------------------------------
    # Obtener la actualizacion individual
    # de cada cliente.
    #
    # Todos parten exactamente del mismo
    # estado global inicial.
    # -----------------------------------------------------

    individual_deltas = []

    for index, dataset in enumerate(
        controlled_datasets
    ):

        output = process.next(
            initial_state,
            [dataset],
        )

        client_bias = get_bias(
            process,
            output.state,
        )

        delta = (
            client_bias
            - initial_bias
        )

        individual_deltas.append(
            delta
        )

        client_weight = (
            CLIENT_SIZES[index]
            / total_examples
        )

        print(
            f"Cliente {index + 1:02d}:"
        )

        print(
            f"  ejemplos = "
            f"{CLIENT_SIZES[index]}"
        )

        print(
            f"  peso FedAvg = "
            f"{client_weight:.6f}"
        )

        print(
            f"  actualizacion local = "
            f"{delta:.8f}"
        )

    # -----------------------------------------------------
    # Calcular manualmente lo que FedAvg
    # deberia producir.
    # -----------------------------------------------------

    expected_delta = 0.0

    print()
    print(
        "Calculo manual de FedAvg:"
    )

    for index, delta in enumerate(
        individual_deltas
    ):

        weight = (
            CLIENT_SIZES[index]
            / total_examples
        )

        contribution = (
            weight * delta
        )

        expected_delta += contribution

        print(
            f"Cliente {index + 1:02d}: "
            f"{weight:.6f} "
            f"x {delta:.8f} "
            f"= {contribution:.8f}"
        )

    # -----------------------------------------------------
    # Ejecutar todos los clientes juntos.
    # -----------------------------------------------------

    mixed_output = process.next(
        initial_state,
        controlled_datasets,
    )

    mixed_bias = get_bias(
        process,
        mixed_output.state,
    )

    obtained_delta = (
        mixed_bias
        - initial_bias
    )

    print()
    print(
        f"Actualizacion esperada: "
        f"{expected_delta:.8f}"
    )

    print(
        f"Actualizacion obtenida: "
        f"{obtained_delta:.8f}"
    )

    difference = abs(
        obtained_delta
        - expected_delta
    )

    print(
        f"Diferencia: "
        f"{difference:.10f}"
    )

    assert np.isclose(
        obtained_delta,
        expected_delta,
        atol=1e-5,
    ), (
        "FedAvg no pondero correctamente "
        "las actualizaciones segun "
        "la cantidad de ejemplos."
    )

    print()

    print(
        "Ponderacion FedAvg "
        "verificada correctamente."
    )


def verify_federated_rounds():
    """
    Ejecuta varias rondas completas de entrenamiento
    federado utilizando la distribucion configurada
    en CLIENT_SIZES.
    """

    print()
    print("=== EJECUCION DE RONDAS FEDERADAS ===")

    num_clients = len(CLIENT_SIZES)

    total_local_examples = sum(
        CLIENT_SIZES
    )

    # Como el dataset se repite LOCAL_EPOCHS veces,
    # TFF procesa esta cantidad de ejemplos por ronda.
    expected_processed_examples = (
        total_local_examples
        * LOCAL_EPOCHS
    )

    result = execute_federated_training(
        num_rounds=NUM_ROUNDS,
        num_clients=num_clients,
        samples_per_client=CLIENT_SIZES,
        batch_size=BATCH_SIZE,
        local_epochs=LOCAL_EPOCHS,
        seed=SEED,
    )

    partitions = result[
        "partitions"
    ]

    history = result[
        "history"
    ]

    # -----------------------------------------------------
    # Verificar clientes utilizados
    # -----------------------------------------------------

    assert (
        len(partitions)
        == num_clients
    )

    print()

    print(
        f"Clientes participantes: "
        f"{len(partitions)}"
    )

    for client_id, client in (
        partitions.items()
    ):

        weight = (
            client.num_examples
            / total_local_examples
        )

        print(
            f"{client_id}: "
            f"{client.num_examples} ejemplos "
            f"-> peso = "
            f"{weight:.4f} "
            f"({weight * 100:.2f} %)"
        )

    # -----------------------------------------------------
    # Verificar rondas
    # -----------------------------------------------------

    assert (
        len(history)
        == NUM_ROUNDS
    )

    print()

    print(
        f"Rondas requeridas: "
        f"{NUM_ROUNDS}"
    )

    print(
        f"Rondas completadas: "
        f"{len(history)}"
    )

    print()

    # -----------------------------------------------------
    # Mostrar y verificar cada ronda
    # -----------------------------------------------------

    for record in history:

        assert (
            record["num_examples"]
            == expected_processed_examples
        )

        assert (
            record["changed_parameters"]
            > 0
        )

        print(
            f"Ronda {record['round']:03d}: "
            f"loss={record['loss']:.6f}, "
            f"accuracy={record['accuracy']:.6f}, "
            f"num_examples="
            f"{record['num_examples']}, "
            f"parametros_actualizados="
            f"{record['changed_parameters']}"
        )

    print()

    print(
        f"Las {NUM_ROUNDS} rondas "
        f"produjeron actualizaciones "
        f"del modelo global."
    )

    print()

    print(
        "Estado: ENTRENAMIENTO FEDERADO "
        "CON PONDERACION OPERATIVO"
    )


def main():

    print(
        "========================================"
    )

    print(
        " PRUEBA DE FEDAVG CON CLIENTES "
        "DESBALANCEADOS"
    )

    print(
        "========================================"
    )

    print()

    print(
        f"Configuracion de clientes: "
        f"{CLIENT_SIZES}"
    )

    print(
        f"Numero de rondas: "
        f"{NUM_ROUNDS}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epocas locales: "
        f"{LOCAL_EPOCHS}"
    )

    # 1. Verificar cantidades de datos
    verify_client_distribution()

    # 2. Verificar matematicamente FedAvg
    verify_fedavg_weighting()

    # 3. Ejecutar entrenamiento federado real
    verify_federated_rounds()

    print()
    print(
        "========================================"
    )

    print(
        " TODAS LAS PRUEBAS FUERON SUPERADAS"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()