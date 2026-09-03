from src.federated.training import execute_federated_training


def main():

    print("=== PRUEBA DE ENTRENAMIENTO FEDERADO ===")

    NUM_ROUNDS = 5

    result = execute_federated_training(
        num_rounds=NUM_ROUNDS,
        num_clients=2,
        samples_per_client=12,
        batch_size=4,
        local_epochs=1,
        seed=42,
    )

    partitions = result["partitions"]
    history = result["history"]

    # -------------------------------------------------
    # Verificar clientes
    # -------------------------------------------------

    assert len(partitions) >= 2

    print(f"Clientes participantes: {len(partitions)}")

    for client_id, client in partitions.items():
        print(
            f"{client_id}: "
            f"{client.num_examples} ejemplos locales"
        )

    # -------------------------------------------------
    # Verificar cantidad de rondas
    # -------------------------------------------------

    assert len(history) == NUM_ROUNDS

    print(f"Rondas requeridas: {NUM_ROUNDS}")
    print(f"Rondas completadas: {len(history)}")

    # -------------------------------------------------
    # Verificar cada ronda
    # -------------------------------------------------

    for record in history:

        assert record["num_examples"] == 24
        assert record["changed_parameters"] > 0

        print(
            f"Ronda {record['round']}: "
            f"loss={record['loss']:.6f}, "
            f"accuracy={record['accuracy']:.6f}, "
            f"num_examples={record['num_examples']}, "
            f"parametros_actualizados="
            f"{record['changed_parameters']}"
        )

    print(
        "Las 5 rondas produjeron una "
        "actualizacion del modelo global."
    )

    print(
        "Estado: ENTRENAMIENTO FEDERADO OPERATIVO"
    )


if __name__ == "__main__":
    main()