import numpy as np

from src.federated.training import execute_single_round


def main():
    print("=== PRUEBA DE UNA RONDA FEDERADA ===")

    result = execute_single_round()

    partitions = result["partitions"]

    # -------------------------------------------------
    # Verificar participación de clientes
    # -------------------------------------------------

    assert len(partitions) >= 2

    print(
        f"Clientes participantes: {len(partitions)}"
    )

    for client_id, client in partitions.items():
        print(
            f"{client_id}: "
            f"{client.num_examples} ejemplos locales"
        )

    # -------------------------------------------------
    # Verificar cambio del modelo global
    # -------------------------------------------------

    initial_weights = result["initial_weights"]
    updated_weights = result["updated_weights"]

    assert len(initial_weights) == len(updated_weights)

    changed_parameters = 0

    for index, (initial, updated) in enumerate(
        zip(initial_weights, updated_weights)
    ):
        changed = not np.allclose(
            initial,
            updated,
        )

        if changed:
            changed_parameters += 1

        print(
            f"Parametro {index + 1}: "
            f"actualizado={changed}"
        )

    assert changed_parameters > 0

    print(
        f"Parametros modificados: "
        f"{changed_parameters}/{len(initial_weights)}"
    )

    # -------------------------------------------------
    # Mostrar métricas generadas por TFF
    # -------------------------------------------------

    print("Metricas de la ronda:")
    print(result["metrics"])

    print("Ronda federada completada correctamente.")
    print("Estado: INTEGRACION FEDERADA OPERATIVA")


if __name__ == "__main__":
    main()