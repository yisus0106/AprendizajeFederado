import numpy as np

from src.federated.server import build_federated_process


def main():
    print("=== PRUEBA DEL SERVIDOR FEDERADO ===")

    # Construcción del proceso federado
    federated_process = build_federated_process()

    print("Proceso FedAvg construido correctamente.")

    # Inicialización del estado global
    state = federated_process.initialize()

    print("Estado global inicializado correctamente.")

    # Recuperar los parámetros globales
    model_weights = federated_process.get_model_weights(state)

    assert len(model_weights.trainable) > 0

    print(
        f"Variables entrenables del modelo global: "
        f"{len(model_weights.trainable)}"
    )

    for index, weight in enumerate(model_weights.trainable):
        weight_array = np.asarray(weight)

        assert np.isfinite(weight_array).all()

        print(
            f"Parametro global {index + 1}: "
            f"shape={weight_array.shape} - OK"
        )

    print("Estado: SERVIDOR Y FEDAVG OPERATIVOS")


if __name__ == "__main__":
    main()