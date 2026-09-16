import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.federated.client_data import (
    ClientPartition,
    create_synthetic_client_partitions,
)

from src.federated.model import (
    create_keras_model,
)

from src.federated.server import (
    build_federated_process,
)

from src.federated.training import (
    execute_federated_training,
)


# =========================================================
# CONFIGURACION DEL EXPERIMENTO
# =========================================================

# Cantidad de ejemplos de cada cliente.
#
# Ejemplos:
# [80, 20]
# [70, 30]
# [60, 30, 10]
# [250, 150, 100]
CLIENT_SIZES = [6, 6]

NUM_ROUNDS = 5

BATCH_SIZE = 4

LOCAL_EPOCHS = 1

SEED = 42


OUTPUT_DIR = Path(
    "results/re1.3"
)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

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


def model_weight_norm(model_weights):
    """
    Calcula la norma L2 de todos los parámetros
    entrenables del modelo.
    """

    squared_sum = 0.0

    for weight in model_weights.trainable:

        array = np.asarray(
            weight,
            dtype=np.float64,
        )

        squared_sum += np.sum(
            array ** 2
        )

    return float(
        np.sqrt(squared_sum)
    )


def evaluate_model_weights(
    model_weights,
    partitions,
):
    """
    Evalúa el modelo global sobre las particiones
    sintéticas utilizadas en RE1.3.

    IMPORTANTE:
    Esta evaluación tiene propósito funcional.
    No corresponde a la evaluación experimental
    sobre UNSW-NB15 de OE2.
    """

    model = create_keras_model()

    keras_weights = [
        np.asarray(weight)
        for weight in model_weights.trainable
    ]

    keras_weights += [
        np.asarray(weight)
        for weight in model_weights.non_trainable
    ]

    model.set_weights(
        keras_weights
    )

    total_examples = sum(
        client.num_examples
        for client in partitions.values()
    )

    weighted_loss = 0.0
    weighted_accuracy = 0.0

    epsilon = 1e-7

    for client in partitions.values():

        predictions = model(
            client.features,
            training=False,
        ).numpy()

        predictions = np.clip(
            predictions,
            epsilon,
            1.0 - epsilon,
        )

        labels = client.labels

        loss = -np.mean(
            labels * np.log(predictions)
            + (1.0 - labels)
            * np.log(
                1.0 - predictions
            )
        )

        predicted_labels = (
            predictions >= 0.5
        ).astype(
            np.float32
        )

        accuracy = np.mean(
            predicted_labels
            == labels
        )

        weight = (
            client.num_examples
            / total_examples
        )

        weighted_loss += (
            weight * loss
        )

        weighted_accuracy += (
            weight * accuracy
        )

    return {
        "loss": float(
            weighted_loss
        ),
        "accuracy": float(
            weighted_accuracy
        ),
    }


def print_model_parameters(
    title,
    model_weights,
):
    """
    Muestra los parámetros entrenables del modelo.
    """

    print()
    print(title)

    for index, weight in enumerate(
        model_weights.trainable,
        start=1,
    ):

        array = np.asarray(
            weight
        )

        print(
            f"Parametro {index}: "
            f"{array.flatten()}"
        )

    print(
        "Norma L2 del modelo: "
        f"{model_weight_norm(model_weights):.8f}"
    )


# =========================================================
# VERIFICAR PARTICIONES
# =========================================================

def verify_client_distribution():

    print()
    print(
        "=== DISTRIBUCION DE CLIENTES ==="
    )

    num_clients = len(
        CLIENT_SIZES
    )

    partitions = (
        create_synthetic_client_partitions(
            num_clients=num_clients,
            samples_per_client=CLIENT_SIZES,
            seed=SEED,
        )
    )

    assert len(partitions) == num_clients

    total_examples = sum(
        CLIENT_SIZES
    )

    print(
        f"Clientes: {num_clients}"
    )

    print(
        f"Total de ejemplos: "
        f"{total_examples}"
    )

    print()

    for index, (
        client_id,
        client,
    ) in enumerate(
        partitions.items()
    ):

        expected = CLIENT_SIZES[
            index
        ]

        assert (
            client.num_examples
            == expected
        )

        weight = (
            client.num_examples
            / total_examples
        )

        print(
            f"{client_id}: "
            f"{client.num_examples} ejemplos "
            f"-> peso FedAvg = "
            f"{weight:.4f} "
            f"({weight * 100:.2f} %)"
        )

    return partitions


# =========================================================
# VERIFICAR PONDERACION FEDAVG
# =========================================================

def verify_fedavg_weighting():

    print()
    print(
        "=== VERIFICACION MATEMATICA "
        "DE FEDAVG ==="
    )

    total_examples = sum(
        CLIENT_SIZES
    )

    controlled_datasets = []

    # Crear clientes controlados.
    #
    # Cliente 1 -> labels 1
    # Cliente 2 -> labels 0
    # Cliente 3 -> labels 1
    # ...
    #
    # Features = 0 para aislar el efecto
    # sobre el bias.

    for index, client_size in enumerate(
        CLIENT_SIZES
    ):

        label_value = (
            1.0
            if index % 2 == 0
            else 0.0
        )

        client = ClientPartition(
            client_id=(
                f"controlled_"
                f"client_{index + 1:02d}"
            ),
            features=np.zeros(
                (
                    client_size,
                    2,
                ),
                dtype=np.float32,
            ),
            labels=np.full(
                (
                    client_size,
                    1,
                ),
                label_value,
                dtype=np.float32,
            ),
        )

        dataset = (
            client.to_tf_dataset(
                batch_size=client_size,
                local_epochs=1,
                shuffle=False,
            )
        )

        controlled_datasets.append(
            dataset
        )

    process = (
        build_federated_process(
            client_learning_rate=0.1,
            server_learning_rate=1.0,
        )
    )

    initial_state = (
        process.initialize()
    )

    initial_bias = get_bias(
        process,
        initial_state,
    )

    individual_deltas = []

    print()

    for index, dataset in enumerate(
        controlled_datasets
    ):

        output = process.next(
            initial_state,
            [dataset],
        )

        local_bias = get_bias(
            process,
            output.state,
        )

        delta = (
            local_bias
            - initial_bias
        )

        individual_deltas.append(
            delta
        )

        weight = (
            CLIENT_SIZES[index]
            / total_examples
        )

        print(
            f"Cliente {index + 1:02d}: "
            f"peso={weight:.4f}, "
            f"delta={delta:.8f}"
        )

    expected_delta = 0.0

    print()
    print(
        "Contribuciones:"
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

        expected_delta += (
            contribution
        )

        print(
            f"Cliente {index + 1:02d}: "
            f"{weight:.4f} "
            f"x {delta:.8f} "
            f"= {contribution:.8f}"
        )

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
        f"Delta esperado: "
        f"{expected_delta:.8f}"
    )

    print(
        f"Delta obtenido: "
        f"{obtained_delta:.8f}"
    )

    print(
        "Diferencia: "
        f"{abs(expected_delta - obtained_delta):.10f}"
    )

    assert np.isclose(
        obtained_delta,
        expected_delta,
        atol=1e-5,
    )

    print(
        "Ponderacion FedAvg: OK"
    )


# =========================================================
# GUARDAR HISTORIA
# =========================================================

def save_history_csv(history):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / "fedavg_training_history.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "round",
                "loss",
                "accuracy",
                "num_examples",
                "num_batches",
                "changed_parameters",
            ],
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    return path


# =========================================================
# GENERAR GRAFICAS
# =========================================================

def save_training_plot(history):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rounds = [
        record["round"]
        for record in history
    ]

    loss = [
        record["loss"]
        for record in history
    ]

    accuracy = [
        record["accuracy"]
        for record in history
    ]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    # Loss

    axes[0].plot(
        rounds,
        loss,
    )

    axes[0].set_title(
        "Evolución de la pérdida (loss)"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].grid(
        alpha=0.3
    )

    # Accuracy

    axes[1].plot(
        rounds,
        accuracy,
    )

    axes[1].set_title(
        "Evolución de la exactitud (accuracy)"
    )

    axes[1].set_xlabel(
        "Ronda federada"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].set_ylim(
        0.0,
        1.0,
    )

    axes[1].grid(
        alpha=0.3
    )

    figure.tight_layout()

    path = (
        OUTPUT_DIR
        / "fedavg_training_curves.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return path


# =========================================================
# MOSTRAR SOLO CHECKPOINTS
# =========================================================

def print_training_checkpoints(
    history,
):

    print()
    print(
        "=== CHECKPOINTS DEL ENTRENAMIENTO ==="
    )

    total = len(
        history
    )

    indexes = {
        0,
        round(
            (total - 1) * 0.25
        ),
        round(
            (total - 1) * 0.50
        ),
        round(
            (total - 1) * 0.75
        ),
        total - 1,
    }

    for index in sorted(
        indexes
    ):

        record = history[
            index
        ]

        print(
            f"Ronda "
            f"{record['round']:>4}: "
            f"loss="
            f"{record['loss']:.6f}, "
            f"accuracy="
            f"{record['accuracy']:.6f}, "
            f"parametros_actualizados="
            f"{record['changed_parameters']}"
        )


# =========================================================
# ENTRENAMIENTO COMPLETO
# =========================================================

def verify_federated_training():

    print()
    print(
        "=== ENTRENAMIENTO FEDERADO ==="
    )

    num_clients = len(
        CLIENT_SIZES
    )

    total_examples = sum(
        CLIENT_SIZES
    )

    expected_examples = (
        total_examples
        * LOCAL_EPOCHS
    )

    result = (
        execute_federated_training(
            num_rounds=NUM_ROUNDS,
            num_clients=num_clients,
            samples_per_client=CLIENT_SIZES,
            batch_size=BATCH_SIZE,
            local_epochs=LOCAL_EPOCHS,
            seed=SEED,
        )
    )

    partitions = result[
        "partitions"
    ]

    history = result[
        "history"
    ]

    initial_weights = result[
        "initial_weights"
    ]

    final_weights = result[
        "final_weights"
    ]

    # -----------------------------------------------------
    # Verificaciones globales
    # -----------------------------------------------------

    assert (
        len(history)
        == NUM_ROUNDS
    )

    # No se imprime num_examples ronda por ronda.
    #
    # Se comprueba que TODAS las rondas procesaron
    # la cantidad esperada.

    assert all(
        record["num_examples"]
        == expected_examples
        for record in history
    )

    # La primera ronda sí debe producir
    # actualización.

    assert (
        history[0][
            "changed_parameters"
        ]
        > 0
    )

    rounds_with_update = sum(
        record[
            "changed_parameters"
        ] > 0
        for record in history
    )

    rounds_without_update = (
        NUM_ROUNDS
        - rounds_with_update
    )

    # -----------------------------------------------------
    # Evaluacion real del modelo global:
    # antes y despues del entrenamiento.
    # -----------------------------------------------------

    initial_evaluation = (
        evaluate_model_weights(
            initial_weights,
            partitions,
        )
    )

    final_evaluation = (
        evaluate_model_weights(
            final_weights,
            partitions,
        )
    )

    # -----------------------------------------------------
    # Salida resumida
    # -----------------------------------------------------

    print()
    print(
        f"Rondas ejecutadas: "
        f"{NUM_ROUNDS}"
    )

    print(
        f"Ejemplos procesados "
        f"por ronda: "
        f"{expected_examples}"
    )

    print(
        f"Rondas con cambio "
        f"detectable del modelo: "
        f"{rounds_with_update}"
    )

    print(
        f"Rondas sin cambio "
        f"detectable: "
        f"{rounds_without_update}"
    )

    # -----------------------------------------------------
    # Checkpoints
    # -----------------------------------------------------

    print_training_checkpoints(
        history
    )

    # -----------------------------------------------------
    # Comparación inicial/final REAL
    # -----------------------------------------------------

    print()
    print(
        "=== MODELO GLOBAL: "
        "ANTES VS DESPUES ==="
    )

    print()

    print(
        "ANTES DEL ENTRENAMIENTO"
    )

    print(
        f"Loss: "
        f"{initial_evaluation['loss']:.6f}"
    )

    print(
        f"Accuracy: "
        f"{initial_evaluation['accuracy']:.6f} "
        f"("
        f"{initial_evaluation['accuracy'] * 100:.2f} %)"
    )

    print()

    print(
        "DESPUES DEL ENTRENAMIENTO"
    )

    print(
        f"Loss: "
        f"{final_evaluation['loss']:.6f}"
    )

    print(
        f"Accuracy: "
        f"{final_evaluation['accuracy']:.6f} "
        f"("
        f"{final_evaluation['accuracy'] * 100:.2f} %)"
    )

    print()

    accuracy_change = (
        final_evaluation[
            "accuracy"
        ]
        - initial_evaluation[
            "accuracy"
        ]
    )

    loss_change = (
        final_evaluation[
            "loss"
        ]
        - initial_evaluation[
            "loss"
        ]
    )

    print(
        "Cambio de accuracy: "
        f"{accuracy_change * 100:+.2f} "
        "puntos porcentuales"
    )

    print(
        "Cambio de loss: "
        f"{loss_change:+.6f}"
    )

    # -----------------------------------------------------
    # Pesos reales del modelo
    # -----------------------------------------------------

    print_model_parameters(
        "PESOS INICIALES DEL MODELO",
        initial_weights,
    )

    print_model_parameters(
        "PESOS FINALES DEL MODELO",
        final_weights,
    )

    # -----------------------------------------------------
    # Guardar evidencia completa
    # -----------------------------------------------------

    csv_path = save_history_csv(
        history
    )

    graph_path = save_training_plot(
        history
    )

    print()
    print(
        "=== EVIDENCIAS GENERADAS ==="
    )

    print(
        f"Historial completo: "
        f"{csv_path}"
    )

    print(
        f"Grafica: "
        f"{graph_path}"
    )

    return result


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "============================================="
    )

    print(
        " EXPERIMENTO DE FEDAVG CON "
        "CLIENTES HETEROGENEOS"
    )

    print(
        "============================================="
    )

    print()

    print(
        f"Distribucion: "
        f"{CLIENT_SIZES}"
    )

    print(
        f"Rondas: "
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

    # 1.
    # Verificar tamaños y pesos
    # de los clientes.

    verify_client_distribution()

    # 2.
    # Demostrar matemáticamente
    # la ponderación de FedAvg.

    verify_fedavg_weighting()

    # 3.
    # Ejecutar entrenamiento completo,
    # analizar convergencia y
    # generar evidencias.

    verify_federated_training()

    print()
    print(
        "============================================="
    )

    print(
        " EXPERIMENTO COMPLETADO CORRECTAMENTE"
    )

    print(
        "============================================="
    )


if __name__ == "__main__":
    main()