from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import time
import uuid

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from src.federated_re1_3.client_data import ClientPartition
from src.federated.model import create_keras_model
from src.federated.server import build_federated_process


BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_TRAIN_CONFIG = (
    BASE_DIR / "src" / "models" / "federated" / "train_config.json"
)


# ============================================================
# Utilidades
# ============================================================

def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"{path}: se esperaba un objeto JSON.")

    return value


def write_json(path, content):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} debe ser un entero positivo.")

    return value


def load_training_config(path):
    config = read_json(path)

    required = (
        "partitions_dir",
        "model_config_path",
        "metadata_path",
        "output_root",
        "num_rounds",
        "local_epochs",
        "server_learning_rate",
        "seed",
    )

    missing = [key for key in required if key not in config]

    if missing:
        raise ValueError(
            f"Faltan campos en train_config.json: {missing}"
        )

    positive_integer(config["num_rounds"], "num_rounds")
    positive_integer(config["local_epochs"], "local_epochs")

    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed debe ser un entero entre 0 y 2**32 - 1.")

    return config


# ============================================================
# Carga de particiones ya generadas
# ============================================================

def load_clients(partitions_dir, input_features):
    """
    Carga exclusivamente los archivos de clientes del manifiesto.

    No abre train.npz, validation.npz ni test.npz.
    """
    manifest_path = partitions_dir / "manifest.json"
    manifest = read_json(manifest_path)

    if manifest.get("test_used") is not False:
        raise ValueError(
            "El manifiesto no declara test_used=False."
        )

    if manifest.get("pool") not in ("train", "train_validation"):
        raise ValueError("El manifiesto indica un conjunto no admitido.")

    if manifest["input_features"] != input_features:
        raise ValueError(
            "Las características de las particiones no coinciden "
            "con la entrada del modelo."
        )

    entries = manifest["clients"]

    if len(entries) != manifest["num_clients"] or len(entries) < 2:
        raise ValueError("Número de clientes inválido en el manifiesto.")

    clients = []
    assigned_indices = []
    fingerprints = {}
    seen_ids = set()

    for entry in entries:
        client_id = entry["client_id"]

        if (
            not isinstance(client_id, str)
            or not client_id
            or Path(client_id).name != client_id
            or "/" in client_id
            or "\\" in client_id
            or client_id in seen_ids
        ):
            raise ValueError(f"Identificador de cliente inválido: {client_id}")

        seen_ids.add(client_id)
        path = partitions_dir / f"{client_id}.npz"

        with np.load(path, allow_pickle=False) as data:
            X = data["X"]
            y = data["y"]
            indices = data["indices"]

        if X.ndim != 2 or X.shape[1] != input_features:
            raise ValueError(f"{client_id}: dimensiones de X incorrectas.")

        if y.ndim != 1 or not np.isin(y, [0, 1]).all():
            raise ValueError(f"{client_id}: etiquetas binarias inválidas.")

        if len(X) != len(y) or len(y) != entry["samples"]:
            raise ValueError(f"{client_id}: cantidad de registros incorrecta.")

        if (
            indices.ndim != 1
            or len(indices) != len(y)
            or not np.issubdtype(indices.dtype, np.integer)
        ):
            raise ValueError(f"{client_id}: índices inválidos.")

        benign = int(np.count_nonzero(y == 0))
        malicious = int(np.count_nonzero(y == 1))

        if benign != entry["benign"] or malicious != entry["malicious"]:
            raise ValueError(
                f"{client_id}: distribución distinta del manifiesto."
            )

        client = ClientPartition(
            client_id=client_id,
            features=X,
            labels=y,
        )

        clients.append(client)
        assigned_indices.append(indices)
        fingerprints[path.name] = sha256_file(path)

    # Cobertura exacta de las filas del conjunto repartido.
    assigned_indices = np.concatenate(assigned_indices)
    expected = np.arange(manifest["total_samples"], dtype=np.int64)

    if not np.array_equal(np.sort(assigned_indices), expected):
        raise ValueError(
            "Las particiones contienen índices repetidos o faltantes."
        )

    fingerprints["manifest.json"] = sha256_file(manifest_path)

    return clients, manifest, fingerprints


# ============================================================
# Entrenamiento
# ============================================================

def train(config_path):
    training_config = load_training_config(config_path)
    seed = training_config["seed"]

    # Inicializa las semillas antes de construir el proceso TFF.
    tf.keras.utils.set_random_seed(seed)

    process, effective_config = build_federated_process(
        config_path=resolve_path(training_config["model_config_path"]),
        metadata_path=resolve_path(training_config["metadata_path"]),
        server_learning_rate=training_config["server_learning_rate"],
    )

    model_config = effective_config["model"]

    # El tamaño de lote procede de la configuración seleccionada.
    batch_size = positive_integer(
        model_config.get("batch_size"), "best_config.batch_size"
    )

    partitions_dir = resolve_path(training_config["partitions_dir"])

    clients, manifest, fingerprints = load_clients(
        partitions_dir=partitions_dir,
        input_features=model_config["input_features"],
    )

    num_rounds = training_config["num_rounds"]
    local_epochs = training_config["local_epochs"]
    total_samples = sum(client.num_examples for client in clients)

    # Todos los clientes participan en cada ronda.
    expected_examples = total_samples * local_epochs

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )

    run_dir = resolve_path(training_config["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    write_json(run_dir / "training_config.json", training_config)
    write_json(run_dir / "effective_config.json", effective_config)
    write_json(run_dir / "partition_manifest.json", manifest)
    write_json(run_dir / "partition_hashes.json", fingerprints)

    print(f"\nEjecución: {run_id}")
    print(f"Clientes: {len(clients)}")
    print(f"Registros únicos: {total_samples:,}")
    print(f"Características: {model_config['input_features']}")
    print(f"Rondas: {num_rounds}")
    print(f"Épocas locales por ronda: {local_epochs}")
    print(f"Tamaño de lote: {batch_size}")
    print(f"Salida: {run_dir.relative_to(BASE_DIR)}\n")

    summary = {
        "run_id": run_id,
        "status": "running",
        "num_clients": len(clients),
        "unique_training_samples": total_samples,
        "num_rounds_requested": num_rounds,
        "completed_rounds": 0,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "training_seed": seed,
        "all_clients_each_round": True,
        "test_loaded": False,
        "independent_evaluation_performed": False,
        "model_selection": "last_round",
        "versions": {
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "tensorflow_federated": tff.__version__,
        },
    }

    summary_path = run_dir / "run_summary.json"
    write_json(summary_path, summary)

    start_time = time.perf_counter()

    try:
        state = process.initialize()

        fields = [
            "round",
            "train_loss",
            "train_accuracy",
            "train_precision",
            "train_recall",
            "examples_processed",
            "round_seconds",
        ]

        with (run_dir / "training_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            for round_number in range(1, num_rounds + 1):
                round_start = time.perf_counter()

                # Las particiones permanecen fijas.
                # Solo cambia su orden de lectura en cada ronda.
                federated_datasets = []

                for client_number, client in enumerate(clients):
                    round_seed = int(
                        np.random.SeedSequence(
                            [seed, round_number, client_number]
                        ).generate_state(1)[0] % (2**31 - 1)
                    )

                    dataset = client.to_tf_dataset(
                        batch_size=batch_size,
                        local_epochs=local_epochs,
                        shuffle=True,
                        seed=round_seed,
                    )

                    federated_datasets.append(dataset)

                output = process.next(state, federated_datasets)
                state = output.state

                metrics = output.metrics["client_work"]["train"]

                row = {
                    "round": round_number,
                    "train_loss": float(metrics["loss"]),
                    "train_accuracy": float(metrics["accuracy"]),
                    "train_precision": float(metrics["precision"]),
                    "train_recall": float(metrics["recall"]),
                    "examples_processed": int(metrics["num_examples"]),
                    "round_seconds": time.perf_counter() - round_start,
                }

                metric_values = [
                    row["train_loss"],
                    row["train_accuracy"],
                    row["train_precision"],
                    row["train_recall"],
                ]

                if not np.isfinite(metric_values).all():
                    raise RuntimeError(
                        f"Ronda {round_number}: métricas no finitas."
                    )

                if row["examples_processed"] != expected_examples:
                    raise RuntimeError(
                        f"Ronda {round_number}: se procesaron "
                        f"{row['examples_processed']} ejemplos; "
                        f"se esperaban {expected_examples}."
                    )

                writer.writerow(row)
                file.flush()

                summary["completed_rounds"] = round_number
                write_json(summary_path, summary)

                print(
                    f"Ronda {round_number:03d}/{num_rounds:03d} | "
                    f"loss={row['train_loss']:.4f} | "
                    f"accuracy={row['train_accuracy']:.4f} | "
                    f"precision={row['train_precision']:.4f} | "
                    f"recall={row['train_recall']:.4f} | "
                    f"{row['round_seconds']:.1f} s"
                )

        # ====================================================
        # Exportación del modelo global de la última ronda
        # ====================================================

        keras_model = create_keras_model(model_config)
        global_weights = process.get_model_weights(state)
        global_weights.assign_weights_to(keras_model)

        for weight in keras_model.get_weights():
            if not np.isfinite(weight).all():
                raise RuntimeError(
                    "El modelo global contiene pesos no finitos."
                )

        model_path = run_dir / "global_model.keras"
        keras_model.save(str(model_path))

        summary.update({
            "status": "completed",
            "elapsed_seconds": time.perf_counter() - start_time,
            "model_file": model_path.name,
            "model_sha256": sha256_file(model_path),
            "parameters": int(keras_model.count_params()),
            "classification_threshold": (
                model_config["classification_threshold"]
            ),
        })

        write_json(summary_path, summary)

        training_history_path = run_dir / "training_history.csv"
        
        print(f"\nModelo global guardado en:\n{model_path.relative_to(BASE_DIR)}")
        print(f"Historial:\n{training_history_path.relative_to(BASE_DIR)}")

        return model_path

    except (Exception, KeyboardInterrupt) as error:
        summary.update({
            "status": "failed_or_interrupted",
            "elapsed_seconds": time.perf_counter() - start_time,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        write_json(summary_path, summary)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Entrena el IDS federado y exporta el modelo Keras."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TRAIN_CONFIG),
        help="Ruta al JSON de configuración del entrenamiento.",
    )

    args = parser.parse_args()
    train(resolve_path(args.config))


if __name__ == "__main__":
    main()
    
# EJEMPLO DE USO
# python src/models/federated/train_federated.py 
# --config src/models/federated/configuraciones/train_config2.json
# Si no se coloca un config, pues se usara por default src/models/federated/train_config.json