from pathlib import Path
import csv
import hashlib
import json

import numpy as np


# ============================================================
# Configuración
# ============================================================

NUM_CLIENTS = 3

NON_IID = False
ALPHA = 0.5             # Solo se utiliza cuando NON_IID = True.

SEED = 42
MIN_CLIENT_SAMPLES = 100
MAX_ATTEMPTS = 1000

# True: entrenamiento definitivo con TRAIN + VALIDATION.
# False: repartir solo TRAIN y reservar VALIDATION para ajuste.
INCLUDE_VALIDATION = True


# ============================================================
# Rutas
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "unsw_nb15"
OUTPUT_ROOT = DATA_DIR / "federated"


def sha256_file(path):
    """Identifica el contenido exacto de un archivo de entrada."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_source(path):
    """Carga y valida un NPZ generado por preprocess_unsw.py."""
    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]

    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError(f"{path.name}: X debe tener forma (n, features).")

    if y.ndim != 1:
        raise ValueError(f"{path.name}: y debe tener forma (n,).")

    if len(X) == 0 or len(X) != len(y):
        raise ValueError(f"{path.name}: tamaños de X e y inválidos.")

    if not np.isfinite(X).all():
        raise ValueError(f"{path.name}: X contiene valores no finitos.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{path.name}: solo se admiten etiquetas 0 y 1.")

    X = X.astype(np.float32, copy=False)
    y = y.astype(np.int32, copy=False)

    if not np.isfinite(X).all():
        raise ValueError(f"{path.name}: valores fuera del rango float32.")

    return X, y


def load_development_data():
    """
    Une las fuentes seleccionadas y registra su procedencia.

    source_split:
        0 = train.npz
        1 = validation.npz

    source_row:
        posición de la fila dentro del archivo correspondiente.
    """
    names = ["train"]

    if INCLUDE_VALIDATION:
        names.append("validation")

    features = []
    labels = []
    split_ids = []
    source_rows = []
    sources = []
    input_dim = None
    offset = 0

    for split_id, name in enumerate(names):
        path = DATA_DIR / f"{name}.npz"
        X, y = load_source(path)

        if input_dim is None:
            input_dim = X.shape[1]
        elif X.shape[1] != input_dim:
            raise ValueError(
                "TRAIN y VALIDATION tienen dimensiones diferentes."
            )

        features.append(X)
        labels.append(y)

        split_ids.append(
            np.full(len(y), split_id, dtype=np.int8)
        )
        source_rows.append(
            np.arange(len(y), dtype=np.int64)
        )

        sources.append({
            "split_id": split_id,
            "file": str(path.relative_to(BASE_DIR)),
            "samples": int(len(y)),
            "offset_in_pool": offset,
            "sha256": sha256_file(path),
        })

        offset += len(y)

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels)

    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError(
            "El conjunto global debe contener benignos y maliciosos."
        )

    return (
        X,
        y,
        np.concatenate(split_ids),
        np.concatenate(source_rows),
        sources,
    )


def partition_iid(y, rng):
    """
    Reparto aleatorio estratificado.

    Por clase, los tamaños difieren como máximo en una fila.
    La rotación de los sobrantes también equilibra el total.
    """
    buckets = [[] for _ in range(NUM_CLIENTS)]
    offset = 0

    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)

        chunks = np.array_split(indices, NUM_CLIENTS)

        for position, chunk in enumerate(chunks):
            client = (offset + position) % NUM_CLIENTS
            buckets[client].append(chunk)

        offset = (offset + len(indices)) % NUM_CLIENTS

    return [
        rng.permutation(np.concatenate(parts))
        for parts in buckets
    ]


def partition_non_iid(y, rng):
    """
    Dirichlet por etiqueta.

    Cada clase tiene su propio vector de proporciones.
    Multinomial convierte esas proporciones en cantidades
    enteras cuya suma conserva todos los registros.

    Se acepta el primer reparto que cumple el tamaño mínimo.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        buckets = [[] for _ in range(NUM_CLIENTS)]

        for label in (0, 1):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)

            proportions = rng.dirichlet(
                np.full(NUM_CLIENTS, ALPHA)
            )
            counts = rng.multinomial(
                len(indices), proportions
            )

            boundaries = np.cumsum(counts)[:-1]
            chunks = np.split(indices, boundaries)

            for client, chunk in enumerate(chunks):
                buckets[client].append(chunk)

        partitions = [
            np.concatenate(parts)
            for parts in buckets
        ]

        if min(map(len, partitions)) >= MIN_CLIENT_SAMPLES:
            return [
                rng.permutation(indices)
                for indices in partitions
            ], attempt

    raise RuntimeError(
        f"No se obtuvo un reparto válido en {MAX_ATTEMPTS} intentos. "
        "Revisa ALPHA, NUM_CLIENTS o MIN_CLIENT_SAMPLES."
    )


def verify_partitions(partitions, total_samples):
    """Comprueba cobertura, exclusividad y tamaños mínimos."""
    if len(partitions) != NUM_CLIENTS:
        raise ValueError("Número incorrecto de particiones.")

    if any(len(p) < MIN_CLIENT_SAMPLES for p in partitions):
        raise ValueError(
            "Algún cliente no cumple MIN_CLIENT_SAMPLES."
        )

    assigned = np.concatenate(partitions)
    expected = np.arange(total_samples, dtype=np.int64)

    if not np.array_equal(np.sort(assigned), expected):
        raise ValueError(
            "Hay índices repetidos, faltantes o fuera de rango."
        )


def main():
    if not isinstance(NUM_CLIENTS, int) or NUM_CLIENTS < 2:
        raise ValueError("NUM_CLIENTS debe ser un entero >= 2.")

    if not isinstance(NON_IID, bool):
        raise ValueError("NON_IID debe ser True o False.")

    if not isinstance(INCLUDE_VALIDATION, bool):
        raise ValueError("INCLUDE_VALIDATION debe ser True o False.")

    if (
        not isinstance(MIN_CLIENT_SAMPLES, int)
        or MIN_CLIENT_SAMPLES < 1
    ):
        raise ValueError("MIN_CLIENT_SAMPLES debe ser un entero >= 1.")

    if not isinstance(SEED, int) or SEED < 0:
        raise ValueError("SEED debe ser un entero no negativo.")

    if NON_IID:
        if not np.isfinite(ALPHA) or ALPHA <= 0:
            raise ValueError("ALPHA debe ser finito y mayor que cero.")

        if not isinstance(MAX_ATTEMPTS, int) or MAX_ATTEMPTS < 1:
            raise ValueError("MAX_ATTEMPTS debe ser un entero >= 1.")

    pool = "train_validation" if INCLUDE_VALIDATION else "train"
    mode = f"non_iid_alpha_{ALPHA}" if NON_IID else "iid"

    scenario = (
        f"{pool}_{mode}_clients_{NUM_CLIENTS}"
        f"_seed_{SEED}_min_{MIN_CLIENT_SAMPLES}"
    )
    output_dir = OUTPUT_ROOT / scenario

    if output_dir.exists():
        raise FileExistsError(
            f"El escenario ya existe: {output_dir}\n"
            "Se cancela para conservar las particiones anteriores."
        )

    X, y, source_split, source_row, sources = load_development_data()

    if NUM_CLIENTS * MIN_CLIENT_SAMPLES > len(y):
        raise ValueError(
            "No hay suficientes registros para satisfacer "
            "el tamaño mínimo de todos los clientes."
        )

    rng = np.random.default_rng(SEED)

    if NON_IID:
        partitions, attempts = partition_non_iid(y, rng)
    else:
        partitions = partition_iid(y, rng)
        attempts = 1

    verify_partitions(partitions, len(y))

    # La creación exclusiva también impide sobrescrituras.
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = []

    for number, indices in enumerate(partitions, start=1):
        client_id = f"client_{number:02d}"
        local_y = y[indices]

        benign = int(np.count_nonzero(local_y == 0))
        malicious = int(np.count_nonzero(local_y == 1))
        total = len(indices)

        np.savez_compressed(
            output_dir / f"{client_id}.npz",
            X=X[indices],
            y=local_y,
            indices=indices,
            source_split=source_split[indices],
            source_row=source_row[indices],
        )

        summary.append({
            "client_id": client_id,
            "samples": total,
            "benign": benign,
            "malicious": malicious,
            "benign_percentage": 100.0 * benign / total,
            "malicious_percentage": 100.0 * malicious / total,
        })

    with (output_dir / "distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary[0])
        )
        writer.writeheader()
        writer.writerows(summary)

    manifest = {
        "dataset": "UNSW-NB15",
        "partition_version": 1,
        "pool": pool,
        "num_clients": NUM_CLIENTS,
        "non_iid": NON_IID,
        "method": (
            "label_dirichlet_multinomial"
            if NON_IID else "random_stratified"
        ),
        "alpha": ALPHA if NON_IID else None,
        "seed": SEED,
        "min_client_samples": MIN_CLIENT_SAMPLES,
        "max_attempts": MAX_ATTEMPTS if NON_IID else None,
        "attempts_used": attempts,
        "numpy_version": np.__version__,
        "total_samples": int(len(y)),
        "input_features": int(X.shape[1]),
        "sources": sources,
        "test_used": False,
        "preprocessing_refitted": False,
        "verification": {
            "all_rows_assigned_once": True,
            "minimum_size_satisfied": True,
        },
        "clients": summary,
    }

    # Registra también los artefactos del preprocesamiento,
    # cuando están disponibles.
    manifest["preprocessing_artifacts"] = {}

    for name in ("metadata.json", "preprocessor.joblib"):
        path = DATA_DIR / name
        if path.is_file():
            manifest["preprocessing_artifacts"][name] = sha256_file(path)

    with (output_dir / "manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"\nEscenario: {scenario}")
    print(f"Registros: {len(y):,}")
    print(f"Características: {X.shape[1]}")
    print(f"Intentos: {attempts}\n")

    for row in summary:
        print(
            f"{row['client_id']}: "
            f"{row['samples']:,} registros | "
            f"benignos={row['benign']:,} | "
            f"maliciosos={row['malicious']:,} "
            f"({row['malicious_percentage']:.2f} %)"
        )

    print(f"\nParticiones guardadas en:\n{output_dir.relative_to(BASE_DIR)}")
    print("Verificación: todas las filas asignadas exactamente una vez.")


if __name__ == "__main__":
    main()