from pathlib import Path

import numpy as np

from src.federated.client_data import (
    ClientPartition,
)


# ============================================================
# Configuración
# ============================================================

SEED = 42

EXPECTED_NUM_FEATURES = 192

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "unsw_nb15"
)


# ============================================================
# Carga de los conjuntos procesados
# ============================================================

def load_unsw_split(split_name):
    """
    Carga uno de los conjuntos procesados
    de UNSW-NB15.

    Valores permitidos:
    - train
    - validation
    - test
    """

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    if split_name not in valid_splits:
        raise ValueError(
            f"Split inválido: {split_name}. "
            f"Valores permitidos: "
            f"{sorted(valid_splits)}"
        )

    file_path = (
        DATA_DIR
        / f"{split_name}.npz"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "No se encontró el dataset "
            f"procesado: {file_path}"
        )

    with np.load(file_path) as data:

        X = np.asarray(
            data["X"],
            dtype=np.float32,
        )

        y = np.asarray(
            data["y"],
            dtype=np.int32,
        )

    _validate_unsw_split(
        split_name=split_name,
        X=X,
        y=y,
    )

    return X, y


# ============================================================
# Validación
# ============================================================

def _validate_unsw_split(
    split_name,
    X,
    y,
):
    """
    Verifica que el conjunto sea compatible
    con el modelo IDS de RE2.1.
    """

    if X.ndim != 2:
        raise ValueError(
            f"{split_name}: X debe ser "
            "una matriz bidimensional."
        )

    if y.ndim != 1:
        raise ValueError(
            f"{split_name}: y debe ser "
            "un vector."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{split_name}: X e y deben "
            "contener la misma cantidad "
            "de registros."
        )

    if len(X) == 0:
        raise ValueError(
            f"{split_name}: el conjunto "
            "no puede estar vacío."
        )

    if (
        X.shape[1]
        != EXPECTED_NUM_FEATURES
    ):
        raise ValueError(
            f"{split_name}: se esperaban "
            f"{EXPECTED_NUM_FEATURES} "
            "características, pero se "
            f"encontraron {X.shape[1]}."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{split_name}: existen "
            "características no finitas."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{split_name}: existen "
            "etiquetas no finitas."
        )

    unique_labels = set(
        np.unique(y).tolist()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"{split_name}: se esperaban "
            "únicamente las clases 0 y 1, "
            "pero se encontraron "
            f"{sorted(unique_labels)}."
        )


# ============================================================
# Particionado estratificado
# ============================================================

def build_stratified_client_indices(
    labels,
    num_clients=2,
    seed=SEED,
):
    """
    Distribuye los índices del conjunto TRAIN
    entre los clientes de forma estratificada.

    Cada registro se asigna exactamente a un
    cliente y no existen registros duplicados.
    """

    if num_clients < 2:
        raise ValueError(
            "El prototipo federado requiere "
            "al menos dos clientes."
        )

    labels = np.asarray(
        labels
    ).reshape(-1)

    if len(labels) < num_clients:
        raise ValueError(
            "No existen suficientes registros "
            "para la cantidad de clientes."
        )

    unique_labels = set(
        np.unique(labels).tolist()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):
        raise ValueError(
            "Las etiquetas deben pertenecer "
            "a las clases 0 y 1."
        )

    rng = np.random.default_rng(
        seed
    )

    client_groups = [
        []
        for _ in range(num_clients)
    ]

    # --------------------------------------------------------
    # Distribuir cada clase entre todos los clientes
    # --------------------------------------------------------

    for class_value in (0, 1):

        class_indices = np.flatnonzero(
            labels == class_value
        )

        if len(class_indices) < num_clients:
            raise ValueError(
                f"La clase {class_value} no "
                "posee suficientes registros "
                "para todos los clientes."
            )

        rng.shuffle(
            class_indices
        )

        class_splits = np.array_split(
            class_indices,
            num_clients,
        )

        for client_index, split in enumerate(
            class_splits
        ):
            client_groups[
                client_index
            ].append(split)

    # --------------------------------------------------------
    # Construir los índices finales
    # --------------------------------------------------------

    client_indices = []

    for groups in client_groups:

        indices = np.concatenate(
            groups
        )

        rng.shuffle(
            indices
        )

        client_indices.append(
            indices
        )

    # --------------------------------------------------------
    # Verificar cobertura y ausencia de duplicados
    # --------------------------------------------------------

    all_indices = np.concatenate(
        client_indices
    )

    if len(all_indices) != len(labels):
        raise RuntimeError(
            "El particionado no cubre todo "
            "el conjunto de entrenamiento."
        )

    if (
        len(np.unique(all_indices))
        != len(labels)
    ):
        raise RuntimeError(
            "Existen registros asignados "
            "a más de un cliente."
        )

    if not np.array_equal(
        np.sort(all_indices),
        np.arange(len(labels)),
    ):
        raise RuntimeError(
            "El particionado contiene índices "
            "inválidos o faltantes."
        )

    return client_indices


# ============================================================
# Creación de clientes federados
# ============================================================

def create_unsw_client_partitions(
    num_clients=2,
    seed=SEED,
):
    """
    Crea las particiones locales de UNSW-NB15.

    Solamente utiliza el conjunto TRAIN.
    Cada cliente recibe su propio subconjunto.
    """

    X_train, y_train = load_unsw_split(
        "train"
    )

    client_indices = (
        build_stratified_client_indices(
            labels=y_train,
            num_clients=num_clients,
            seed=seed,
        )
    )

    partitions = {}

    for index, indices in enumerate(
        client_indices,
        start=1,
    ):

        client_id = (
            f"client_{index:02d}"
        )

        partitions[
            client_id
        ] = ClientPartition(
            client_id=client_id,

            features=X_train[
                indices
            ],

            labels=y_train[
                indices
            ],
        )

    return partitions


# ============================================================
# Conjunto TEST común
# ============================================================

def load_common_test_set():
    """
    Carga el conjunto TEST oficial.

    Este conjunto no participa en el
    entrenamiento federado.
    """

    return load_unsw_split(
        "test"
    )