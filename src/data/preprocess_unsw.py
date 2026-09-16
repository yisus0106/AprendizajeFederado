from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Configuración
# ============================================================

SEED = 42
VALIDATION_SIZE = 0.20

TARGET = "label"

CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "state",
]

DROP_COLUMNS = [
    "id",
    "attack_cat",
    TARGET,
]


# ============================================================
# Rutas
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "unsw_nb15"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "unsw_nb15"

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Funciones auxiliares
# ============================================================

def class_distribution(y):
    """
    Obtiene cantidad y porcentaje de cada clase.
    """

    values, counts = np.unique(
        y,
        return_counts=True,
    )

    total = len(y)

    return {
        str(int(value)): {
            "count": int(count),
            "percentage": round(
                (int(count) / total) * 100,
                4,
            ),
        }
        for value, count in zip(
            values,
            counts,
        )
    }


def verify_processed_dataset(
    name,
    X,
    y,
):
    """
    Verifica que el conjunto procesado sea válido.
    """

    if X.ndim != 2:
        raise ValueError(
            f"{name}: X debe ser bidimensional."
        )

    if y.ndim != 1:
        raise ValueError(
            f"{name}: y debe ser unidimensional."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{name}: X e y tienen distinto "
            "número de registros."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{name}: existen valores no finitos "
            "en las características."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{name}: existen valores no finitos "
            "en las etiquetas."
        )


# ============================================================
# Carga de datos
# ============================================================

print("=" * 70)
print("PREPROCESAMIENTO - UNSW-NB15")
print("=" * 70)

print("\nCargando datasets originales...")

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

print(
    f"TRAIN oficial: {train_df.shape}"
)

print(
    f"TEST oficial:  {test_df.shape}"
)


# ============================================================
# Separación de características y objetivo
# ============================================================

X_development = train_df.drop(
    columns=DROP_COLUMNS
)

y_development = train_df[
    TARGET
].to_numpy(
    dtype=np.int32
)


X_test = test_df.drop(
    columns=DROP_COLUMNS
)

y_test = test_df[
    TARGET
].to_numpy(
    dtype=np.int32
)


# ============================================================
# TRAIN / VALIDATION
# ============================================================

print(
    "\nGenerando TRAIN y VALIDATION "
    "mediante partición estratificada..."
)

(
    X_train,
    X_validation,
    y_train,
    y_validation,
) = train_test_split(
    X_development,
    y_development,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=y_development,
)


print(
    f"TRAIN interno:      {X_train.shape}"
)

print(
    f"VALIDATION interno: {X_validation.shape}"
)

print(
    f"TEST oficial:       {X_test.shape}"
)


# ============================================================
# Identificación de columnas
# ============================================================

numeric_columns = [
    column
    for column in X_train.columns
    if column not in CATEGORICAL_COLUMNS
]


print(
    "\nCaracterísticas originales utilizadas:"
)

print(
    f"Numéricas:    {len(numeric_columns)}"
)

print(
    f"Categóricas: {len(CATEGORICAL_COLUMNS)}"
)

print(
    f"Total:        {X_train.shape[1]}"
)


print(
    "\nColumnas categóricas:"
)

print(
    CATEGORICAL_COLUMNS
)


# ============================================================
# Pipeline de preprocesamiento
# ============================================================

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=np.float32,
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_COLUMNS,
        ),
    ]
)


# ============================================================
# FIT EXCLUSIVAMENTE SOBRE TRAIN
# ============================================================

print(
    "\nAjustando preprocesador "
    "EXCLUSIVAMENTE con TRAIN..."
)

X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)


# ============================================================
# Transformación de VALIDATION y TEST
# ============================================================

print(
    "Transformando VALIDATION..."
)

X_validation_processed = (
    preprocessor.transform(
        X_validation
    )
)


print(
    "Transformando TEST..."
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


# ============================================================
# Conversión a float32
# ============================================================

X_train_processed = (
    X_train_processed.astype(
        np.float32
    )
)

X_validation_processed = (
    X_validation_processed.astype(
        np.float32
    )
)

X_test_processed = (
    X_test_processed.astype(
        np.float32
    )
)


# ============================================================
# Verificaciones
# ============================================================

verify_processed_dataset(
    "TRAIN",
    X_train_processed,
    y_train,
)

verify_processed_dataset(
    "VALIDATION",
    X_validation_processed,
    y_validation,
)

verify_processed_dataset(
    "TEST",
    X_test_processed,
    y_test,
)


print(
    "\nResultado del preprocesamiento:"
)

print(
    f"X_train:      "
    f"{X_train_processed.shape}"
)

print(
    f"y_train:      "
    f"{y_train.shape}"
)

print(
    f"X_validation: "
    f"{X_validation_processed.shape}"
)

print(
    f"y_validation: "
    f"{y_validation.shape}"
)

print(
    f"X_test:       "
    f"{X_test_processed.shape}"
)

print(
    f"y_test:       "
    f"{y_test.shape}"
)


print(
    "\nDistribución TRAIN:"
)

print(
    class_distribution(
        y_train
    )
)


print(
    "\nDistribución VALIDATION:"
)

print(
    class_distribution(
        y_validation
    )
)


print(
    "\nDistribución TEST:"
)

print(
    class_distribution(
        y_test
    )
)


# ============================================================
# Guardado de datasets procesados
# ============================================================

print(
    "\nGuardando datasets procesados..."
)


np.savez_compressed(
    PROCESSED_DIR / "train.npz",
    X=X_train_processed,
    y=y_train,
)


np.savez_compressed(
    PROCESSED_DIR / "validation.npz",
    X=X_validation_processed,
    y=y_validation,
)


np.savez_compressed(
    PROCESSED_DIR / "test.npz",
    X=X_test_processed,
    y=y_test,
)


# ============================================================
# Guardado del preprocesador
# ============================================================

joblib.dump(
    preprocessor,
    PROCESSED_DIR / "preprocessor.joblib",
)


# ============================================================
# Metadatos
# ============================================================

feature_names = (
    preprocessor.get_feature_names_out()
)


metadata = {
    "dataset": "UNSW-NB15",
    "classification": "binary",
    "target": TARGET,
    "seed": SEED,
    "validation_size": VALIDATION_SIZE,

    "removed_columns": [
        "id",
        "attack_cat",
    ],

    "original_input_features": int(
        X_train.shape[1]
    ),

    "processed_features": int(
        X_train_processed.shape[1]
    ),

    "categorical_columns":
        CATEGORICAL_COLUMNS,

    "numeric_columns":
        numeric_columns,

    "samples": {
        "official_training_set":
            int(len(y_development)),

        "train":
            int(len(y_train)),

        "validation":
            int(len(y_validation)),

        "test":
            int(len(y_test)),
    },

    "class_distribution": {
        "train":
            class_distribution(
                y_train
            ),

        "validation":
            class_distribution(
                y_validation
            ),

        "test":
            class_distribution(
                y_test
            ),
    },

    "processed_feature_names":
        feature_names.tolist(),
}


with open(
    PROCESSED_DIR / "metadata.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# Finalización
# ============================================================

print(
    "\nPreprocesamiento completado "
    "correctamente."
)

print(
    f"\nArchivos generados en:\n"
    f"{PROCESSED_DIR}"
)

print(
    "\n- train.npz"
)

print(
    "- validation.npz"
)

print(
    "- test.npz"
)

print(
    "- preprocessor.joblib"
)

print(
    "- metadata.json"
)