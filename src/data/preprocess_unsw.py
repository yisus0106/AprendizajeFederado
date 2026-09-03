from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Rutas
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "unsw_nb15"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "unsw_nb15"

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Carga
# ============================================================

print("Cargando UNSW-NB15...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train original: {train_df.shape}")
print(f"Test original:  {test_df.shape}")


# ============================================================
# Separación de características y objetivo
# ============================================================

TARGET = "label"

DROP_COLUMNS = [
    "id",
    "attack_cat",
    TARGET,
]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df[TARGET].to_numpy(dtype=np.int32)

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df[TARGET].to_numpy(dtype=np.int32)


# ============================================================
# Identificación de columnas
# ============================================================

categorical_columns = [
    "proto",
    "service",
    "state",
]

numeric_columns = [
    column
    for column in X_train.columns
    if column not in categorical_columns
]

print("\nCaracterísticas utilizadas:")
print(f"Numéricas:    {len(numeric_columns)}")
print(f"Categóricas: {len(categorical_columns)}")
print(f"Total:        {X_train.shape[1]}")

print("\nColumnas categóricas:")
print(categorical_columns)


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
            categorical_columns,
        ),
    ]
)


# ============================================================
# IMPORTANTE:
# El preprocesador se ajusta únicamente con TRAIN
# ============================================================

print("\nAjustando preprocesador con TRAIN...")

X_train_processed = preprocessor.fit_transform(X_train)

print("Transformando TEST...")

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# Conversión a float32 para TensorFlow
# ============================================================

X_train_processed = X_train_processed.astype(np.float32)
X_test_processed = X_test_processed.astype(np.float32)


# ============================================================
# Verificaciones
# ============================================================

print("\nResultado del preprocesamiento:")
print(f"X_train: {X_train_processed.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test:  {X_test_processed.shape}")
print(f"y_test:  {y_test.shape}")

print(f"\nTipo X_train: {X_train_processed.dtype}")
print(f"Tipo X_test:  {X_test_processed.dtype}")

print(
    "\nNaN en train:",
    np.isnan(X_train_processed).sum(),
)

print(
    "NaN en test:",
    np.isnan(X_test_processed).sum(),
)

print(
    "Inf en train:",
    np.isinf(X_train_processed).sum(),
)

print(
    "Inf en test:",
    np.isinf(X_test_processed).sum(),
)


# ============================================================
# Guardado
# ============================================================

print("\nGuardando datasets procesados...")

np.savez_compressed(
    PROCESSED_DIR / "train.npz",
    X=X_train_processed,
    y=y_train,
)

np.savez_compressed(
    PROCESSED_DIR / "test.npz",
    X=X_test_processed,
    y=y_test,
)

joblib.dump(
    preprocessor,
    PROCESSED_DIR / "preprocessor.joblib",
)


# ============================================================
# Metadatos
# ============================================================

feature_names = preprocessor.get_feature_names_out()

metadata = {
    "dataset": "UNSW-NB15",
    "classification": "binary",
    "target": TARGET,
    "removed_columns": [
        "id",
        "attack_cat",
    ],
    "original_features": int(X_train.shape[1]),
    "processed_features": int(X_train_processed.shape[1]),
    "train_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "categorical_columns": categorical_columns,
    "numeric_columns": numeric_columns,
    "processed_feature_names": feature_names.tolist(),
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


print("\nPreprocesamiento completado correctamente.")
print(f"Archivos guardados en:\n{PROCESSED_DIR}")