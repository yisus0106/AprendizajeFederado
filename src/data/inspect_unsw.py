from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "raw" / "unsw_nb15"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"


def analizar_dataset(nombre, ruta):
    print("=" * 70)
    print(f"DATASET: {nombre}")
    print("=" * 70)

    df = pd.read_csv(ruta)

    print(f"\nDimensiones: {df.shape}")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    print("\nColumnas:")
    print(df.columns.tolist())

    print("\nPrimeras 5 filas:")
    print(df.head())

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores nulos:")
    nulos = df.isnull().sum()
    print(nulos[nulos > 0])

    print(f"\nTotal de valores nulos: {df.isnull().sum().sum()}")

    duplicados = df.duplicated().sum()
    print(f"\nFilas duplicadas: {duplicados}")

    numericas = df.select_dtypes(include=[np.number])

    infinitos = np.isinf(numericas).sum().sum()

    print(f"Valores infinitos: {infinitos}")

    if "label" in df.columns:
        print("\nDistribución de label:")
        print(df["label"].value_counts())

        print("\nDistribución porcentual:")
        print(
            df["label"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

    if "attack_cat" in df.columns:
        print("\nDistribución de tipos de ataque:")
        print(df["attack_cat"].value_counts())

    print("\nResumen estadístico:")
    print(df.describe().transpose())

    print("\n")


if __name__ == "__main__":
    analizar_dataset("TRAIN", TRAIN_FILE)
    analizar_dataset("TEST", TEST_FILE)