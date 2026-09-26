from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.federated.unsw_client_data import (
    load_common_test_set,
)


# ============================================================
# Configuración
# ============================================================

CLASSIFICATION_THRESHOLD = 0.45
BATCH_SIZE = 256

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "models"
    / "federated"
    / "federated_ids.keras"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "re2"
    / "federated"
)


# ============================================================
# Evaluación
# ============================================================

def evaluate_federated_ids_model(
    threshold=CLASSIFICATION_THRESHOLD,
):
    """
    Evalúa el modelo global federado sobre el conjunto
    TEST común de UNSW-NB15.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "El umbral debe pertenecer "
            "al intervalo [0, 1]."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo global: "
            f"{MODEL_PATH}"
        )

    print(
        "Cargando modelo global federado..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print(
        "Cargando conjunto TEST..."
    )

    X_test, y_test = (
        load_common_test_set()
    )

    if (
        model.input_shape[-1]
        != X_test.shape[1]
    ):
        raise ValueError(
            "La cantidad de características de TEST "
            "no coincide con la entrada del modelo."
        )

    print(
        f"Ejemplos de TEST: {len(y_test)}"
    )

    print(
        f"Umbral de clasificación: {threshold}"
    )

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    ).ravel()

    predictions = (
        probabilities >= threshold
    ).astype(
        np.int32
    )

    # ========================================================
    # Métricas
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    results = {
        "dataset": "UNSW-NB15",
        "model": "Federated MLP",
        "aggregation": "FedAvg",
        "num_clients": 2,
        "num_rounds": 5,
        "test_samples": int(
            len(y_test)
        ),
        "classification_threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1_score": float(
            f1
        ),
        "roc_auc": float(
            roc_auc
        ),
        "specificity": float(
            specificity
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "confusion_matrix": (
            cm.tolist()
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # ========================================================
    # Guardar resultados
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        RESULTS_DIR
        / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    predictions_path = (
        RESULTS_DIR
        / "predictions.csv"
    )

    predictions_df = pd.DataFrame({
        "real": y_test,
        "probability": probabilities,
        "prediction": predictions,
    })

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    return {
        "metrics": results,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
    }