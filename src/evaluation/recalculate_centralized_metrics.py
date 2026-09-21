from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


THRESHOLD = 0.45

BASE_DIR = Path(__file__).resolve().parents[2]

ORIGINAL_METRICS_PATH = (
    BASE_DIR
    / "results"
    / "re2"
    / "centralized"
    / "metrics.json"
)

ORIGINAL_PREDICTIONS_PATH = (
    BASE_DIR
    / "artifacts"
    / "predictions"
    / "centralized_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "re2"
    / "comparison"
)

OUTPUT_METRICS_PATH = (
    OUTPUT_DIR
    / "centralized_metrics_threshold_045.json"
)

OUTPUT_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "centralized_predictions_threshold_045.csv"
)


def recalculate_centralized_metrics():

    if not ORIGINAL_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {ORIGINAL_METRICS_PATH}"
        )

    # if not ORIGINAL_PREDICTIONS_PATH.exists():
    #     raise FileNotFoundError(
    #         f"No existe: {ORIGINAL_PREDICTIONS_PATH}"
    #     )

    with open(
        ORIGINAL_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        original_metrics = json.load(
            file
        )

    prediction_data = pd.read_csv(
        ORIGINAL_PREDICTIONS_PATH
    )

    required_columns = {
        "real",
        "probability",
    }

    if not required_columns.issubset(
        prediction_data.columns
    ):
        raise ValueError(
            "predictions.csv debe contener "
            "las columnas real y probability."
        )

    y_true = prediction_data[
        "real"
    ].to_numpy(
        dtype=np.int32
    )

    probabilities = prediction_data[
        "probability"
    ].to_numpy(
        dtype=np.float32
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(
        np.int32
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    cm = confusion_matrix(
        y_true,
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

    updated_metrics = dict(
        original_metrics
    )

    updated_metrics.update({
        "classification_threshold":
            THRESHOLD,
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
        "confusion_matrix":
            cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            updated_metrics,
            file,
            indent=4,
        )

    updated_predictions = (
        prediction_data.copy()
    )

    updated_predictions[
        "prediction"
    ] = predictions

    updated_predictions.to_csv(
        OUTPUT_PREDICTIONS_PATH,
        index=False,
    )

    print(
        "=== MÉTRICAS CENTRALIZADAS "
        "RECALCULADAS ==="
    )

    print(
        f"Umbral:   {THRESHOLD}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        "\nMatriz de confusión:"
    )

    print(
        cm
    )

    print(
        "\nArchivo generado:"
    )

    print(
        OUTPUT_METRICS_PATH
    )


if __name__ == "__main__":
    recalculate_centralized_metrics()