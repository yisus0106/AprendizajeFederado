import numpy as np

from src.federated.ids_evaluation import (
    evaluate_federated_ids_model,
)


def verify_ids_federated_evaluation():

    print(
        "=== EVALUACIÓN DEL MODELO "
        "FEDERADO IDS ==="
    )

    result = (
        evaluate_federated_ids_model()
    )

    metrics = result["metrics"]
    predictions = result["predictions"]
    probabilities = result["probabilities"]

    evaluated_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    for metric_name in evaluated_metrics:

        metric_value = metrics[
            metric_name
        ]

        assert np.isfinite(
            metric_value
        )

        assert (
            0.0
            <= metric_value
            <= 1.0
        )

    assert len(predictions) == (
        metrics["test_samples"]
    )

    assert len(probabilities) == (
        metrics["test_samples"]
    )

    predicted_classes = set(
        np.unique(
            predictions
        ).tolist()
    )

    assert predicted_classes == {
        0,
        1,
    }

    assert result[
        "metrics_path"
    ].exists()

    assert result[
        "predictions_path"
    ].exists()

    print(
        "\nRESULTADOS SOBRE TEST"
    )

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score:  "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        "\nMatriz de confusión:"
    )

    print(
        np.asarray(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    print(
        "\nClases producidas: "
        f"{sorted(predicted_classes)}"
    )

    print(
        "Estado: MODELO GLOBAL "
        "EVALUADO CORRECTAMENTE"
    )


if __name__ == "__main__":
    verify_ids_federated_evaluation()