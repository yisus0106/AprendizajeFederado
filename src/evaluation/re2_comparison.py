from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# Rutas
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CENTRALIZED_METRICS_PATH = (
    BASE_DIR
    / "results"
    / "re2"
    / "comparison"
    / "centralized_metrics_threshold_045.json"
)

FEDERATED_METRICS_PATH = (
    BASE_DIR
    / "results"
    / "re2"
    / "federated"
    / "metrics.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "re2"
    / "comparison"
)


# ============================================================
# Configuración de métricas
# ============================================================

METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "roc_auc": "ROC-AUC",
}


def load_json(file_path):
    """
    Carga y devuelve un archivo JSON.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def extract_confusion_values(metrics):
    """
    Extrae TN, FP, FN y TP de una matriz
    de confusión binaria.
    """

    confusion = np.asarray(
        metrics["confusion_matrix"],
        dtype=np.int64,
    )

    if confusion.shape != (2, 2):
        raise ValueError(
            "La matriz de confusión debe "
            "tener dimensiones 2x2."
        )

    tn, fp, fn, tp = confusion.ravel()

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

    return {
        "matrix": confusion,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(
            specificity
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
    }


def create_metrics_chart(
    comparison_df,
    output_path,
):
    """
    Genera un gráfico de barras con las métricas
    centralizadas y federadas.
    """

    sns.set_theme(
        style="whitegrid"
    )

    labels = comparison_df[
        "metric_name"
    ].tolist()

    centralized = comparison_df[
        "centralized"
    ].to_numpy()

    federated = comparison_df[
        "federated"
    ].to_numpy()

    positions = np.arange(
        len(labels)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    central_bars = axis.bar(
        positions - width / 2,
        centralized,
        width,
        label="Centralizado",
    )

    federated_bars = axis.bar(
        positions + width / 2,
        federated,
        width,
        label="Federado",
    )

    axis.set_title(
        "Comparación del desempeño "
        "centralizado y federado"
    )

    axis.set_ylabel(
        "Valor de la métrica"
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        labels
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.legend()

    axis.bar_label(
        central_bars,
        fmt="%.4f",
        padding=3,
    )

    axis.bar_label(
        federated_bars,
        fmt="%.4f",
        padding=3,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_confusion_chart(
    centralized_confusion,
    federated_confusion,
    output_path,
):
    """
    Genera las matrices de confusión de ambos
    enfoques utilizando la misma escala.
    """

    sns.set_theme(
        style="white"
    )

    maximum_value = max(
        centralized_confusion.max(),
        federated_confusion.max(),
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    matrices = [
        centralized_confusion,
        federated_confusion,
    ]

    titles = [
        "Modelo centralizado",
        "Modelo federado",
    ]

    for axis, matrix, title in zip(
        axes,
        matrices,
        titles,
    ):

        sns.heatmap(
            matrix,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            vmin=0,
            vmax=maximum_value,
            xticklabels=[
                "Benigno",
                "Malicioso",
            ],
            yticklabels=[
                "Benigno",
                "Malicioso",
            ],
            ax=axis,
        )

        axis.set_title(
            title
        )

        axis.set_xlabel(
            "Clase predicha"
        )

        axis.set_ylabel(
            "Clase real"
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def generate_re2_comparison():
    """
    Compara el modelo centralizado y el federado
    sobre el mismo conjunto TEST.
    """

    centralized = load_json(
        CENTRALIZED_METRICS_PATH
    )

    federated = load_json(
        FEDERATED_METRICS_PATH
    )

    central_threshold = float(
        centralized[
            "classification_threshold"
        ]
    )

    federated_threshold = float(
        federated[
            "classification_threshold"
        ]
    )

    if not np.isclose(
        central_threshold,
        federated_threshold,
    ):
        raise ValueError(
            "Los modelos no utilizan el mismo "
            "umbral de clasificación."
        )

    central_confusion = (
        extract_confusion_values(
            centralized
        )
    )

    federated_confusion = (
        extract_confusion_values(
            federated
        )
    )

    comparison_rows = []

    for metric_key, metric_name in (
        METRICS.items()
    ):

        central_value = float(
            centralized[metric_key]
        )

        federated_value = float(
            federated[metric_key]
        )

        difference = (
            federated_value
            - central_value
        )

        comparison_rows.append({
            "metric": metric_key,
            "metric_name": metric_name,
            "centralized": central_value,
            "federated": federated_value,
            "difference": difference,
            "difference_percentage_points":
                difference * 100.0,
        })

    # Agregar FPR a la comparación.
    central_fpr = central_confusion[
        "false_positive_rate"
    ]

    federated_fpr = federated_confusion[
        "false_positive_rate"
    ]

    comparison_rows.append({
        "metric": "false_positive_rate",
        "metric_name": "FPR",
        "centralized": central_fpr,
        "federated": federated_fpr,
        "difference": (
            federated_fpr
            - central_fpr
        ),
        "difference_percentage_points": (
            federated_fpr
            - central_fpr
        ) * 100.0,
    })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    false_negatives_avoided = (
        central_confusion["fn"]
        - federated_confusion["fn"]
    )

    additional_false_positives = (
        federated_confusion["fp"]
        - central_confusion["fp"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_csv_path = (
        OUTPUT_DIR
        / "metrics_comparison.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "comparison_summary.json"
    )

    metrics_chart_path = (
        OUTPUT_DIR
        / "metrics_comparison.png"
    )

    confusion_chart_path = (
        OUTPUT_DIR
        / "confusion_matrices.png"
    )

    comparison_df.to_csv(
        comparison_csv_path,
        index=False,
    )
    recall_difference_pp = (
        float(federated["recall"])
        - float(centralized["recall"])
    ) * 100.0
    
    summary = {
        "dataset": "UNSW-NB15",
        "test_samples": int(
            federated["test_samples"]
        ),
        "classification_threshold":
            federated_threshold,
        "federated_configuration": {
            "num_clients": int(
                federated["num_clients"]
            ),
            "num_rounds": int(
                federated["num_rounds"]
            ),
            "aggregation": federated[
                "aggregation"
            ],
        },
        "centralized_confusion_matrix":
            central_confusion[
                "matrix"
            ].tolist(),
        "federated_confusion_matrix":
            federated_confusion[
                "matrix"
            ].tolist(),
        "false_negatives_avoided":
            int(false_negatives_avoided),
        "additional_false_positives":
            int(additional_false_positives),
        "findings": [
            (
                "El modelo federado incrementó "
                "el recall en "
                f"{recall_difference_pp:.2f} "
                "puntos porcentuales."
            ),
            (
                "El modelo federado evitó "
                f"{false_negatives_avoided} "
                "falsos negativos respecto "
                "al modelo centralizado."
            ),
            (
                "El modelo federado produjo "
                f"{additional_false_positives} "
                "falsos positivos adicionales."
            ),
        ],
        "limitations": [
            (
                "El experimento utiliza dos clientes "
                "simulados dentro de un entorno "
                "académico controlado."
            ),
            (
                "Las particiones son estratificadas "
                "y no representan todavía un "
                "escenario non-IID."
            ),
        ],
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    create_metrics_chart(
        comparison_df[
            comparison_df["metric"]
            != "false_positive_rate"
        ],
        metrics_chart_path,
    )

    create_confusion_chart(
        central_confusion["matrix"],
        federated_confusion["matrix"],
        confusion_chart_path,
    )

    return {
        "comparison": comparison_df,
        "summary": summary,
        "comparison_csv_path":
            comparison_csv_path,
        "summary_path":
            summary_path,
        "metrics_chart_path":
            metrics_chart_path,
        "confusion_chart_path":
            confusion_chart_path,
    }