import numpy as np

from src.evaluation.re2_comparison import (
    generate_re2_comparison,
)


def verify_re2_comparison():

    print(
        "=== COMPARACIÓN CENTRALIZADO "
        "VS. FEDERADO ==="
    )

    result = (
        generate_re2_comparison()
    )

    comparison = result[
        "comparison"
    ]

    assert len(comparison) == 6

    assert set(
        comparison["metric"]
    ) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "false_positive_rate",
    }

    numeric_columns = [
        "centralized",
        "federated",
        "difference",
        "difference_percentage_points",
    ]

    for column in numeric_columns:
        assert np.isfinite(
            comparison[column]
        ).all()

    for path_key in [
        "comparison_csv_path",
        "summary_path",
        "metrics_chart_path",
        "confusion_chart_path",
    ]:
        assert result[
            path_key
        ].exists()

    indexed = comparison.set_index(
        "metric"
    )

    assert (
        indexed.loc[
            "recall",
            "difference",
        ]
        > 0
    )

    assert (
        indexed.loc[
            "accuracy",
            "difference",
        ]
        < 0
    )

    assert (
        indexed.loc[
            "false_positive_rate",
            "difference",
        ]
        > 0
    )

    print()

    print(
        comparison[
            [
                "metric_name",
                "centralized",
                "federated",
                "difference_percentage_points",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nFalsos negativos evitados: "
        f"{result['summary']['false_negatives_avoided']}"
    )

    print(
        "Falsos positivos adicionales: "
        f"{result['summary']['additional_false_positives']}"
    )

    print(
        "\nEstado: COMPARACIÓN RE2.3 "
        "GENERADA CORRECTAMENTE"
    )


if __name__ == "__main__":
    verify_re2_comparison()