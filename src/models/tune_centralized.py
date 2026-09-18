from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# Configuración experimental
# ============================================================

SEED = 42

BATCH_SIZE = 256
MAX_EPOCHS = 20
PATIENCE = 3

BASE_THRESHOLD = 0.50
MIN_RECALL = 0.80
F1_TOLERANCE = 0.001

# ============================================================
# Configuraciones a comparar
# ============================================================

CONFIGURATIONS = [
    {
        "id": "C0",
        "hidden_layers": [64, 32],
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
    {
        "id": "C1",
        "hidden_layers": [32, 16],
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
    {
        "id": "C2",
        "hidden_layers": [64, 32],
        "dropout": 0.3,
        "learning_rate": 0.001,
    },
    {
        "id": "C3",
        "hidden_layers": [64, 32],
        "dropout": 0.2,
        "learning_rate": 0.0005,
    },
    {
        "id": "C4",
        "hidden_layers": [128, 64],
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
]


# ============================================================
# Rutas
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "unsw_nb15"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "re2"
    / "centralized"
    / "tuning"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_FILE = DATA_DIR / "train.npz"
VALIDATION_FILE = DATA_DIR / "validation.npz"


# ============================================================
# Reproducibilidad
# ============================================================

def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


# ============================================================
# Construcción del modelo
# ============================================================

def create_model(
    input_dim,
    hidden_layers,
    dropout,
    learning_rate,
):
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Input(
            shape=(input_dim,)
        )
    )

    for index, units in enumerate(
        hidden_layers
    ):
        model.add(
            tf.keras.layers.Dense(
                units,
                activation="relu"
            )
        )

        # Dropout después de la primera
        # capa oculta.
        if index == 0:
            model.add(
                tf.keras.layers.Dropout(
                    dropout
                )
            )

    model.add(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    return model


# ============================================================
# Cálculo de métricas
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

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

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "specificity": float(
            specificity
        ),

        "false_positive_rate": float(
            false_positive_rate
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# Carga de TRAIN y VALIDATION
# ============================================================

print("=" * 70)
print("AJUSTE DE HIPERPARÁMETROS - UNSW-NB15")
print("=" * 70)


train_data = np.load(
    TRAIN_FILE
)

validation_data = np.load(
    VALIDATION_FILE
)


X_train = train_data["X"]
y_train = train_data["y"]

X_validation = validation_data["X"]
y_validation = validation_data["y"]


print(
    f"\nX_train:      {X_train.shape}"
)

print(
    f"X_validation: {X_validation.shape}"
)

print(
    "\nTEST no será utilizado "
    "durante el ajuste."
)


input_dim = X_train.shape[1]


# ============================================================
# Evaluación de configuraciones
# ============================================================

configuration_results = []


for configuration in CONFIGURATIONS:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Configuración "
        f"{configuration['id']}"
    )

    print(
        "=" * 70
    )

    print(
        f"Capas: "
        f"{configuration['hidden_layers']}"
    )

    print(
        f"Dropout: "
        f"{configuration['dropout']}"
    )

    print(
        f"Learning rate: "
        f"{configuration['learning_rate']}"
    )


    tf.keras.backend.clear_session()
    reset_seed()


    model = create_model(
        input_dim=input_dim,

        hidden_layers=(
            configuration[
                "hidden_layers"
            ]
        ),

        dropout=(
            configuration[
                "dropout"
            ]
        ),

        learning_rate=(
            configuration[
                "learning_rate"
            ]
        ),
    )


    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        )
    )


    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_validation,
            y_validation
        ),

        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,

        callbacks=[
            early_stopping
        ],

        verbose=0,
    )


    probabilities = model.predict(
        X_validation,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).ravel()


    metrics = calculate_metrics(
        y_validation,
        probabilities,
        BASE_THRESHOLD,
    )


    best_epoch = (
        int(
            np.argmin(
                history.history[
                    "val_loss"
                ]
            )
        )
        + 1
    )


    result = {
        "configuration":
            configuration["id"],

        "hidden_layers":
            str(
                configuration[
                    "hidden_layers"
                ]
            ),

        "dropout":
            configuration[
                "dropout"
            ],

        "learning_rate":
            configuration[
                "learning_rate"
            ],

        "batch_size":
            BATCH_SIZE,

        "epochs_executed":
            len(
                history.history[
                    "loss"
                ]
            ),

        "best_epoch":
            best_epoch,

        "best_val_loss":
            float(
                min(
                    history.history[
                        "val_loss"
                    ]
                )
            ),

        **metrics,
    }


    configuration_results.append(
        result
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
        f"FPR:       "
        f"{metrics['false_positive_rate']:.4f}"
    )

    print(
        f"AUC:       "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Mejor epoch: "
        f"{best_epoch}"
    )


# ============================================================
# Guardar búsqueda de hiperparámetros
# ============================================================

results_df = pd.DataFrame(
    configuration_results
)


results_df.to_csv(
    RESULTS_DIR
    / "hyperparameter_search.csv",

    index=False,
)


# ============================================================
# Selección de mejor configuración
# ============================================================

eligible_df = results_df[
    results_df["recall"] >= MIN_RECALL
].copy()


if eligible_df.empty:
    raise RuntimeError(
        "Ninguna configuración cumple "
        "el recall mínimo."
    )


eligible_df = eligible_df.sort_values(
    by=[
        "f1_score",
        "false_positive_rate",
    ],

    ascending=[
        False,
        True,
    ],
)


best_row = eligible_df.iloc[0]

best_id = best_row[
    "configuration"
]


best_configuration = next(
    configuration
    for configuration in CONFIGURATIONS
    if configuration["id"] == best_id
)


print(
    "\n"
    + "=" * 70
)

print(
    "MEJOR CONFIGURACIÓN"
)

print(
    "=" * 70
)

print(
    best_configuration
)


# ============================================================
# Reentrenar mejor configuración
# para ajuste del threshold
# ============================================================

tf.keras.backend.clear_session()
reset_seed()


best_model = create_model(
    input_dim=input_dim,

    hidden_layers=(
        best_configuration[
            "hidden_layers"
        ]
    ),

    dropout=(
        best_configuration[
            "dropout"
        ]
    ),

    learning_rate=(
        best_configuration[
            "learning_rate"
        ]
    ),
)


early_stopping = (
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )
)


best_model.fit(
    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,

    callbacks=[
        early_stopping
    ],

    verbose=0,
)


validation_probabilities = (
    best_model.predict(
        X_validation,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).ravel()
)


# ============================================================
# Búsqueda de threshold
# ============================================================

threshold_results = []


thresholds = np.arange(
    0.10,
    0.91,
    0.05,
)


for threshold in thresholds:

    threshold = round(
        float(threshold),
        2
    )

    metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        threshold,
    )

    threshold_results.append(
        metrics
    )


threshold_df = pd.DataFrame(
    threshold_results
)


threshold_df.to_csv(
    RESULTS_DIR
    / "threshold_search.csv",

    index=False,
)


eligible_thresholds = threshold_df[
    threshold_df["recall"]
    >= MIN_RECALL
].copy()


eligible_thresholds = threshold_df[
    threshold_df["recall"] >= MIN_RECALL
].copy()


# ============================================================
# Selección de thresholds con F1 prácticamente equivalente
# ============================================================

max_f1 = eligible_thresholds[
    "f1_score"
].max()


near_best_thresholds = eligible_thresholds[
    eligible_thresholds["f1_score"]
    >= (max_f1 - F1_TOLERANCE)
].copy()


# Entre los thresholds con F1 prácticamente equivalente,
# se selecciona el que tenga menor tasa de falsos positivos.
near_best_thresholds = (
    near_best_thresholds.sort_values(
        by=[
            "false_positive_rate",
            "f1_score",
        ],
        ascending=[
            True,
            False,
        ],
    )
)


best_threshold_row = (
    near_best_thresholds.iloc[0]
)


best_threshold = float(
    best_threshold_row[
        "threshold"
    ]
)


# ============================================================
# Configuración seleccionada
# ============================================================

final_configuration = {
    "dataset": "UNSW-NB15",

    "seed": SEED,

    "selection_dataset":
        "validation",

    "model_selection_metric":
        "f1_score",

    "threshold_selection": {
        "minimum_recall":
            MIN_RECALL,

        "primary_metric":
            "f1_score",

        "f1_tolerance":
            F1_TOLERANCE,

        "secondary_metric":
            "false_positive_rate",

        "secondary_objective":
            "minimize",
    },

    "hidden_layers":
        best_configuration[
            "hidden_layers"
        ],

    "dropout":
        best_configuration[
            "dropout"
        ],

    "learning_rate":
        best_configuration[
            "learning_rate"
        ],

    "batch_size":
        BATCH_SIZE,

    "max_epochs":
        MAX_EPOCHS,

    "patience":
        PATIENCE,

    "classification_threshold":
        best_threshold,

    "validation_metrics": {
        "accuracy":
            float(
                best_threshold_row[
                    "accuracy"
                ]
            ),

        "precision":
            float(
                best_threshold_row[
                    "precision"
                ]
            ),

        "recall":
            float(
                best_threshold_row[
                    "recall"
                ]
            ),

        "f1_score":
            float(
                best_threshold_row[
                    "f1_score"
                ]
            ),

        "roc_auc":
            float(
                best_threshold_row[
                    "roc_auc"
                ]
            ),

        "specificity":
            float(
                best_threshold_row[
                    "specificity"
                ]
            ),

        "false_positive_rate":
            float(
                best_threshold_row[
                    "false_positive_rate"
                ]
            ),
    },
}


with open(
    RESULTS_DIR
    / "best_config.json",

    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_configuration,
        file,
        indent=4,
    )


# ============================================================
# Resultado final
# ============================================================

print(
    f"F1 máximo observado: "
    f"{max_f1:.6f}"
)

print(
    f"Tolerancia de F1: "
    f"{F1_TOLERANCE:.4f}"
)

print(
    "\nThresholds considerados "
    "prácticamente equivalentes:"
)

print(
    near_best_thresholds[
        [
            "threshold",
            "f1_score",
            "recall",
            "false_positive_rate",
        ]
    ].to_string(index=False)
)

print(
    "\n"
    + "=" * 70
)

print(
    "THRESHOLD SELECCIONADO"
)

print(
    "=" * 70
)

print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy:  "
    f"{best_threshold_row['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{best_threshold_row['precision']:.4f}"
)

print(
    f"Recall:    "
    f"{best_threshold_row['recall']:.4f}"
)

print(
    f"F1-score:  "
    f"{best_threshold_row['f1_score']:.4f}"
)

print(
    f"FPR:       "
    f"{best_threshold_row['false_positive_rate']:.4f}"
)


print(
    "\nArchivos generados:"
)

print(
    RESULTS_DIR
    / "hyperparameter_search.csv"
)

print(
    RESULTS_DIR
    / "threshold_search.csv"
)

print(
    RESULTS_DIR
    / "best_config.json"
)
