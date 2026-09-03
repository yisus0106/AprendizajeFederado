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
    classification_report,
)


# ============================================================
# Reproducibilidad
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Rutas
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "unsw_nb15"

RESULTS_DIR = BASE_DIR / "src" / "results" / "centralized"
MODEL_DIR = BASE_DIR / "src" / "models" / "centralized"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = DATA_DIR / "train.npz"
TEST_FILE = DATA_DIR / "test.npz"


# ============================================================
# Carga de datos
# ============================================================

print("=" * 70)
print("MODELO CENTRALIZADO - UNSW-NB15")
print("=" * 70)

print("\nCargando datos procesados...")

train_data = np.load(TRAIN_FILE)
test_data = np.load(TEST_FILE)

X_train = train_data["X"]
y_train = train_data["y"]

X_test = test_data["X"]
y_test = test_data["y"]


print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")


# ============================================================
# Modelo
# ============================================================

input_dim = X_train.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    ),
])


# ============================================================
# Compilación
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
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
    ]
)


print("\nArquitectura del modelo:")
model.summary()


# ============================================================
# Entrenamiento
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


print("\nIniciando entrenamiento...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=20,
    batch_size=256,
    shuffle=True,
    callbacks=[
        early_stopping
    ],
    verbose=1
)


# ============================================================
# Evaluación
# ============================================================

print("\nEvaluando sobre TEST...")

probabilities = model.predict(
    X_test,
    batch_size=256,
    verbose=1
).ravel()


predictions = (
    probabilities >= 0.5
).astype(np.int32)


# ============================================================
# Métricas
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)


print("\n" + "=" * 70)
print("RESULTADOS")
print("=" * 70)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")

print("\nMatriz de confusión:")
print(cm)

print("\nReporte de clasificación:")
print(
    classification_report(
        y_test,
        predictions,
        digits=4
    )
)


# ============================================================
# Guardar resultados
# ============================================================

results = {
    "dataset": "UNSW-NB15",
    "model": "MLP",
    "seed": SEED,
    "input_features": int(input_dim),
    "batch_size": 256,
    "max_epochs": 20,
    "classification_threshold": 0.5,

    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(auc),

    "confusion_matrix": cm.tolist(),
}


with open(
    RESULTS_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# Guardar historial
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    RESULTS_DIR / "training_history.csv",
    index=False
)


# ============================================================
# Guardar predicciones
# ============================================================

predictions_df = pd.DataFrame({
    "real": y_test,
    "probability": probabilities,
    "prediction": predictions
})

predictions_df.to_csv(
    RESULTS_DIR / "predictions.csv",
    index=False
)


# ============================================================
# Guardar modelo
# ============================================================

model.save(
    MODEL_DIR / "unsw_nb15_baseline.keras"
)


print("\nArchivos generados:")

print(
    RESULTS_DIR / "metrics.json"
)

print(
    RESULTS_DIR / "training_history.csv"
)

print(
    RESULTS_DIR / "predictions.csv"
)

print(
    MODEL_DIR / "unsw_nb15_baseline.keras"
)

print("\nEntrenamiento centralizado finalizado.")