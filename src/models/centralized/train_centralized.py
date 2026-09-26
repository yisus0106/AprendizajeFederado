"""Entrena y evalúa el baseline centralizado con la configuración congelada por CV.

Ubicar en src/models/ y ejecutar: python src/models/train_centralized.py
Lee results/re2/centralized/tuning/best_config.json. No ajusta nada usando TEST.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# ============================================================
# Rutas
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "unsw_nb15"
RESULTS_DIR = BASE_DIR / "results" / "re2" / "centralized" / "train"
PREDICTIONS_DIR = BASE_DIR / "artifacts" / "predictions"
MODEL_DIR = BASE_DIR / "artifacts" / "models" / "centralized"
BEST_CONFIG_FILE = BASE_DIR / "results" / "re2" / "centralized" / "tuning" / "best_config.json"
TRAIN_FILE = DATA_DIR / "train.npz"
VALIDATION_FILE = DATA_DIR / "validation.npz"
TEST_FILE = DATA_DIR / "test.npz"


def load_split(path):
    with np.load(path) as data:
        X, y = data["X"], data["y"].ravel()
    if X.ndim != 2 or len(X) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError(f"Partición inválida (X 2D, y binaria 0/1): {path}")
    if not np.isfinite(X).all():
        raise ValueError(f"X contiene NaN o infinitos: {path}")
    return X, y.astype(np.int32)


def load_configuration():
    config_bytes = BEST_CONFIG_FILE.read_bytes()
    config = json.loads(config_bytes)
    required = ("dataset", "seed", "selection_dataset", "class_weight", "configuration_id", "hidden_layers", "dropout",
                "learning_rate", "batch_size", "num_params", "classification_threshold", "final_training_epochs")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Faltan claves en best_config.json: {missing}")
    if config["dataset"] != "UNSW-NB15" or config["selection_dataset"] != "train+validation":
        raise ValueError("Se espera la configuración de UNSW-NB15 seleccionada sobre train+validation.")
    if config["class_weight"] is not None:
        raise ValueError("Este baseline corresponde al tuning sin class_weight; revisa el JSON antes de entrenar.")
    if not config["hidden_layers"] or any(not isinstance(units, int) or units <= 0 for units in config["hidden_layers"]):
        raise ValueError("hidden_layers debe contener tamaños enteros positivos.")
    if not 0 <= config["dropout"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("dropout o learning_rate inválido.")
    if config["batch_size"] <= 0 or config["final_training_epochs"] <= 0:
        raise ValueError("batch_size y final_training_epochs deben ser positivos.")
    if not 0 <= config["classification_threshold"] <= 1:
        raise ValueError("classification_threshold debe estar entre 0 y 1.")
    return config, hashlib.sha256(config_bytes).hexdigest()


def create_model(input_dim, config):
    model = tf.keras.Sequential([tf.keras.Input(shape=(input_dim,))])
    for index, units in enumerate(config["hidden_layers"]):
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        if index == 0:
            model.add(tf.keras.layers.Dropout(config["dropout"]))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    config, config_sha256 = load_configuration()
    X_train, y_train = load_split(TRAIN_FILE)
    X_validation, y_validation = load_split(VALIDATION_FILE)
    X_test, y_test = load_split(TEST_FILE)
    if X_train.shape[1] != X_validation.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("TRAIN, VALIDATION y TEST deben tener el mismo número de características.")
    if len(np.unique(y_test)) != 2:
        raise ValueError("TEST debe contener ambas clases para calcular ROC-AUC y FPR.")

    X_dev = np.vstack((X_train, X_validation))
    y_dev = np.concatenate((y_train, y_validation))
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    model = create_model(X_dev.shape[1], config)
    if model.count_params() != config["num_params"]:
        raise ValueError(f"Parámetros incompatibles: modelo={model.count_params()}, JSON={config['num_params']}.")

    print(f"Modelo: {config['configuration_id']} | parámetros: {model.count_params()} | umbral: {config['classification_threshold']}")
    print(f"Entrenamiento: train+validation ({len(y_dev)} filas), {config['final_training_epochs']} épocas, sin class_weight.")
    # No hay early stopping: las épocas ya se eligieron en CV y todo el desarrollo se usa para entrenar.
    history = model.fit(X_dev, y_dev, epochs=config["final_training_epochs"], batch_size=config["batch_size"],
                        shuffle=True, verbose=0)

    # TEST solo se usa para medir el modelo ya entrenado; aquí no se cambia el umbral.
    probabilities = model.predict(X_test, batch_size=config["batch_size"], verbose=0).ravel()
    predictions = (probabilities >= config["classification_threshold"]).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    cm = [[int(tn), int(fp)], [int(fn), int(tp)]]
    results = {
        "dataset": config["dataset"], "model": "MLP", "configuration_id": config["configuration_id"],
        "selection_dataset": config["selection_dataset"], "evaluation_dataset": "test",
        "best_config_file": str(BEST_CONFIG_FILE.relative_to(BASE_DIR)), "best_config_sha256": config_sha256,
        "seed": config["seed"], "input_features": int(X_dev.shape[1]),
        "samples": {"train": len(y_train), "validation": len(y_validation), "train_plus_validation": len(y_dev), "test": len(y_test)},
        "architecture": {"hidden_layers": config["hidden_layers"], "activation": "relu", "output_activation": "sigmoid",
                         "dropout": config["dropout"], "num_params": model.count_params()},
        "optimizer": "Adam", "learning_rate": config["learning_rate"], "batch_size": config["batch_size"],
        "epochs_trained": len(history.history["loss"]), "final_training_epochs": config["final_training_epochs"],
        "cv_best_epochs": config.get("cv_best_epochs"), "early_stopping_final": False, "class_weight": None,
        "classification_threshold": config["classification_threshold"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "false_positive_rate": float(fp / (tn + fp)), "specificity": float(tn / (tn + fp)),
        "confusion_matrix": cm,
    }

    for directory in (RESULTS_DIR, PREDICTIONS_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.json"
    history_path = RESULTS_DIR / "training_history.csv"
    predictions_path = PREDICTIONS_DIR / "centralized_predictions.csv"
    model_path = MODEL_DIR / "unsw_nb15_baseline.keras"
    model.save(model_path)
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    pd.DataFrame({"real": y_test, "probability": probabilities, "prediction": predictions}).to_csv(predictions_path, index=False)

    print("\nRESULTADOS DEL BASELINE CENTRALIZADO (TEST)")
    for metric in ("accuracy", "precision", "recall", "f1_score", "false_positive_rate", "roc_auc"):
        print(f"{metric}: {results[metric]:.4f}")
    print("Matriz de confusión (0=benigno, 1=ataque):")
    print(np.array(cm))
    print(classification_report(y_test, predictions, digits=4, zero_division=0))
    print("Archivos generados:")
    for path in (metrics_path, history_path, predictions_path, model_path):
        print(path.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
