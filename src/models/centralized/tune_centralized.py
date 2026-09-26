"""Grafica CV existente o selecciona arquitectura; evalúa test si EVALUATE_TEST=True.

Ubicar en src/models/ y ejecutar: python src/models/tune_centralized_final.py
Se esperan data/processed/unsw_nb15/{train,validation,test}.npz con claves X e y.
Con PLOT_ONLY=True se generan las figuras sin reentrenar ni modificar best_config.json.
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 20
PATIENCE = 3
N_SPLITS = 3
MIN_RECALL_ALLOWED = 0.80
MAX_FPR_ALLOWED = 0.03
FALLBACK_MAX_FPR = 0.05
F1_TOLERANCE_MARGIN = 0.005
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)
PLOT_ONLY = False  # Usa la matriz CV guardada; False para volver a ejecutar todo el tuning.
EVALUATE_TEST = False  # Poner True una sola vez, después de revisar best_config.json.
CONFIGURATIONS = [
    {"id": "C1_Nano", "hidden_layers": [16, 8], "dropout": 0.1, "learning_rate": 0.001},
    {"id": "C2_Micro_Base", "hidden_layers": [32, 16], "dropout": 0.2, "learning_rate": 0.001},
    {"id": "C3_Micro_Robust_FL", "hidden_layers": [32, 16], "dropout": 0.35, "learning_rate": 0.001},
    {"id": "C4_Micro_Slow_FL", "hidden_layers": [32, 16], "dropout": 0.2, "learning_rate": 0.0005},
    {"id": "C5_Pyme_Max", "hidden_layers": [64, 32], "dropout": 0.2, "learning_rate": 0.001},
]
METRICS = ("accuracy", "precision", "recall", "f1_score", "roc_auc", "specificity", "false_positive_rate")

# ============================================================
# Rutas (archivo ubicado en src/models/)
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "unsw_nb15"
RESULTS_DIR = BASE_DIR / "results" / "re2" / "centralized" / "tuning"
TRAIN_FILE = DATA_DIR / "train.npz"
VALIDATION_FILE = DATA_DIR / "validation.npz"
TEST_FILE = DATA_DIR / "test.npz"


def create_model(input_dim, config):
    model = tf.keras.Sequential([tf.keras.Input(shape=(input_dim,))])
    for index, units in enumerate(config["hidden_layers"]):
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        if index == 0:
            model.add(tf.keras.layers.Dropout(config["dropout"]))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss="binary_crossentropy")
    return model


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "specificity": float(tn / (tn + fp)),
        "false_positive_rate": float(fp / (tn + fp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def load_split(path):
    with np.load(path) as data:
        X, y = data["X"], data["y"].ravel()
    if X.ndim != 2 or len(X) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError(f"Partición inválida (X debe ser 2D; y, binaria 0/1): {path}")
    if not np.isfinite(X).all():
        raise ValueError(f"X contiene NaN o infinitos: {path}")
    return X, y.astype(np.int32)


def evaluate_cv(X_dev, y_dev):
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X_dev, y_dev))
    rows, training_rows = [], []
    for config in CONFIGURATIONS:
        fold_results = {float(t): [] for t in THRESHOLDS}
        for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
            tf.keras.backend.clear_session()
            random.seed(SEED + fold_number)
            np.random.seed(SEED + fold_number)
            tf.keras.utils.set_random_seed(SEED + fold_number)
            model = create_model(X_dev.shape[1], config)
            stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
            history = model.fit(X_dev[train_idx], y_dev[train_idx], validation_data=(X_dev[val_idx], y_dev[val_idx]),
                                epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, callbacks=[stopper], verbose=0)
            best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
            training_rows.append({"configuration": config["id"], "fold": fold_number, "best_epoch": best_epoch,
                                  "epochs_executed": len(history.history["loss"]), "best_val_loss": float(min(history.history["val_loss"]))})
            probabilities = model.predict(X_dev[val_idx], batch_size=BATCH_SIZE, verbose=0).ravel()
            for threshold in THRESHOLDS:
                fold_results[float(threshold)].append(calculate_metrics(y_dev[val_idx], probabilities, float(threshold)))
            print(f"{config['id']} | fold {fold_number}/{N_SPLITS} | best_epoch={best_epoch}")
        for threshold, results in fold_results.items():
            row = {"configuration": config["id"], "hidden_layers": str(config["hidden_layers"]),
                   "dropout": config["dropout"], "learning_rate": config["learning_rate"],
                   "num_params": int(model.count_params()), "threshold": threshold}
            for metric in METRICS:
                values = [result[metric] for result in results]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row["fpr_mean_plus_std"] = row["false_positive_rate_mean"] + row["false_positive_rate_std"]
            rows.append(row)
    tf.keras.backend.clear_session()
    return pd.DataFrame(rows), pd.DataFrame(training_rows)


def select_configuration(matrix):
    for fpr_limit in (MAX_FPR_ALLOWED, FALLBACK_MAX_FPR):
        eligible = matrix.loc[(matrix["recall_mean"] >= MIN_RECALL_ALLOWED) &
                              (matrix["fpr_mean_plus_std"] <= fpr_limit)].copy()
        if not eligible.empty:
            break
    if eligible.empty:
        raise RuntimeError("No hay candidatos con recall >= 0.80 y FPR medio + desv. <= 0.05; revisa full_cv_matrix_search.csv.")
    eligible = eligible.sort_values(["f1_score_mean", "fpr_mean_plus_std", "num_params", "configuration", "threshold"],
                                    ascending=[False, True, True, True, True])
    top_f1 = float(eligible.iloc[0]["f1_score_mean"])
    top = eligible.loc[eligible["f1_score_mean"] >= top_f1 - F1_TOLERANCE_MARGIN - 1e-12].copy()
    top = top.sort_values(["num_params", "fpr_mean_plus_std", "f1_score_mean", "configuration", "threshold"],
                          ascending=[True, True, False, True, True])
    return eligible, top, top.iloc[0], fpr_limit


def plot_cv_results(matrix, best_config):
    """Una figura por arquitectura: media y desviación entre folds, sin datos de TEST."""
    required = {"configuration", "threshold", "num_params", "f1_score_mean", "f1_score_std",
                "recall_mean", "recall_std", "false_positive_rate_mean", "false_positive_rate_std"}
    missing = required - set(matrix.columns)
    if missing:
        raise ValueError(f"Faltan columnas en full_cv_matrix_search.csv: {sorted(missing)}")
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for config in CONFIGURATIONS:
        rows = matrix.loc[matrix["configuration"] == config["id"]].sort_values("threshold")
        if len(rows) != len(THRESHOLDS):
            raise ValueError(f"{config['id']}: se esperaban {len(THRESHOLDS)} umbrales en la matriz CV.")
        x = rows["threshold"].to_numpy(dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for metric, label, color in (("f1_score", "F1", "#176b9c"), ("recall", "Recall", "#cf6a2e")):
            mean = rows[f"{metric}_mean"].to_numpy(dtype=float)
            std = rows[f"{metric}_std"].to_numpy(dtype=float)
            axes[0].plot(x, mean, "o-", color=color, linewidth=1.8, markersize=4, label=label)
            axes[0].fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.13)
        axes[0].axhline(MIN_RECALL_ALLOWED, color="#cf6a2e", linestyle=":", linewidth=1.3, label="Recall mínimo (0.80)")
        axes[0].set(title="Detección según umbral", xlabel="Umbral de clasificación", ylabel="Puntuación (0 a 1)", ylim=(0, 1.03))

        fpr = rows["false_positive_rate_mean"].to_numpy(dtype=float) * 100
        fpr_std = rows["false_positive_rate_std"].to_numpy(dtype=float) * 100
        axes[1].plot(x, fpr, "o-", color="#397a63", linewidth=1.8, markersize=4, label="FPR medio")
        axes[1].fill_between(x, np.maximum(0, fpr - fpr_std), fpr + fpr_std, color="#397a63", alpha=0.17, label="± 1 desv. entre folds")
        axes[1].axhline(MAX_FPR_ALLOWED * 100, color="#aa3743", linestyle="--", label="Objetivo CV (3 %)")
        axes[1].axhline(FALLBACK_MAX_FPR * 100, color="#8b6a37", linestyle=":", label="Respaldo CV (5 %)")
        axes[1].set(title="Falsas alarmas según umbral", xlabel="Umbral de clasificación", ylabel="FPR (%)",
                    ylim=(0, max(6, float(np.max(fpr + fpr_std)) * 1.08)))

        if config["id"] == best_config["configuration_id"]:
            for ax in axes:
                ax.axvline(best_config["classification_threshold"], color="#292d32", linestyle="--", linewidth=1.2,
                           label=f"Ganador (umbral {best_config['classification_threshold']:.2f})")
        for ax in axes:
            ax.set_xticks(x[::2])
            ax.grid(alpha=0.2)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"{config['id']} · {int(rows['num_params'].iloc[0]):,} parámetros · CV de {N_SPLITS} folds", fontsize=13)
        fig.text(0.5, 0.018, "Bandas: ± 1 desviación entre folds (descriptiva, no intervalo de confianza). Solo train + validation.",
                 ha="center", fontsize=8.5, color="#51555a")
        fig.tight_layout(rect=(0, 0.055, 1, 0.93))
        for extension in ("png", "pdf"):
            fig.savefig(figures_dir / f"cv_{config['id']}.{extension}", dpi=240, bbox_inches="tight")
        plt.close(fig)
    print(f"Figuras CV generadas (PNG y PDF): {figures_dir.relative_to(BASE_DIR)}")


def evaluate_final(X_dev, y_dev, best_config):
    test_metrics_path = RESULTS_DIR / "test_metrics.json"
    if test_metrics_path.exists():
        raise FileExistsError(f"TEST ya fue evaluado aquí: {test_metrics_path}. No lo uses para reajustar la selección.")
    X_test, y_test = load_split(TEST_FILE)
    if X_test.shape[1] != X_dev.shape[1] or len(np.unique(y_test)) != 2:
        raise ValueError("TEST debe tener el mismo número de atributos y ambas clases (0 y 1).")
    config = next(item for item in CONFIGURATIONS if item["id"] == best_config["configuration_id"])
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = create_model(X_dev.shape[1], config)
    # Sin conjunto de validación durante este reentrenamiento: épocas fijadas por CV.
    model.fit(X_dev, y_dev, epochs=best_config["final_training_epochs"], batch_size=BATCH_SIZE, shuffle=True, verbose=0)
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    metrics = calculate_metrics(y_test, probabilities, best_config["classification_threshold"])
    model.save(RESULTS_DIR / "centralized_final.keras")
    test_metrics_path.write_text(json.dumps({"dataset": "UNSW-NB15", "split": "test", "configuration_id": config["id"],
                                             "threshold": best_config["classification_threshold"], **metrics}, indent=2), encoding="utf-8")
    print(f"TEST final: F1={metrics['f1_score']:.4f}, Recall={metrics['recall']:.4f}, FPR={metrics['false_positive_rate']:.4f}")


def main():
    if PLOT_ONLY:
        matrix = pd.read_csv(RESULTS_DIR / "full_cv_matrix_search.csv")
        best_config = json.loads((RESULTS_DIR / "best_config.json").read_text(encoding="utf-8"))
        plot_cv_results(matrix, best_config)
        return
    if EVALUATE_TEST and (RESULTS_DIR / "test_metrics.json").exists():
        raise FileExistsError("Ya hay métricas de TEST en RESULTS_DIR; evita repetir la evaluación.")
    X_train, y_train = load_split(TRAIN_FILE)
    X_val, y_val = load_split(VALIDATION_FILE)
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train y validation deben tener el mismo número de atributos.")
    X_dev, y_dev = np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))
    counts = np.bincount(y_dev, minlength=2)
    if min(counts) < N_SPLITS:
        raise ValueError(f"Se requieren al menos {N_SPLITS} ejemplos de cada clase para StratifiedKFold.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"CV con {N_SPLITS} folds sobre train+validation: {X_dev.shape}; test aislado durante selección.")
    matrix, training = evaluate_cv(X_dev, y_dev)
    matrix.to_csv(RESULTS_DIR / "full_cv_matrix_search.csv", index=False)
    training.to_csv(RESULTS_DIR / "cv_training_epochs.csv", index=False)
    eligible, top, winner, fpr_limit = select_configuration(matrix)
    eligible.to_csv(RESULTS_DIR / "eligible_matrix.csv", index=False)
    top.to_csv(RESULTS_DIR / "top_candidates_cv.csv", index=False)
    config = next(item for item in CONFIGURATIONS if item["id"] == winner["configuration"])
    epochs = training.loc[training["configuration"] == config["id"], "best_epoch"].tolist()
    final_epochs = int(np.floor(np.median(epochs) + 0.5))
    best_config = {
        "dataset": "UNSW-NB15", "seed": SEED, "selection_dataset": "train+validation",
        "selection_strategy": "stratified_3_fold_cv_architecture_threshold", "class_weight": None,
        "selection_criteria": {"minimum_recall_mean": MIN_RECALL_ALLOWED, "maximum_fpr_mean_plus_std": fpr_limit,
                               "f1_tolerance": F1_TOLERANCE_MARGIN, "tie_break_order": ["num_params", "fpr_mean_plus_std", "f1_score_mean"]},
        "configuration_id": config["id"], "hidden_layers": config["hidden_layers"], "dropout": config["dropout"],
        "learning_rate": config["learning_rate"], "batch_size": BATCH_SIZE, "max_epochs_cv": MAX_EPOCHS,
        "patience_cv": PATIENCE, "num_params": int(winner["num_params"]),
        "classification_threshold": float(winner["threshold"]), "cv_best_epochs": epochs,
        "final_training_epochs": final_epochs,
        "cv_metrics": {metric: {"mean": float(winner[f"{metric}_mean"]), "std": float(winner[f"{metric}_std"])} for metric in METRICS},
        "cv_fpr_mean_plus_std": float(winner["fpr_mean_plus_std"]),
    }
    (RESULTS_DIR / "best_config.json").write_text(json.dumps(best_config, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_cv_results(matrix, best_config)
    print(f"Ganador: {config['id']} | parámetros={best_config['num_params']} | umbral={winner['threshold']:.2f}")
    print(f"CV: F1={winner['f1_score_mean']:.4f} ± {winner['f1_score_std']:.4f}, "
          f"Recall={winner['recall_mean']:.4f}, FPR={winner['false_positive_rate_mean']:.4f} ± {winner['false_positive_rate_std']:.4f}")
    print(f"Épocas finales: {final_epochs} | Resultados: {RESULTS_DIR.relative_to(BASE_DIR)}")
    if EVALUATE_TEST:
        evaluate_final(X_dev, y_dev, best_config)


if __name__ == "__main__":
    main()
