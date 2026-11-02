"""RE2.3: compara los modelos oficiales guardados, sin entrenar ni ajustar TEST.

Guardar en src/evaluation/compare_models.py y ejecutar desde la raiz:
    python src/evaluation/compare_models.py
"""

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data/processed/unsw_nb15"
FEDERATED_RUN_ID = "20260926T183336Z_3fc39f61"
FED_DIR = BASE_DIR / "artifacts/runs/federated" / FEDERATED_RUN_ID
FED_RESULTS = BASE_DIR / "results/re2/federated" / FEDERATED_RUN_ID
BATCH_SIZE = 256
THRESHOLD = 0.80
REFERENCE_TOLERANCE = 1e-6

METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "false_positive_rate": "FPR",
    "false_negative_rate": "FNR",
    "specificity": "Especificidad",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average Precision (AP)",
}
SCORE_METRICS = {"roc_auc", "average_precision"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        result = subprocess.run(
            ["git", "-C", str(BASE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "tensorflow"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def validate_inputs(files, hashes):
    central = read_json(files["central_reference"])
    federated = read_json(files["federated_reference"])
    config = read_json(files["effective_config"])["model"]
    summary = read_json(files["run_summary"])
    partition = read_json(files["partition_manifest"])

    require(central["dataset"] == federated["dataset"] == "UNSW-NB15",
            "Los reportes deben corresponder a UNSW-NB15.")
    require(central["evaluation_dataset"] == "test"
            and federated["evaluation_split"] == "official_test",
            "Los reportes no declaran el TEST esperado.")
    require(federated["run_id"] == summary["run_id"] == FEDERATED_RUN_ID,
            "La ejecucion federada no coincide con la referencia oficial.")
    require(summary["status"] == "completed"
            and summary["completed_rounds"] == summary["num_rounds_requested"] == 20,
            "La ejecucion debe haber completado sus 20 rondas.")
    require(summary["local_epochs"] == 1 and summary["num_clients"] == 3
            and summary["all_clients_each_round"] is True,
            "Se espera una epoca local y participacion de los tres clientes.")
    require(summary["test_loaded"] is False and partition["test_used"] is False,
            "Los registros no declaran TEST excluido del entrenamiento.")
    require(partition["pool"] == "train_validation"
            and partition["non_iid"] is False
            and partition["method"] == "random_stratified"
            and partition["num_clients"] == 3,
            "Este script corresponde a la comparacion principal IID.")
    require(partition["verification"]["all_rows_assigned_once"] is True
            and partition["preprocessing_refitted"] is False,
            "Revisar las verificaciones declaradas en el manifiesto.")
    require(central["selection_dataset"] == "train+validation"
            and central["samples"]["train_plus_validation"]
            == partition["total_samples"] == summary["unique_training_samples"] == 175341,
            "Los reportes no declaran el mismo volumen de entrenamiento.")
    require(central["epochs_trained"] == central["final_training_epochs"] == 20,
            "El baseline esperado utiliza 20 epocas.")
    require(central["input_features"] == federated["input_features"]
            == config["input_features"] == partition["input_features"] == 192,
            "Se esperan 192 caracteristicas en ambos enfoques.")
    require(central["classification_threshold"]
            == federated["classification_threshold"]
            == config["classification_threshold"] == THRESHOLD,
            "Los umbrales deben coincidir con 0.80.")
    require(central["configuration_id"] == config["configuration_id"] == "C5_Pyme_Max"
            and central["architecture"]["hidden_layers"] == config["hidden_layers"] == [64, 32]
            and central["architecture"]["dropout"] == config["dropout"] == 0.2
            and central["architecture"]["num_params"] == config["num_params"] == 14465,
            "Las configuraciones no corresponden a la arquitectura comun.")
    require(hashes["federated_model"] == summary["model_sha256"]
            == federated["model_sha256"], "Cambio el modelo federado oficial.")
    require(hashes["test"] == federated["test_sha256"],
            "TEST no coincide con la evaluacion federada oficial.")
    require(hashes["effective_config"] == federated["effective_config_sha256"],
            "Cambio la configuracion efectiva de la ejecucion.")
    for name in ("metadata.json", "preprocessor.joblib"):
        key = "metadata" if name == "metadata.json" else "preprocessor"
        require(hashes[key] == partition["preprocessing_artifacts"][name]
                == federated["preprocessing_artifacts_verified"][name],
                f"Cambio el artefacto de preprocesamiento: {name}")
    sources = partition["sources"]
    require(len(sources) == 2, "Se esperan TRAIN y VALIDATION como fuentes.")
    expected = {files["train"].resolve(), files["validation"].resolve()}
    observed = set()
    for source in sources:
        path = (BASE_DIR / source["file"]).resolve()
        require(path in expected and path not in observed,
                "Fuentes de entrenamiento inesperadas o repetidas.")
        observed.add(path)
        key = "train" if path == files["train"].resolve() else "validation"
        require(hashes[key] == source["sha256"], f"Cambio la fuente: {path.name}")
    return central, federated


def load_test(path):
    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"].ravel()
    require(X.shape == (82332, 192) and y.shape == (82332,),
            "TEST debe tener 82332 filas y 192 caracteristicas.")
    require(np.isfinite(X).all() and np.array_equal(np.unique(y), [0, 1]),
            "TEST contiene caracteristicas invalidas o etiquetas inesperadas.")
    require(np.count_nonzero(y == 0) == 37000
            and np.count_nonzero(y == 1) == 45332,
            "La distribucion de clases de TEST no coincide.")
    X = X.astype(np.float32, copy=False)
    require(np.isfinite(X).all(), "TEST excede el rango float32.")
    return X, y.astype(np.int32)


def predict_model(path, X):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(str(path), compile=False)
    require(tuple(model.input_shape) == (None, 192)
            and tuple(model.output_shape) == (None, 1)
            and model.count_params() == 14465,
            f"Dimensiones o parametros inesperados: {path.name}")
    signature = []
    for layer in model.layers:
        kind = type(layer).__name__
        if kind == "InputLayer":
            continue
        if kind == "Dense":
            signature.append((kind, layer.units, layer.activation.__name__))
        elif kind == "Dropout":
            signature.append((kind, layer.rate))
        else:
            raise ValueError(f"Capa inesperada en {path.name}: {kind}")
    require(signature == [("Dense", 64, "relu"), ("Dropout", 0.2),
                          ("Dense", 32, "relu"), ("Dense", 1, "sigmoid")],
            f"La arquitectura de {path.name} no coincide con C5_Pyme_Max.")
    output = np.asarray(model.predict(X, batch_size=BATCH_SIZE, verbose=0))
    require(output.shape == (len(X), 1) and np.isfinite(output).all(),
            f"Salida invalida de {path.name}")
    probabilities = output[:, 0]
    require(((probabilities >= 0) & (probabilities <= 1)).all(),
            "El modelo debe producir probabilidades entre 0 y 1.")
    return probabilities


def evaluate(y, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(np.int32)
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "specificity": tn / (tn + fp),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y, predictions, labels=[0, 1], target_names=["benigno", "malicioso"],
            output_dict=True, zero_division=0,
        ),
    }, predictions


def reference_differences(current, reference):
    differences = {}
    for key in METRICS:
        if key in reference and not np.isclose(
            current[key], reference[key], rtol=0, atol=REFERENCE_TOLERANCE
        ):
            differences[key] = {"current": current[key], "reference": reference[key]}
    if current["confusion_matrix"] != reference["confusion_matrix"]:
        differences["confusion_matrix"] = {
            "current": current["confusion_matrix"], "reference": reference["confusion_matrix"],
        }
    return differences


def save_figure(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(directory, y, results, probabilities):
    colors = ["#2563A6", "#DB7B24"]
    labels = ["Centralizado", "Federado IID"]
    names = ["centralized", "federated"]
    keys = ["accuracy", "precision", "recall", "f1_score"]
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    positions = np.arange(len(keys))
    for index, name in enumerate(names):
        values = [100 * results[name][key] for key in keys]
        bars = ax.bar(positions + (index - 0.5) * 0.36, values, 0.36,
                      label=labels[index], color=colors[index])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set(xticks=positions, xticklabels=[METRICS[key] for key in keys],
           ylabel="Porcentaje (%)", ylim=(0, 108),
           title="Metricas sobre TEST (umbral 0.80)")
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(loc="lower left")
    save_figure(fig, directory, "metrics_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, name, label in zip(axes, names, labels):
        cm = np.asarray(results[name]["confusion_matrix"])
        percentages = 100 * cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{cm[row, col]:,}\n{percentages[row, col]:.2f}%",
                        ha="center", va="center",
                        color="white" if percentages[row, col] > 50 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["Benigno", "Malicioso"], yticklabels=["Benigno", "Malicioso"],
               xlabel="Clase predicha", ylabel="Clase real", title=label)
    fig.colorbar(im, ax=list(axes), label="Porcentaje por clase real (%)", shrink=0.8)
    save_figure(fig, directory, "confusion_matrices")

    fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    fig_pr, ax_pr = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    for index, name in enumerate(names):
        scores = results[name]
        fpr, tpr, thresholds = roc_curve(y, probabilities[name], drop_intermediate=False)
        pd.DataFrame({"fpr": fpr, "tpr": tpr,
                      "threshold": np.where(np.isfinite(thresholds), thresholds, np.nan)}
                     ).to_csv(directory.parent / f"roc_{name}.csv", index=False)
        ax_roc.plot(fpr, tpr, color=colors[index],
                    label=f"{labels[index]} (AUC={scores['roc_auc']:.4f})")
        ax_roc.scatter(scores["false_positive_rate"], scores["recall"],
                       color=colors[index], edgecolors="black", zorder=3)
        precision, recall, thresholds = precision_recall_curve(y, probabilities[name])
        pd.DataFrame({"precision": precision, "recall": recall,
                      "threshold": np.r_[thresholds, np.nan]}
                     ).to_csv(directory.parent / f"pr_{name}.csv", index=False)
        ax_pr.step(recall, precision, where="post", color=colors[index],
                   label=f"{labels[index]} (AP={scores['average_precision']:.4f})")
        ax_pr.scatter(scores["recall"], scores["precision"],
                      color=colors[index], edgecolors="black", zorder=3)
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", label="Referencia aleatoria")
    ax_pr.axhline(float(y.mean()), linestyle="--", color="gray", label="Prevalencia maliciosa")
    ax_roc.set(xlabel="Tasa de falsos positivos", ylabel="Recall malicioso",
               title="Curvas ROC; puntos: umbral 0.80", xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.set(xlabel="Recall malicioso", ylabel="Precision maliciosa",
              title="Curvas precision-recall; puntos: umbral 0.80",
              xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_pr.legend(loc="lower left", fontsize=9)
    save_figure(fig_roc, directory, "roc_comparison")
    save_figure(fig_pr, directory, "precision_recall_comparison")


def main():
    comparison_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = BASE_DIR / "results/re2/comparison" / comparison_id
    prediction_dir = BASE_DIR / "artifacts/predictions/comparison" / comparison_id
    log_path = BASE_DIR / "logs/re2_3" / f"{comparison_id}.log"
    output.mkdir(parents=True, exist_ok=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8"),
    ], force=True)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    record = {
        "status": "running", "comparison_id": comparison_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "federated_run_id": FEDERATED_RUN_ID, "git_commit": git_commit(),
        "versions": package_versions(), "classification_threshold": THRESHOLD,
        "positive_class": 1, "decision_rule": "probability >= threshold",
        "reference_tolerance": REFERENCE_TOLERANCE,
        "training_performed": False, "threshold_adjusted_on_test": False,
        "confusion_matrix_axes": {"rows": "true_class", "columns": "predicted_class"},
        "historical_provenance_note": (
            "El reporte centralizado original no incluye hashes del modelo ni de TEST. "
            "Esta comparacion registra sus hashes actuales y contrasta sus predicciones "
            "con las metricas historicas. No certifica retrospectivamente esos archivos."
        ),
    }
    write_json(output / "manifest.json", record)
    try:
        files = {
            "centralized_model": BASE_DIR / "artifacts/models/centralized/unsw_nb15_baseline.keras",
            "federated_model": FED_DIR / "global_model.keras",
            "central_reference": BASE_DIR / "results/re2/centralized/train/metrics.json",
            "federated_reference": FED_RESULTS / "metrics.json",
            "effective_config": FED_DIR / "effective_config.json",
            "run_summary": FED_DIR / "run_summary.json",
            "partition_manifest": FED_DIR / "partition_manifest.json",
            "test": DATA_DIR / "test.npz", "train": DATA_DIR / "train.npz",
            "validation": DATA_DIR / "validation.npz",
            "metadata": DATA_DIR / "metadata.json",
            "preprocessor": DATA_DIR / "preprocessor.joblib",
            "comparison_script": Path(__file__).resolve(),
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        require(not missing, "Faltan archivos:\n" + "\n".join(missing))
        hashes = {key: sha256(path) for key, path in files.items()}
        record["inputs"] = {key: {"path": str(path), "sha256": hashes[key]}
                            for key, path in files.items()}
        central_ref, federated_ref = validate_inputs(files, hashes)
        X, y = load_test(files["test"])
        logging.info("Comparacion: %s | TEST: %s filas | Umbral: %.2f",
                     comparison_id, len(y), THRESHOLD)
        results, probabilities, predictions = {}, {}, {}
        for name in ("centralized", "federated"):
            logging.info("Evaluando %s...", name)
            probabilities[name] = predict_model(files[f"{name}_model"], X)
            results[name], predictions[name] = evaluate(y, probabilities[name])
        differences = {
            "centralized": reference_differences(results["centralized"], central_ref),
            "federated": reference_differences(results["federated"], federated_ref),
        }
        write_json(output / "reference_check.json", differences)
        write_json(output / "evaluated_metrics.json", results)
        require(not any(differences.values()),
                "Las metricas no coinciden con las referencias. Revisar reference_check.json; "
                "no usar esta ejecucion como comparacion oficial.")
        require(all(sha256(path) == hashes[key] for key, path in files.items()),
                "Un archivo de entrada cambio durante la evaluacion.")

        rows = []
        for key, label in METRICS.items():
            c, f = results["centralized"][key], results["federated"][key]
            rows.append({"metric": key, "label": label, "centralized": c, "federated": f,
                         "difference": f - c,
                         "difference_percentage_points": None if key in SCORE_METRICS else 100 * (f - c),
                         "preferred_direction": "lower" if key in {"false_positive_rate", "false_negative_rate"} else "higher"})
        table = pd.DataFrame(rows)
        table.to_csv(output / "metrics_comparison.csv", index=False)
        errors = pd.DataFrame([
            {"outcome": key, "centralized": results["centralized"][key],
             "federated": results["federated"][key],
             "difference": results["federated"][key] - results["centralized"][key]}
            for key in ("tn", "fp", "fn", "tp")
        ])
        errors.to_csv(output / "confusion_counts_comparison.csv", index=False)
        prediction_dir.mkdir(parents=True, exist_ok=False)
        pd.DataFrame({
            "test_row": np.arange(len(y)), "real": y,
            "centralized_probability": probabilities["centralized"],
            "centralized_prediction": predictions["centralized"],
            "federated_probability": probabilities["federated"],
            "federated_prediction": predictions["federated"],
        }).to_csv(prediction_dir / "predictions.csv", index=False)
        figures = output / "figures"
        figures.mkdir()
        make_figures(figures, y, results, probabilities)
        record.update({
            "status": "completed", "reference_metrics_matched": True,
            "test_samples": len(y), "benign_samples": int((y == 0).sum()),
            "malicious_samples": int((y == 1).sum()),
            "training_example_visits": {"centralized": 175341 * 20, "federated": 175341 * 20 * 1},
            "predictions_path": str(prediction_dir / "predictions.csv"),
            "predictions_sha256": sha256(prediction_dir / "predictions.csv"),
            "log_path": str(log_path),
        })
        write_json(output / "comparison.json", {"protocol": record, "models": results, "differences": rows})
        logging.info("\n%s", table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        logging.info("\nDiferencia FP: %+d | Diferencia FN: %+d",
                     results["federated"]["fp"] - results["centralized"]["fp"],
                     results["federated"]["fn"] - results["centralized"]["fn"])
        logging.info("COMPARACION COMPLETADA\nResultados: %s\nPredicciones: %s\nLog: %s",
                     output, prediction_dir, log_path)
    except Exception as error:
        record.update({"status": "failed", "error": str(error)})
        logging.exception("COMPARACION CANCELADA. Revisar insumos antes de continuar.")
        raise
    finally:
        write_json(output / "manifest.json", record)


if __name__ == "__main__":
    main()