from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import sklearn
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


BASE_DIR = Path(__file__).resolve().parents[3]

TEST_FILE = (
    BASE_DIR / "data" / "processed" / "unsw_nb15" / "test.npz"
)

RESULTS_ROOT = (
    BASE_DIR / "results" / "re2" / "federated" 
)

PREDICTION_ROOT = (
    BASE_DIR / "artifacts" / "predictions"
)

# ============================================================
# Utilidades
# ============================================================

def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"{path}: se esperaba un objeto JSON.")

    return content


def write_json(path, content):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


# ============================================================
# Verificación del preprocesamiento
# ============================================================

def verify_preprocessing(manifest):
    """
    Comprueba los artefactos registrados al crear las particiones.

    Esto detecta cambios en el preprocesador o en sus metadatos.
    No sustituye la necesidad de utilizar el test.npz generado
    con ese mismo preprocesador.
    """
    registered = manifest.get("preprocessing_artifacts", {})
    checked = {}

    for name in ("metadata.json", "preprocessor.joblib"):
        expected_hash = registered.get(name)

        if expected_hash is None:
            continue

        path = TEST_FILE.parent / name

        if not path.is_file():
            raise FileNotFoundError(
                f"Falta el artefacto de preprocesamiento: {path}"
            )

        actual_hash = sha256_file(path)

        if actual_hash != expected_hash:
            raise ValueError(
                f"{name} cambió respecto del particionamiento. "
                "Utiliza los datos y el preprocesador del experimento."
            )

        checked[name] = actual_hash

    return checked


# ============================================================
# Carga de TEST
# ============================================================

def load_test(input_features):
    with np.load(TEST_FILE, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]

    if X.ndim != 2 or X.shape[1] != input_features:
        raise ValueError(
            "Las características de TEST no coinciden con el modelo."
        )

    if y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("Dimensiones inválidas de TEST.")

    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError(
            "TEST debe contener exactamente las clases 0 y 1."
        )

    if not np.isfinite(X).all():
        raise ValueError("TEST contiene características no finitas.")

    X = X.astype(np.float32, copy=False)
    y = y.astype(np.int32, copy=False)

    if not np.isfinite(X).all():
        raise ValueError("TEST contiene valores fuera del rango float32.")

    # Distribución del conjunto TEST oficial utilizado en la tesis.
    benign = int(np.count_nonzero(y == 0))
    malicious = int(np.count_nonzero(y == 1))

    if len(y) != 82332 or benign != 37000 or malicious != 45332:
        raise ValueError(
            "El archivo no coincide con el tamaño y distribución "
            "del TEST oficial esperado: 82 332 registros, "
            "37 000 benignos y 45 332 maliciosos."
        )

    return X, y


# ============================================================
# Evaluación del modelo global definitivo
# ============================================================

def evaluate(run_dir):
    run_dir = resolve_path(run_dir)

    model_path = run_dir / "global_model.keras"
    config_path = run_dir / "effective_config.json"
    summary_path = run_dir / "run_summary.json"
    manifest_path = run_dir / "partition_manifest.json"

    effective_config = read_json(config_path)
    training_summary = read_json(summary_path)
    manifest = read_json(manifest_path)

    if training_summary.get("status") != "completed":
        raise ValueError(
            "La ejecución no figura como completada."
        )

    if (
        training_summary["completed_rounds"]
        != training_summary["num_rounds_requested"]
    ):
        raise ValueError(
            "No se completaron todas las rondas configuradas."
        )

    if manifest.get("test_used") is not False:
        raise ValueError(
            "El manifiesto no declara la exclusión de TEST."
        )

    if training_summary.get("test_loaded") is not False:
        raise ValueError(
            "El registro de entrenamiento no declara TEST excluido."
        )

    model_hash = sha256_file(model_path)

    if model_hash != training_summary["model_sha256"]:
        raise ValueError(
            "El modelo no coincide con el archivo registrado "
            "al finalizar el entrenamiento."
        )

    model_config = effective_config["model"]

    threshold = model_config["classification_threshold"]

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("Umbral de clasificación inválido.")

    input_features = model_config["input_features"]
    batch_size = training_summary["batch_size"]

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Tamaño de lote inválido.")

    output_dir = RESULTS_ROOT / run_dir.name
    
    prediction_dir = PREDICTION_ROOT / run_dir.name

    if output_dir.exists():
        raise FileExistsError(
            f"Ya existe una evaluación en: {output_dir}\n"
            "Se cancela para evitar sobrescribir sus resultados."
        )

    preprocessing_checks = verify_preprocessing(manifest)
    X_test, y_test = load_test(input_features)

    # No se necesita compilar el modelo para obtener predicciones.
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )

    if tuple(model.input_shape) != (None, input_features):
        raise ValueError("La entrada del .keras es incompatible.")

    if tuple(model.output_shape) != (None, 1):
        raise ValueError("El modelo debe tener una salida binaria.")

    print("\nEVALUACIÓN FINAL DEL MODELO GLOBAL FEDERADO")
    print(f"Modelo: {model_path.relative_to(BASE_DIR)}")
    print(f"Rondas completadas: {training_summary['completed_rounds']}")
    print(f"Registros TEST: {len(y_test):,}")
    print(f"Umbral guardado: {threshold}\n")

    # Una sola inferencia sobre TEST; no hay ajuste ni entrenamiento.
    output = np.asarray(
        model.predict(
            X_test,
            batch_size=batch_size,
            verbose=1,
        )
    )

    if output.shape != (len(y_test), 1):
        raise ValueError("Dimensiones inesperadas en las predicciones.")

    probabilities = output[:, 0]

    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("El modelo produjo probabilidades inválidas.")

    # Misma convención que en la evaluación centralizada.
    predictions = (probabilities >= threshold).astype(np.int32)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = [int(value) for value in cm.ravel()]

    scores = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(
                y_test, predictions, pos_label=1, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, pos_label=1, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test, predictions, pos_label=1, zero_division=0
            )
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "false_positive_rate": fp / (fp + tn),
        "specificity": tn / (tn + fp),
        "false_negative_rate": fn / (fn + tp),
    }

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["benigno", "malicioso"],
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["benigno", "malicioso"],
        digits=6,
        zero_division=0,
    )

    results = {
        "dataset": "UNSW-NB15",
        "evaluation_split": "official_test",
        "model": "federated_global_MLP",
        "run_id": training_summary["run_id"],
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.relative_to(BASE_DIR)),
        "model_sha256": model_hash,
        "test_path": str(TEST_FILE.relative_to(BASE_DIR)),
        "test_sha256": sha256_file(TEST_FILE),
        "effective_config_sha256": sha256_file(config_path),
        "preprocessing_artifacts_verified": preprocessing_checks,
        "input_features": input_features,
        "num_clients": training_summary["num_clients"],
        "completed_rounds": training_summary["completed_rounds"],
        "local_epochs": training_summary["local_epochs"],
        "classification_threshold": float(threshold),
        "decision_rule": "probability >= threshold",
        "positive_class": 1,
        "zero_division_policy": 0,
        "samples": {
            "test": int(len(y_test)),
            "benign": tn + fp,
            "malicious": fn + tp,
        },
        **scores,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_order": [0, 1],
        "confusion_matrix_axes": {
            "rows": "true_class",
            "columns": "predicted_class",
        },
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "classification_report": report,
        "recall_minimum_check": {
            "minimum": 0.80,
            "satisfied": bool(scores["recall"] >= 0.80),
        },
        "threshold_adjusted_on_test": False,
        "versions": {
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir.mkdir(parents=True, exist_ok=False)

    write_json(output_dir / "metrics.json", results)
    write_json(output_dir / "effective_config.json", effective_config)

    pd.DataFrame(
        [{"metric": name, "value": value}
         for name, value in scores.items()]
    ).to_csv(output_dir / "metrics.csv", index=False)

    pd.DataFrame(
        cm,
        index=["real_benigno", "real_malicioso"],
        columns=["pred_benigno", "pred_malicioso"],
    ).to_csv(
        output_dir / "confusion_matrix.csv",
        index_label="clase_real",
    )

    pd.DataFrame({
        # Posición dentro de test.npz, no ID del CSV original.
        "test_row": np.arange(len(y_test)),
        "real": y_test,
        "probability": probabilities,
        "prediction": predictions,
    }).to_csv(prediction_dir / "predictions.csv", index=False)

    (output_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    print("\nMÉTRICAS FINALES SOBRE TEST")

    for name, value in scores.items():
        if name == "roc_auc":
            print(f"{name:22s}: {value:.6f}")
        else:
            print(f"{name:22s}: {value:.6f} ({value * 100:.2f} %)")

    print("\nMatriz de confusión [[TN, FP], [FN, TP]]:")
    print(cm)

    print("\nReporte por clase:")
    print(report_text)

    print(
        "Recall mínimo del 80 %: "
        + ("CUMPLE" if scores["recall"] >= 0.80 else "NO CUMPLE")
    )

    print(f"\nResultados guardados en:\n{output_dir.relative_to(BASE_DIR)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa el modelo global federado sobre TEST."
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Carpeta de la ejecución que contiene global_model.keras.",
    )

    args = parser.parse_args()
    evaluate(args.run_dir)


if __name__ == "__main__":
    main()
     
# Ejecución mediante argumentos (IMPORTANTE)
# Ejemplo:
# python -m src.models.federated.evaluate_federated --run-dir artifacts/runs/federated/ID_DE_TU_EJECUCION
# El ultimo valor corresponde la carpeta en donde esta el .keras