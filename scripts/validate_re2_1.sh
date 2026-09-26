#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo " VALIDACION DEL RE2.1"
echo " Baseline centralizado sobre UNSW-NB15"
echo "============================================"

# Evita omitir el tuning o evaluar TEST también desde el selector.
python - <<'PY'
import ast
from pathlib import Path

source = Path("src/models/centralized/tune_centralized.py")
flags = {}
for node in ast.parse(source.read_text(encoding="utf-8")).body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"PLOT_ONLY", "EVALUATE_TEST"}:
                flags[target.id] = ast.literal_eval(node.value)
if any(flags.get(name) is not False for name in ("PLOT_ONLY", "EVALUATE_TEST")):
    raise SystemExit("Configura PLOT_ONLY = False y EVALUATE_TEST = False en tune_centralized.py.")
for split in ("train", "validation", "test"):
    path = Path("data/processed/unsw_nb15") / f"{split}.npz"
    if not path.is_file():
        raise SystemExit(f"Falta {path}. Ejecuta primero python src/data/preprocess_unsw.py.")
PY

mkdir -p logs/re2_1

echo "[1/2] Seleccion por CV y generacion automatica de figuras"
python src/models/centralized/tune_centralized.py 2>&1 | tee logs/re2_1/tune_centralized.log

echo "[2/2] Entrenamiento final y evaluacion sobre TEST"
python src/models/centralized/train_centralized.py 2>&1 | tee logs/re2_1/train_centralized.log

echo "============================================"
echo " RE2.1: EJECUCION FINALIZADA CORRECTAMENTE"
echo " Revisa metrics.json y verifica recall >= 0.80."
echo " results/re2/centralized/train/metrics.json"
echo "============================================"
