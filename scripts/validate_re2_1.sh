#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo " VALIDACION COMPLETA DEL RE2.1"
echo " Baseline centralizado sobre UNSW-NB15"
echo "============================================"
echo

echo "[1/3] Evaluacion de configuraciones y seleccion de la mejor"
echo "(Asegúrate de que PLOT_ONLY = False en tune_centralized.py)"
python src/models/centralized/tune_centralized.py
echo

echo "============================================"
echo " ACCIÓN REQUERIDA ANTES DE CONTINUAR"
echo "============================================"
echo "Para generar las figuras de la validación cruzada (CV), debes"
echo "modificar manualmente el archivo:"
echo "src/models/centralized/tune_centralized.py"
echo ""
echo "Abre el archivo, busca la variable global y cámbiala a:"
echo "PLOT_ONLY = True"
echo "============================================"
echo
read -p "Presiona [Enter] cuando hayas guardado el cambio para generar las figuras..."
echo

echo "[2/3] Generación de figuras CV"
python src/models/centralized/tune_centralized.py
echo

echo "[3/3] Entrenamiento y evaluacion del baseline centralizado"
python src/models/centralized/train_centralized.py
echo

echo "============================================"
echo " RE2.1 FINALIZADO CORRECTAMENTE"
echo "============================================"