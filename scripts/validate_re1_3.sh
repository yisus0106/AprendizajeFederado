#!/usr/bin/env bash

set -euo pipefail

echo "============================================"
echo " VALIDACION COMPLETA DEL RE1.3"
echo " Prototipo de aprendizaje federado"
echo "============================================"
echo

echo "[1/5] Verificacion del entorno"
python tests/verify_environment.py
echo

echo "[2/5] Verificacion de clientes federados"
python -m tests.test_client_data
echo

echo "[3/5] Verificacion del servidor y FedAvg"
python -m tests.test_server
echo

echo "[4/5] Verificacion de 5 rondas federadas"
python -m tests.test_federated_training
echo

echo "[5/5] Verificacion de localidad de datos"
python -m tests.test_data_locality
echo

echo "============================================"
echo " RE1.3 VALIDADO CORRECTAMENTE"
echo "============================================"