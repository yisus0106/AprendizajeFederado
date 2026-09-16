# 1. Crear nueva estructura
mkdir -p results/re1_3
mkdir -p results/re2/centralized
mkdir -p artifacts/models/centralized
mkdir -p artifacts/predictions
mkdir -p artifacts/runs

# 2. Mover resultados RE1.3 ya versionados
git mv src/results/federated/fedavg_training_curves.png \
       results/re1_3/

git mv src/results/federated/fedavg_training_history.csv \
       results/re1_3/

# 3. Mover resultados centralizados ya versionados
git mv src/results/centralized/metrics.json \
       results/re2/centralized/

git mv src/results/centralized/training_history.csv \
       results/re2/centralized/

# 4. Dejar de versionar las predicciones individuales
git rm --cached src/results/centralized/predictions.csv

mv src/results/centralized/predictions.csv \
   artifacts/predictions/centralized_predictions.csv

# 5. Mover nuevos resultados de tuning
mv src/results/centralized/tuning \
   results/re2/centralized/

# 6. Mover el modelo entrenado
mv src/models/centralized/unsw_nb15_baseline.keras \
   artifacts/models/centralized/

# 7. Limpiar carpetas antiguas
rm -rf src/results
rm -rf src/data/processed
rm -rf src/privacy
rm -rf src/utils

# 8. Limpiar caché Python
find . -type d -name "__pycache__" -prune -exec rm -rf {} +