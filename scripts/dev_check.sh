#!/bin/bash

echo "=== VERIFICACION DEL ENTORNO DE TESIS ==="
echo

# 1. Verificar WSL
echo "[1] Verificando WSL..."
if grep -qi microsoft /proc/version; then
    echo "OK: Estas ejecutando dentro de WSL."
else
    echo "ERROR: No estas ejecutando dentro de WSL."
    exit 1
fi

echo

# 2. Verificar ruta del proyecto
echo "[2] Verificando proyecto..."
PROJECT_DIR="$HOME/proyectos/AprendizajeFederado"

if [ "$PWD" = "$PROJECT_DIR" ]; then
    echo "OK: Estas en la raiz del proyecto."
else
    echo "ADVERTENCIA: No estas en la raiz del proyecto."
    echo "Ruta actual: $PWD"
    echo "Ruta esperada: $PROJECT_DIR"
fi

echo

# 3. Verificar entorno virtual
echo "[3] Verificando entorno virtual..."

if [ -n "$VIRTUAL_ENV" ]; then
    echo "OK: Entorno virtual activo:"
    echo "$VIRTUAL_ENV"
else
    echo "ERROR: No hay un entorno virtual activo."
    echo
    echo "Activalo con:"
    echo "source ~/env_federado_tff/bin/activate"
    exit 1
fi

echo

# 4. Verificar que sea el entorno esperado
if [[ "$VIRTUAL_ENV" == *"env_federado_tff"* ]]; then
    echo "OK: Estas usando env_federado_tff."
else
    echo "ADVERTENCIA: Estas usando otro entorno virtual."
fi

echo

# 5. Verificar Python
echo "[4] Verificando Python..."
which python
python --version

echo

# 6. Verificar TensorFlow
echo "[5] Verificando TensorFlow..."

python - <<'PY'
try:
    import tensorflow as tf
    print("OK: TensorFlow", tf.__version__)
except Exception as e:
    print("ERROR al importar TensorFlow:")
    print(e)
    raise SystemExit(1)
PY

echo

# 7. Verificar TensorFlow Federated
echo "[6] Verificando TensorFlow Federated..."

python - <<'PY'
try:
    import tensorflow_federated as tff
    print("OK: TensorFlow Federated", tff.__version__)
except Exception as e:
    print("ERROR al importar TensorFlow Federated:")
    print(e)
    raise SystemExit(1)
PY

echo

# 8. Verificar estructura basica del proyecto
echo "[7] Verificando estructura..."

for dir in src tests data requirements scripts
do
    if [ -d "$dir" ]; then
        echo "OK: $dir/"
    else
        echo "ERROR: No existe $dir/"
    fi
done

echo
echo "=== ENTORNO LISTO ==="