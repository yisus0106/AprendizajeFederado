import platform
import sys

import numpy as np
import scipy
import tensorflow as tf
import tensorflow_federated as tff


def main():
    print("=== VERIFICACION DEL ENTORNO FEDERADO ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.platform()}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"TensorFlow Federated: {tff.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print("Estado: ENTORNO OPERATIVO")


if __name__ == "__main__":
    main()