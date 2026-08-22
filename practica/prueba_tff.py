import tensorflow_federated as tff
import numpy as np


@tff.federated_computation(
    tff.FederatedType(np.int32, tff.CLIENTS)
)
def sumar_clientes(valores):
    return tff.federated_sum(valores)


resultado = sumar_clientes([1, 2, 3])

print("Resultado federado:", resultado)