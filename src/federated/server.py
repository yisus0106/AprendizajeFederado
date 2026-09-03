import tensorflow_federated as tff

from src.federated.model import model_fn


def build_federated_process(
    # Controla cuánto puede avanzar en cada actualización
    # Ejemplo: peso_nuevo = peso_actual - 0.1 × gradiente
    client_learning_rate=0.1,
    # Estás indicando, simplificando, que el servidor aplica completamente 
    # la actualización agregada calculada mediante FedAvg.
    server_learning_rate=1.0,
):
    """
    Construye el proceso federado utilizado por el prototipo.

    El proceso emplea Federated Averaging (FedAvg) para
    combinar posteriormente las actualizaciones generadas
    por los clientes participantes.
    """

    # Se definen los optimizadores para el cliente y el servidor utilizando SGD con las tasas 
    # de aprendizaje especificadas.
    client_optimizer = tff.learning.optimizers.build_sgdm(
        learning_rate=client_learning_rate
    )
    # El optimizador del servidor también se define como SGD, pero con una 
    # tasa de aprendizaje diferente.
    server_optimizer = tff.learning.optimizers.build_sgdm(
        learning_rate=server_learning_rate
    )

    # Se construye el proceso federado utilizando la función de construcción de FedAvg 
    # proporcionada por TFF.
    federated_process = (
        tff.learning.algorithms.build_weighted_fed_avg(
            model_fn=model_fn, # Modelo que entrenaran los clientes
            client_optimizer_fn=client_optimizer, # cada cliente usara este optimizador para actualizar sus pesos locales
            server_optimizer_fn=server_optimizer, # el servidor usara este optimizador para actualizar los pesos globales agregados
        )
    )

    return federated_process