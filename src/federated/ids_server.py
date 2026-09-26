import tensorflow_federated as tff

from src.federated.ids_model import (
    ids_model_fn,
)


def build_ids_federated_process(
    client_learning_rate=0.001,
    server_learning_rate=1.0,
):
    """
    Construye FedAvg utilizando el modelo IDS
    basado en UNSW-NB15.
    """

    client_optimizer = (
        tff.learning.optimizers.build_adam(
            learning_rate=client_learning_rate
        )
    )

    server_optimizer = (
        tff.learning.optimizers.build_sgdm(
            learning_rate=server_learning_rate
        )
    )

    return (
        tff.learning.algorithms.build_weighted_fed_avg(
            model_fn=ids_model_fn,

            client_optimizer_fn=(
                client_optimizer
            ),

            server_optimizer_fn=(
                server_optimizer
            ),

            client_weighting=(
                tff.learning.ClientWeighting.NUM_EXAMPLES
            ),
        )
    )