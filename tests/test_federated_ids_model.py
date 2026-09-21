import numpy as np
import tensorflow as tf

from src.federated.ids_model import (
    IDS_INPUT_SPEC,
    NUM_INPUT_FEATURES,
    create_ids_keras_model,
)

from src.federated.ids_server import (
    build_ids_federated_process,
)


def verify_federated_ids_model():

    print(
        "=== PRUEBA DEL MODELO IDS FEDERADO ==="
    )

    model = create_ids_keras_model()

    assert model.input_shape == (
        None,
        NUM_INPUT_FEATURES,
    )

    assert model.output_shape == (
        None,
        1,
    )

    dense_units = [
        layer.units
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Dense,
        )
    ]

    assert dense_units == [64, 32, 1]

    assert IDS_INPUT_SPEC[0].shape.as_list() == [
        None,
        192,
    ]

    assert IDS_INPUT_SPEC[1].shape.as_list() == [
        None,
        1,
    ]

    print(
        "Arquitectura: 192 -> 64 -> 32 -> 1"
    )

    process = build_ids_federated_process()

    state = process.initialize()

    model_weights = process.get_model_weights(
        state
    )

    trainable_shapes = [
        tuple(np.asarray(weight).shape)
        for weight in model_weights.trainable
    ]

    expected_shapes = [
        (192, 64),
        (64,),
        (64, 32),
        (32,),
        (32, 1),
        (1,),
    ]

    assert trainable_shapes == expected_shapes

    print(
        f"Parámetros entrenables: "
        f"{trainable_shapes}"
    )

    print(
        "Estado: MODELO IDS FEDERADO OPERATIVO"
    )


if __name__ == "__main__":
    verify_federated_ids_model()