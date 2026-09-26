import tensorflow as tf
import tensorflow_federated as tff


NUM_INPUT_FEATURES = 192

IDS_INPUT_SPEC = (
    tf.TensorSpec(
        shape=[None, NUM_INPUT_FEATURES],
        dtype=tf.float32,
    ),
    tf.TensorSpec(
        shape=[None, 1],
        dtype=tf.float32,
    ),
)


def create_ids_keras_model():
    """
    Crea el modelo IDS utilizado en RE2.2.

    Su arquitectura reproduce exactamente el modelo
    centralizado seleccionado en RE2.1.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(NUM_INPUT_FEATURES,)
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),

            tf.keras.layers.Dropout(
                0.2
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
            ),
        ],
        name="federated_unsw_nb15_ids",
    )


def ids_model_fn():
    """
    Adapta el modelo IDS a la interfaz requerida
    por TensorFlow Federated.
    """

    keras_model = create_ids_keras_model()

    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=IDS_INPUT_SPEC,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            )
        ],
    )