import tensorflow as tf
import tensorflow_federated as tff


# Estructura de los datos sintéticos utilizados únicamente en RE1.3:
# - 2 características numéricas
# - 1 etiqueta binaria
INPUT_SPEC = (
    # Características: 2 valores numéricos (float32)
    tf.TensorSpec(shape=[None, 2], dtype=tf.float32), 
    # Etiqueta: 1 valor binario (float32)
    tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
)


def create_keras_model():
    """
    Crea el modelo mínimo utilizado para validar
    funcionalmente el mecanismo federado de RE1.3.

    Este modelo no corresponde al modelo IDS de OE2.
    """
    return tf.keras.Sequential(
        [
            # Capa de entrada que espera 2 características numéricas
            tf.keras.layers.Input(shape=(2,)),
            # capa de salida con 1 unidad y función de activación sigmoide para producir una probabilidad binaria
            tf.keras.layers.Dense(
                1, 
                activation="sigmoid",
            ),
        ]
    )


def model_fn():
    """
    Adapta el modelo Keras a la interfaz requerida
    por TensorFlow Federated.
    """
    keras_model = create_keras_model()

    return tff.learning.models.from_keras_model(
        keras_model=keras_model, 
        input_spec=INPUT_SPEC, # conocer el formato de entrada del modelo, que es una tupla de tensores con forma [None, 2] para las características y [None, 1] para las etiquetas.
        loss=tf.keras.losses.BinaryCrossentropy(), # definir la función de pérdida utilizada para entrenar el modelo, que es la entropía cruzada binaria.
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            )
        ],
    )