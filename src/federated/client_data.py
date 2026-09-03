import numpy as np
import tensorflow as tf


class ClientPartition:
    """
    Representa la partición local de entrenamiento
    asignada a un cliente federado simulado.
    """
    # Estamos asumiendo que cada cliente tiene exactamente 2 características numéricas 
    # y 1 etiqueta binaria, como se define en INPUT_SPEC en model.py.
    # Por eso no verificamos que las features tengan exactamente 2 columnas, pero si que 
    # las labels tengan exactamente 1 columna.
    def __init__(self, client_id, features, labels):
        self.client_id = client_id
        # Convertir las features y labels a matrices NumPy de tipo float32
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)

        # Asegurarse de que las labels tengan la forma correcta (n, 1)
        if self.labels.ndim == 1:
            self.labels = self.labels.reshape(-1, 1)

        # Validar la estructura de la partición local
        self._validate()

    def _validate(self):
        """
        Valida la estructura de la partición local antes
        de permitir su utilización en el entrenamiento.
        """

        if self.features.ndim != 2:
            raise ValueError(
                f"{self.client_id}: features debe ser una matriz bidimensional."
            )

        if self.labels.ndim != 2:
            raise ValueError(
                f"{self.client_id}: labels debe ser una matriz bidimensional."
            )

        if len(self.features) == 0:
            raise ValueError(
                f"{self.client_id}: la partición local no puede estar vacía."
            )

        if len(self.features) != len(self.labels):
            raise ValueError(
                f"{self.client_id}: features y labels deben tener "
                "la misma cantidad de registros."
            )

        if not np.isfinite(self.features).all():
            raise ValueError(
                f"{self.client_id}: features contiene valores no finitos."
            )

        if not np.isfinite(self.labels).all():
            raise ValueError(
                f"{self.client_id}: labels contiene valores no finitos."
            )

    @property
    def num_examples(self):
        """
        Cantidad de ejemplos disponibles localmente.
        """
        return len(self.features)

    def to_tf_dataset(
        self,
        batch_size=4,
        local_epochs=1, # Número de veces que cada cliente iterará sobre su dataset local durante el entrenamiento.
        shuffle=True,
        # La semilla me indica desde donde abrira un libro puede ser cualquier numero 
        # si no te convence
        seed=42,
    ):
        """
        Construye el tf.data.Dataset que será utilizado
        exclusivamente para el entrenamiento de este cliente.
        """
        
        # Crear un dataset a partir de las features y labels locales
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.features, self.labels)
        )
        # Si se requiere barajar los datos, aplicar el método shuffle con un buffer 
        # del tamaño de la partición local y una semilla para reproducibilidad.
        if shuffle:
            dataset = dataset.shuffle(
                # El tamaño del buffer de barajado se establece en el número de ejemplos 
                # locales para garantizar un barajado completo.
                buffer_size=self.num_examples,
                # Semilla para garantizar que el barajado sea reproducible entre ejecuciones.
                seed=seed,
                # Indica que se debe volver a barajar los datos en cada iteración del dataset.
                reshuffle_each_iteration=True,
            )

        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(local_epochs)

        return dataset


def create_synthetic_client_partitions(
    num_clients=2,
    samples_per_client=12,
    seed=42,
):
    """
    Genera datos sintéticos únicamente para validar
    funcionalmente el mecanismo federado de RE1.3.

    No corresponde al dataset UNSW-NB15 utilizado en OE2.
    """

    if num_clients < 2:
        raise ValueError(
            "El prototipo requiere al menos dos clientes federados."
        )

    if samples_per_client <= 0:
        raise ValueError(
            "Cada cliente debe recibir al menos un ejemplo."
        )
    # Se utiliza un generador de números aleatorios de NumPy para garantizar
    # la reproducibilidad de los datos sintéticos generados. 
    # La semilla proporcionada asegura que cada ejecución del código produzca 
    # los mismos datos para los clientes.
    rng = np.random.default_rng(seed)

    total_samples = num_clients * samples_per_client

    features = rng.normal(
        loc=0.0, # Media de la distribución normal, centrada en 0.
        scale=1.0, # Desviación estándar de la distribución normal, controlando la dispersión de los datos.
        size=(total_samples, 2),
    ).astype(np.float32)

    # Problema binario sintético y determinista.
    scores = features[:, 0] + 0.5 * features[:, 1]

    # La etiqueta binaria se determina a partir de la suma ponderada de las características.
    # Si la suma ponderada es mayor que cero, la etiqueta es 1; de lo contrario, es 0.
    labels = (scores > 0).astype(np.float32).reshape(-1, 1)

    partitions = {}

    for index in range(num_clients):
        start = index * samples_per_client
        end = start + samples_per_client

        client_id = f"client_{index + 1:02d}"

        partitions[client_id] = ClientPartition(
            client_id=client_id,
            features=features[start:end],
            labels=labels[start:end],
        )

    return partitions