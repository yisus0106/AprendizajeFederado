import tensorflow as tf

clientes_pyme1 = tf.constant([50, 60, 70])

clientes_pyme2 = tf.constant([30, 40, 20])

total_clientes = clientes_pyme1 + clientes_pyme2

print("Total de clientes:")
print(total_clientes)