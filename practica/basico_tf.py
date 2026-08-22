import tensorflow as tf

# 1. Definimos las ventas diarias de una PYME usando un Tensor Constante
ventas_pyme1 = tf.constant([150, 200, 180])

# 2. Imprimimos el tensor completo
print("Tensor de ventas:")
print(ventas_pyme1)

# 3. Hacemos una operación sencilla: sumar 50 a todas las ventas (ej. bono de ventas)
ventas_con_bono = ventas_pyme1 + 50

print("\nVentas con bono:")
print(ventas_con_bono)