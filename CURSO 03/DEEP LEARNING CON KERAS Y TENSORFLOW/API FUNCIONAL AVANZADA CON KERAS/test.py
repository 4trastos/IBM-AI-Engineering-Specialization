import tensorflow as tf
print(tf.executing_eagerly())

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
result = tf.add(a, b)
print(result)