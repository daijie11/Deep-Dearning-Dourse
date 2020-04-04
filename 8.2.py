import tensorflow as tf
import numpy as np


x = tf.constant([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
y = tf.constant([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

n = len(x)
x_sum = tf.reduce_sum(x)
y_sum = tf.reduce_sum(y)
Numerator = n * tf.reduce_sum(tf.multiply(x, y)) - tf.reduce_sum(y_sum * x)
Denominator = n * tf.reduce_sum(tf.pow(x, 2)) - tf.pow(x_sum, 2)


w = Numerator / Denominator

b = (y_sum - w * x_sum) / n

print("w: {}".format(w))
print("b: {}".format(b))