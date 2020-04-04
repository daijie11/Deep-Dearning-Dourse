import tensorflow as tf


# import numpy as np


def dataInput():
    while (True):
        try:
            a = float(input("输入商品房面积："))
            b = int(input("输入商品房房间数： "))
            break
        except:
            print("Input ERROR !!! Please input again !")
            continue

    return a, b


x1 = tf.constant(
    [137.97, 104.5, 100, 124.32, 79.2, 99, 124, 114, 106.69, 138.05, 53.75, 46.91, 68, 63.02, 81.26, 86.21])
x2 = tf.constant([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2], dtype=tf.float32)
y = tf.constant([145, 110, 93, 116, 65.32, 104, 118, 91, 62, 133, 51, 45, 78.5, 69.65, 75.69, 95.3])
x0 = tf.fill([16], 1.0)

Y = tf.expand_dims(y, -1)  # 实现转置

X = tf.stack((x0, x1, x2), axis=1)
Xt = tf.transpose(X)

XtX_1 = tf.linalg.inv(tf.matmul(Xt, X))  # 求逆
XtX_1_Xt = tf.matmul(XtX_1, Xt)

W = tf.matmul(XtX_1_Xt, Y)
W = W.numpy()

print("面积：20-500之间的实数")
print("房间数：1-10之间的整数")
x1_test, x2_test = dataInput()
y_spread = W[1] * x1_test + W[2] * x2_test + W[0]

print("价格估计数是: {}".format(y_spread[0]))

