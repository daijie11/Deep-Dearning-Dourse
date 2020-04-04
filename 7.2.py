import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

mnist = tf.keras.datasets.mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

plt.figure()
plt.suptitle("MNIST测试集样本", fontsize="20", c="R")
for i in range(16):
    num = np.random.randint(1, len(test_x))
    plt.subplot(4, 4, i + 1)
    plt.title("标签值：{}".format(test_y[num]))

    plt.imshow(test_x[num])

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()