import tensorflow as tf
import numpy as np
import mnist
import NNutils
from datetime import datetime

input = tf.Variable([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
output = tf.split(input, 5, axis=1)

list = []
for i in output:
    list.append(i)

for i in list:
    print(i)

con = tf.concat(list, axis=1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    print(sess.run(con))