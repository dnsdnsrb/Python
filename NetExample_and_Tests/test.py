import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

a =[1, 2, 3, 4, 5, 6, 7]
b = [2, 4, 6]
print(a[0:5])
print(a[5])

plt.plot(a)
plt.plot(b)
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()


# hello = tf.constant("Hello")
#
# sess = tf.Session() #세션을 정의한다
#
# print(hello)
# print(sess.run(hello))
#
# a = "hello"
# b = 'hello'
#
# if a == b:
#     print("hell")
#
# a = 1
# b = 1
# print(a + b)
#
# value = tf.Variable(tf.random_normal([5,60], stddev=0.01))
#
# split0, split1, split2 = tf.split(1, 3, value)
# tf.shape(split0)
#
# range(0, 2, 3)