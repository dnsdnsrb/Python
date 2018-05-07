import tensorflow as tf
import numpy as np

xy = np.loadtxt('train4', unpack=True, dtype='float32')

x = xy[0:-1]
y = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x, Y:y})
    if step % 20 == 0:
        print (step, sess.run(cost, feed_dict={X:x, Y:y}), sess.run(W))

print(sess.run(hypothesis, feed_dict={X: [ [1], [2], [2] ]}) > 0.5) #0.5보다 크면 참
print(sess.run(hypothesis, feed_dict={X: [ [1], [5], [5] ]}) > 0.5) #0.5보다 작으면 거
