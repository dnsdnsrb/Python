import tensorflow as tf

#tf Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)  #

hypothesis = tf.mul(X, W)

cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / (m)

init = tf.initialize_all_variables()

W_val = []
cost_val = []

sess = tf.Session()
sess.run(init)

for i in range(-30, 50):#실제론 0.1을 곱해서 -3~5까지 범위의 각 cost를 구하는 과정이다.
    print(i * 0.1, sess.run(cost, feed_dict={W: i*0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))