import tensorflow as tf
import numpy as np
import input_data
import os
import shutil
import Imagenet
import DataSet

batch_size = 128
test_size = batch_size

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X ,p_keep_conv, p_keep_hidden):


    w1 = init_weights([5, 5, 3, 32])
    b1 = tf.Variable(tf.zeros([32]))

    w2 = init_weights([5, 5, 32, 64])
    b2 = tf.Variable(tf.zeros([64]))

    w3 = init_weights([5, 5, 64, 128])
    b3 = tf.Variable(tf.zeros([128]))

    w4 = init_weights([128 * 4 * 4, 1000])
    b4 = tf.Variable(tf.zeros([1000]))

    w5 = init_weights([1000, 500])
    b5 = tf.Variable(tf.zeros([500]))

    w6 = init_weights([500, 10])
    b6 = tf.Variable(tf.zeros([10]))
                                                                                               #(32, 32, 3)
    with tf.name_scope("conv1"):
        conv1 = tf.nn.relu(tf.nn.conv2d(X, w1, strides =[1, 1, 1, 1], padding='SAME'))         #(32, 32, 64)
        l1 = tf.nn.bias_add(conv1, b1)

    with tf.name_scope("maxpool1"):
        p1 = tf.nn.max_pool(l1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       #(16, 16, 64)
        p1 = tf.nn.dropout(p1, p_keep_conv)

    with tf.name_scope("conv2"):
        conv2 = tf.nn.relu(tf.nn.conv2d(p1, w2, strides=[1, 1, 1, 1], padding='SAME'))         #(16, 16, 128)
        l2 = tf.nn.bias_add(conv2, b2)

    with tf.name_scope("maxpool2"):
        m2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')      #(8, 8, 128)
        m2 = tf.nn.dropout(m2, p_keep_conv)

    with tf.name_scope("conv3"):
        conv3 = tf.nn.relu(tf.nn.conv2d(m2, w3, strides=[1, 1, 1, 1], padding='SAME'))         #(8, 8, 256)
        l3 = tf.nn.bias_add(conv3, b3)

    with tf.name_scope("maxpool3"):
        p3 = tf.nn.max_pool(l3, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')      #(4, 4, 256)
        p3 = tf.reshape(p3, [-1, w4.get_shape().as_list()[0]])
        p3 = tf.nn.dropout(p3, p_keep_conv)

    with tf.name_scope("fc4"):
        l4 = tf.nn.relu(tf.matmul(p3, w4) + b4)
        l4 = tf.nn.dropout(l4, p_keep_hidden)

    with tf.name_scope("fc5"):
        l5 = tf.nn.relu(tf.matmul(l4, w5) + b5)
        l5 = tf.nn.dropout(l5, p_keep_hidden)

    with tf.name_scope("fc6"):
        pyx = tf.matmul(l5, w6) + b6
    return pyx


X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

images, labels = Net.create_sets(128)

py_x = model(X, p_keep_conv, p_keep_hidden)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= py_x, labels= Y))
    train_op = tf.train.RMSPropOptimizer(learning_rate= 0.01, momentum= 0.9).minimize(cost)
    tf.summary.scalar("cost", cost)

with tf.name_scope("accuracy"): #accuracy는 tf를 이용하는게 좋다. tensorboard로 볼 수 있기 때문
    predict_op = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))
    acc_op = tf.reduce_mean(tf.cast(predict_op, "float"))
    tf.summary.scalar("accuracy", acc_op)



with tf.Session() as sess:
    tf.global_variables_initializer().run()

    sess.run(train_op, feed_dict={X: images, Y: labels,
                                  p_keep_conv: 0.8, p_keep_hidden: 0.8})