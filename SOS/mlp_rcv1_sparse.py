from sys import path    #파이썬은 현재 경로를 살핀 후, sys path를 살핀다. 따라서 sys path에 추가해주면 외부 파일도 인식
path.append('../Data/')

import tensorflow as tf
import numpy as np
import input_data
import NNutils

import scipy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from sklearn.datasets import fetch_rcv1
from scipy.sparse import csr_matrix

def lstm(num):
    return tf.nn.rnn_cell.BasicLSTMCell(num, activation=tf.nn.relu)


def fc_layer(x, output_num, activation='none', dropout=None, name='fc'):
    weight = tf.get_variable(name, [47236, output_num],
                             initializer=tf.random_normal_initializer(stddev=0.01))
    bias = tf.Variable(tf.zeros(output_num))

    y = tf.sparse_matmul(x, weight, a_is_sparse=True)
    y = tf.sparse_add(y, bias)

    if activation == 'relu':
        y = tf.nn.relu(y)
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(y)
    elif activation == 'none':
        pass

    if not dropout is None:
        y = tf.nn.dropout(y, dropout)
    return y

class Network():
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.time_step = 1

        self.x = tf.sparse_placeholder(tf.float64, [None, 47236])
        self.y = tf.sparse_placeholder("float64", [None, 103])

        self.dropout_conv = tf.placeholder("float64")
        self.dropout_normal = tf.placeholder("float64")

        self.train_list = []

    def model(self, x, layers = [1000]):
        image_size = 32
        reshape_size = 0
        layer_num = 0

        output = x
        print(self.x)
        print(x.get_shape())

        for layer in layers:
            with tf.variable_scope('fc' + str(layer_num)):
                output = fc_layer(output, layer, activation='relu')
                layer_num += 1

        with tf.variable_scope('output_fc'):
            output = fc_layer(output, 103, activation='none')

        y = output
        return y

    def train(self):
        with tf.name_scope("learning_rate"):
            learning_rate = tf.train.exponential_decay(0.001,
                                                       self.global_step,
                                                       (50000 / self.batch_size) * 10,
                                                       0.95, staircase=True)
            learning_rate = tf.maximum(0.0001, learning_rate)
            tf.summary.scalar("learning_rate", learning_rate)

        #model
        y_ = self.model(self.x)

        #cost and training
        with tf.name_scope("cost"):
            self.cost = tf.sparse_reduce_sum(tf.contrib.sparse_softmax_cross_entropy(logits=y_, labels=self.y))
            self.training = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(self.cost, global_step=self.global_step)

            tf.summary.scalar("cost", self.cost)

        with tf.name_scope("accuracy"):
            compare = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(compare, "float"))

            tf.summary.scalar("accuarcy", self.accuracy)

    def run(self, step_limit):
        self.train()

        rcv1 = fetch_rcv1(subset='train')
        train_data = rcv1.data
        train_label = rcv1.target

        rcv1 = fetch_rcv1(subset='test', random_state=1)
        test_data = rcv1.data
        test_label = rcv1.target

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            path = "LSTM/" + str(step_limit) + "rcv1"
            saver = NNutils.save(path, sess)
            writer, writer_test, merged = NNutils.graph(path, sess)

            step = sess.run(self.global_step)
            while step < step_limit:
                print("step :", step)

                for start, end in zip(range(0, train_data.shape[0], self.batch_size),
                                      range(self.batch_size, train_data.shape[0], self.batch_size)):
                    data = scipy.sparse.coo_matrix(train_data[start:end])
                    label = scipy.sparse.coo_matrix(train_label[start:end])
                    indices = np.array([data.row, data.col]).T

                    summary, \
                    _, loss, \
                    step = sess.run([merged,
                                     self.training, self.cost,
                                     self.global_step],
                                    feed_dict={self.x: (indices, data.data, data.shape),
                                               self.y: (indices, label.data, label.shape),
                                               self.dropout_conv: 1.0, self.dropout_normal: 1.0})

                    if step % 50 == 0:
                        writer.add_summary(summary, step)
                        print(step, datetime.now(), loss)

                        summary, \
                        loss, \
                        accuracy = sess.run([merged, self.cost, self.accuracy],
                                            feed_dict={self.x: test_data[0:1000],
                                                       self.y: test_label[0:1000],
                                                       self.dropout_conv: 1.0, self.dropout_normal: 1.0})

                        writer_test.add_summary(summary, step)

                        print("test results : ", accuracy, loss)
                # saver.save(sess, path + "/model.ckpt", step)

if __name__=="__main__":
    model = Network()
    model.run(100000)