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


class Network():
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.time_step = 1

        self.x = tf.placeholder("float64", [None, 47236])
        self.y = tf.placeholder("float64", [None, 103])

        self.dropout_conv = tf.placeholder("float64")
        self.dropout_normal = tf.placeholder("float64")

        self.train_list = []

    def model(self, x, layer = [200]):
        image_size = 32
        reshape_size = 0
        layer_num = 0

        x = tf.reshape(x, [-1, self.time_step, int(47236 / self.time_step)])
        output = x

        for i in layer:
            layer_num += 1
            with tf.variable_scope('lstm' + str(layer_num)):
                cell = tf.nn.rnn_cell.BasicLSTMCell(i, activation=tf.nn.relu)
                output, state = tf.nn.dynamic_rnn(cell, output, dtype=tf.float64)
                print(output)
                output = tf.nn.dropout(output, self.dropout_normal)

        # cell = tf.nn.rnn_cell.MultiRNNCell([lstm(i) for i in [800, 400, 200, 100]])
        # output, state = tf.nn.dynamic_rnn(cell, output, dtype=tf.float32)

        with tf.variable_scope('fc'):
            output = tf.layers.dense(output[:, -1], 103, activation=None)

        y = output
        return y

    def train(self):

        #input reshape
        # x = self.x
        # x = tf.sparse_tensor_to_dense(x)
        #
        # y = self.y
        # y = tf.sparse_tensor_to_dense(y)

        #learning rate
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
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y))
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
                    summary, \
                    _, loss, \
                    step = sess.run([merged,
                                     self.training, self.cost,
                                     self.global_step],
                                    feed_dict={self.x: csr_matrix(train_data[start:end]).toarray(),
                                               self.y: csr_matrix(train_label[start:end]).toarray(),
                                               self.dropout_conv: 1.0, self.dropout_normal: 1.0})

                    if step % 50 == 0:
                        writer.add_summary(summary, step)
                        print(step, datetime.now(), loss)

                        summary, \
                        loss, \
                        accuracy = sess.run([merged, self.cost, self.accuracy],
                                            feed_dict={self.x: csr_matrix(test_data[0:1000]).toarray(),
                                                       self.y: csr_matrix(test_label[0:1000]).toarray(),
                                                       self.dropout_conv: 1.0, self.dropout_normal: 1.0})

                        writer_test.add_summary(summary, step)

                        print("test results : ", accuracy, loss)
                # saver.save(sess, path + "/model.ckpt", step)

if __name__=="__main__":
    model = Network()
    model.run(100000)