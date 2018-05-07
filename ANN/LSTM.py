import tensorflow as tf
import numpy as np
import mnist
import NNutils

import scipy
from sklearn.datasets import fetch_rcv1
from datetime import datetime

def fc_layer(name, x, output_num, activation='none', dropout=None):
    weight = tf.get_variable(name, [x.get_shape().as_list()[1], output_num],
                             initializer=tf.random_normal_initializer(stddev=0.01))
    bias = tf.Variable(tf.zeros(output_num))

    y = tf.matmul(x, weight) + bias

    if activation == 'relu':
        y = tf.nn.relu(y)
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(y)
    elif activation == 'none':
        pass

    if not dropout is None:
        y = tf.nn.dropout(y, dropout)

    return y

def lstm(num):
    return tf.nn.rnn_cell.BasicLSTMCell(num)

class Network():
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)
        self.time_step = 4
        self.x = tf.placeholder("float32", [None, self.time_step, (28 * 28 / self.time_step)])
        self.y = tf.placeholder("int32", [None, 10])

        self.dropout_conv = tf.placeholder("float")
        self.dropout_normal = tf.placeholder("float")

        self.train_list = []

    def model(self, x, layer = [50, 50, 20]):
        image_size = 32
        output = x
        reshape_size = 0

        layer_num = 0

        # for i in layer:
        # layer_num += 1
        # with tf.variable_scope('lstm' + str(layer_num)):

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm(i) for i in [50, 50, 50]])
        output, state = tf.nn.dynamic_rnn(cell, output, dtype=tf.float32)


        with tf.variable_scope('fc'):
            output = tf.contrib.layers.fully_connected(output[:, -1], 10, activation_fn=None)

        y = output
        return y

    def train(self):
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
            weights = tf.ones([self.batch_size, self.time_step])

            # self.cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=y_, targets=self.y, weights=weights))

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y))
            self.training = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(self.cost, global_step=self.global_step)

            tf.summary.scalar("cost", self.cost)

        with tf.name_scope("accuracy"):
            compare = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(compare, "float"))

            tf.summary.scalar("accuarcy", self.accuracy)

    def run(self, step_limit):
        self.train()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # MNIST 데이터
            dataset = mnist.read_data_sets("MNIST_data/", one_hot=True)
            train_data, train_label, test_data, test_label = dataset.train.images, dataset.train.labels, \
                                                             dataset.test.images, dataset.test.labels
            train_data = train_data.reshape(-1, self.time_step, int(28 * 28 / self.time_step)) #TF의 RNN은 입력데이터의 rank가 높아야함(열이 많아야 함)
            test_data = test_data.reshape(-1, self.time_step, int(28 * 28 / self.time_step))   #mnist 데이터는 사실 적절하진 못함

            test_indices = np.arange(len(train_data))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:10000]

            path = "LSTM_mnist/" + str(step_limit) + ""
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
                                    feed_dict={self.x: train_data[start:end],
                                               self.y: train_label[start:end],
                                               self.dropout_conv: 1.0,
                                               self.dropout_normal: 0.5})

                    if step % 50 == 0:
                        writer.add_summary(summary, step)
                        print(step, datetime.now(), loss)

                summary, \
                loss, \
                accuracy = sess.run([merged, self.cost, self.accuracy],
                                    feed_dict={self.x: test_data,
                                               self.y: test_label,
                                               self.dropout_conv: 1.0,
                                               self.dropout_normal: 1.0})

                writer_test.add_summary(summary, step)
                print("test results : ", accuracy, loss)
                saver.save(sess, path + "/model.ckpt", step)

model = Network()
model.run(100000)