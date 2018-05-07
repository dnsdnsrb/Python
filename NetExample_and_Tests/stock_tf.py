import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# if "DISPLAY" not in os.environ:
#     matplotlib.use('Agg')

import matplotlib.pyplot as plt

class Stock():
    def __init__(self):
        seq_length = 7

    def read_file(self):
        file = tf.train.string_input_producer(["../Data/data-02-stock_daily.csv"], shuffle=False)

        reader = tf.TextLineReader()
        key, value



class Network():
    def __init__(self):
        self.seq_length = 7
        self.data_dim = 5
        self.hidden_dim = 10
        self.output_dim = 1
        self.learing_rate = 0.01

        self.x = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim])
        self.y = tf.placeholder(tf.float32, [None, 1])

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.train()

    def model(self, x, layers=[400, 300, 200, 100]):
        layer_num = 0
        output = x

        for layer in layers:
            with tf.variable_scope('lstm' + str(layer_num)):
                cell = tf.nn.rnn_cell.BasicLSTMCell(layer, activation=tf.nn.relu)
                output, state = tf.nn.dynamic_rnn(cell, output, dtype=tf.float32)
                layer_num += 1
                print(output.shape)
                # output = tf.nn.dropout(output, self.dropout_normal)

        output = tf.layers.dense(output[:, -1], 1, activation=None)
        y_ = output

        return y_

    def train(self):
        self.y_ = self.model(self.x)
        self.loss = tf.reduce_sum(tf.square(self.y_ - self.y))
        self.opt = tf.train.AdamOptimizer(self.learing_rate).minimize(self.loss, global_step=self.global_step)

        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.y_ - self.y)))

    def run(self, iterations = 500):
        dataset = Stock()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range(iterations):
                _, cost = sess.run([self.opt, self.loss], feed_dict={self.x: dataset.train_data,
                                                                     self.y: dataset.train_label})
                print(cost)

            predict, rmse = sess.run([self.y_, self.rmse], feed_dict={self.x: dataset.test_data,
                                                                      self.y: dataset.test_label})
            # print(rmse)
            # print(predict)
            plt.plot(dataset.test_label)
            plt.plot(predict)
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.show()




if __name__ == '__main__':
    net = Network()
    net.run()