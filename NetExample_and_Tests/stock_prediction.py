import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# if "DISPLAY" not in os.environ:
#     matplotlib.use('Agg')

import matplotlib.pyplot as plt

class Stock():
    def __init__(self):
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.seq_length = 7

        self.make_data()

    def make_data(self):
        data =  np.loadtxt('../Data/data-02-stock_daily.csv', delimiter=',')
        data = data[::-1]  # 열 순서를 뒤집음(끝 열이 첫 열로)

        data = self.MinMaxScalar(data)

        train_data = data
        train_label = data[:, [-1]] #-1을 하면 끝을 나타냄 => 뒤집는다.

        train_data, train_label = self.make_seq(train_data, train_label)
        self.train_data, self.train_label, \
        self.test_data, self.test_label = self.spilt_data(train_data, train_label)

    def make_seq(self, train_data, train_label):
        data = []
        label = []

        for i in range(0, len(train_label) - self.seq_length):
            x = train_data[i:i + self.seq_length]   #i ~ i + 6개를 묶는다.(0부터 시작)
            y = train_label[i + self.seq_length]    #i + 7번째 label
            # print(x, "->", y)
            data.append(x)
            label.append(y)
        print(data[0])
        return data, label

    def spilt_data(self, data, label, split_rate=0.7):
        train_size = int(len(label) * split_rate)

        train_data = np.array(data[:train_size])
        test_data = np.array(data[train_size:])

        train_label = np.array(label[:train_size])
        test_label = np.array(label[train_size:])

        return train_data, train_label, test_data, test_label

    def MinMaxScalar(self, data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)

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