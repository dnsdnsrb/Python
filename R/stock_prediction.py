import tensorflow as tf
import numpy as np

class Stock():
    def __init__(self):
        seq_length = 7

    def read_file1(self):
        file = tf.train.string_input_producer(["stock.csv"], shuffle=False)

        reader = tf.TextLineReader()
        key, value = reader.read(file)
        record_defaults = [[0.]]*5
        data = tf.decode_csv(value, record_defaults=record_defaults)

        x = []
        y = []
        for i in range(100):
            x.append(tf.train.batch([data], batch_size=7))  #time step이 7이여서 7개를 뽑아냄
            y.append(tf.train.batch([data[4]], batch_size=1))
        self.x_batch = tf.stack(x)
        self.y_batch = tf.stack(y)

        # self.y_batch = tf.train.batch([data[4:]], batch_size=1)

        # print(data)

    def read_file2(self):
        data =  np.loadtxt('../Data/data-02-stock_daily.csv', delimiter=',')
        x = data[::]
        y = data[::7,4:]

        x = tf.reshape(data, [-1, 7, 5])
        self.x_batch, self.y_batch = tf.train.batch([x[:,:4], x[:,4:]], batch_size=100, enqueue_many=True)


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
        dataset.read_file1()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(iterations):
                x, y = sess.run([dataset.x_batch, dataset.y_batch])
                print(x.shape)
                _, cost = sess.run([self.opt, self.loss], feed_dict={self.x: x,
                                                                     self.y: y})
                print(cost)

            coord.request_stop()
            coord.join(threads)




if __name__ == '__main__':
    dataset = Stock()
    dataset.read_file1()
    net = Network()
    net.run(500)