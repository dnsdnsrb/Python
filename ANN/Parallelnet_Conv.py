import tensorflow as tf
import numpy as np
import cifar
import NNutils

from datetime import datetime

def conv_layer(x, output_num, kernel, stride=1, name='var', activation='none', dropout=None, pow=1):
    kernel = tf.get_variable(name, [kernel, kernel, x.get_shape().as_list()[3], output_num])
    bias = tf.Variable(tf.zeros(output_num))

    kernel = tf.pow(kernel, pow)
    y = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding='SAME')
    y = tf.nn.bias_add(y, bias)

    if activation == 'relu':
        y = tf.nn.relu(y)
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(y)
    elif activation == 'none':
        pass

    if dropout is not None:
        y = tf.nn.dropout(y, dropout)

    return y

def fc_layer(x, output_num, pow = 1, activation='none', dropout=None, name='var'):
    weight = tf.get_variable(name, [x.get_shape().as_list()[1], output_num],
                             initializer=tf.random_normal_initializer(stddev=0.01))
    bias = tf.Variable(tf.zeros(output_num))

    weight = tf.pow(weight, pow)
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

class Network():
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.x = tf.placeholder("float32", [None, 32, 32, 3])
        self.y = tf.placeholder("float32", [None, 10])

        self.dropout_conv = tf.placeholder("float")
        self.dropout_normal = tf.placeholder("float")

        self.train_list = []

    def model(self, x, layer=32):
        image_size = 32
        output = x
        reshape_size = 0
        #w1 + w2 + w3 / (w4 + w5 + w6) => 이런 형태를 구현해볼 것
        layer_num = 0

        layer_num += 1
        with tf.variable_scope('conv' + str(layer_num)):
            output1 = conv_layer(output, layer, 5, 1, pow=1, activation='sigmoid', dropout=self.dropout_normal)
            output1 = tf.layers.max_pooling2d(output1, [2, 2], [2, 2], padding='SAME')

        layer_num += 1
        with tf.variable_scope('conv' + str(layer_num)):
            output2 = conv_layer(output, layer, 5, 1, pow=2, activation='sigmoid', dropout=self.dropout_normal)
            output2 = tf.layers.max_pooling2d(output2, [2, 2], [2, 2], padding='SAME')

        layer_num += 1
        with tf.variable_scope('conv' + str(layer_num)):
            output3 = conv_layer(output, layer, 5, 1, pow=3, activation='sigmoid', dropout=self.dropout_normal)
            output3 = tf.layers.max_pooling2d(output3, [2, 2], [2, 2], padding='SAME')

        layer_num += 1
        with tf.variable_scope('conv' + str(layer_num)):
            output4 = conv_layer(output, layer, 5, 1, pow=4, activation='sigmoid', dropout=self.dropout_normal)
            output4 = tf.layers.max_pooling2d(output4, [2, 2], [2, 2], padding='SAME')

        with tf.variable_scope('sum' + str(layer_num)):
            output = output1 + output2 + output3 + output4

        with tf.name_scope('reshape'):
            output = tf.reshape(output, [-1, 16 * 16 * layer])

        with tf.variable_scope('fc1'):
            output = tf.contrib.layers.fully_connected(output, 375, activation_fn=None)

        with tf.variable_scope('fc2'):
            output = tf.contrib.layers.fully_connected(output, 10, activation_fn=None)

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

            dataset = cifar.Cifar()
            train_data, train_label, test_data, test_label = dataset.getdata()

            test_indices = np.arange(len(test_data))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:1000]

            path = "PNconv/" + str(step_limit) +"cifar"
            saver = NNutils.save(path, sess)
            writer, writer_test, merged = NNutils.graph(path, sess)

            step = sess.run(self.global_step)
            while step < step_limit:
                print("step :", step)
                for start, end in zip(range(0, len(train_data), self.batch_size),
                                      range(self.batch_size, len(train_data), self.batch_size)):
                    summary, \
                    _, loss, \
                    step = sess.run([merged,
                                     self.training, self.cost,
                                     self.global_step],
                                    feed_dict={self.x: train_data[start:end],
                                               self.y: train_label[start:end],
                                               self.dropout_conv: 1.0,
                                               self.dropout_normal: 1.0})

                    if step % 50 == 0:
                        writer.add_summary(summary, step)
                        print(step, datetime.now(), loss)

                summary, \
                loss, \
                accuracy = sess.run([merged, self.cost, self.accuracy],
                                    feed_dict={self.x: test_data[test_indices],
                                               self.y: test_label[test_indices],
                                               self.dropout_conv: 1.0,
                                               self.dropout_normal: 1.0})

                writer_test.add_summary(summary, step)
                print("test results : ", accuracy, loss)
                saver.save(sess, path + "/model.ckpt", step)

model = Network()
model.run(100000)