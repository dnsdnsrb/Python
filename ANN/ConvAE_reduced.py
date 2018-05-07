import tensorflow as tf
import numpy as np
import cifar
import NNutils
from scipy import misc
from datetime import datetime


class Network():
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.x = tf.placeholder("float32", [None, 32, 32, 3])
        self.x_reduced = tf.placeholder("float", [None, 16, 16, 3], name='x_reduced')
        self.y = tf.placeholder("float32", [None, 10])

        self.dropout_conv = tf.placeholder("float")
        self.dropout_normal = tf.placeholder("float")

        self.train_list = []

    # def model(self, layers, x): #layers = [[name, output_num] ... [name, output_num]], x = input
    #     output = x
    #     after_conv = False
    #     image_width = 32
    #     for layer_num in range(len(layers)):
    #         name = layers[layer_num][0] + str(layer_num)
    #         print(layers[layer_num][0], after_conv)
    #
    #         if layers[layer_num][0] == 'normal' and after_conv == True:
    #             for i_num in range(layer_num):
    #                 image_width = int(image_width / 2)
    #
    #             with tf.name_scope("reshape"):  #layer num은 증가시키지 않음 => 실제로 layers에 포함되지 않고 알아서 만들어준다.
    #                 output = tf.reshape(output, [-1, image_width * image_width * layers[layer_num][1]])
    #
    #         with tf.variable_scope(name):
    #             if not layer_num == len(layers):
    #                 output = NNutils.create_layer(layers[layer_num][0], output, layers[layer_num][1],
    #                                               kernel_shape=[4, 4])
    #             else:
    #                 output = NNutils.create_layer(layers[layer_num][0], output, layers[layer_num][1],
    #                                               kernel_shape=[4, 4], activation='none')
    #
    #
    #         if layers[layer_num][0] == 'conv':
    #             after_conv = True
    #
    #             maxpool = "maxpool" + str(layer_num)
    #
    #             with tf.name_scope(maxpool):
    #                 output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #         else:
    #             after_conv = False
    #
    #
    #
    #     y = output
    #     return y

    def model(self, x, layer = [3, 32, 64, 128, 256]):
        output = x

        layer_num = 0

        for i in range(4):
            layer_num += 1
            with tf.variable_scope('conv' + str(layer_num)):
                output = tf.layers.conv2d(output, layer[layer_num], [5 - i, 5 - i], padding='SAME',
                                          activation=tf.nn.relu)
                output = tf.layers.max_pooling2d(output, [2, 2], [2, 2], padding='SAME')

        z = output

        for i in range(2):
            layer_num -= 1
            with tf.variable_scope('deconv' + str(layer_num)):
                output = tf.layers.conv2d_transpose(output, layer[layer_num], [i + 2, i + 2], (2, 2),
                                                    padding='SAME',
                                                    activation=tf.nn.relu)

        layer_num -= 2
        with tf.variable_scope('deconv' + str(layer_num)):
            output = tf.layers.conv2d_transpose(output, layer[layer_num], [4, 4], (2, 2), padding='SAME')

        x = output

        return x, z

    def model_fc(self, z, layer = [2*2*256, 2*2*128, 2*2*64, 2*2*16, 10]):
        output = z

        layer_num = 0

        with tf.name_scope('reshape'):
            reshape_size = 2 * 2 * 256
            output = tf.reshape(output, [-1, reshape_size])

        for i in range(3):
            layer_num += 1
            with tf.variable_scope('fc' + str(layer_num)):
                output = NNutils.create_layer('fc', output, layer[layer_num], var_list=self.train_list,
                                              dropout=self.dropout_normal)

        layer_num += 1
        with tf.variable_scope('fc' + str(layer_num)):
            output = NNutils.create_layer('fc', output, layer[layer_num], var_list=self.train_list, activation='none')
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
        x_, z = self.model(self.x)
        y_ = self.model_fc(z)

        #cost and training
        with tf.name_scope("cost"):
            # x_softmax = tf.nn.softmax(self.x)
            # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_, labels=x_softmax))
            self.cost = tf.reduce_mean(tf.pow(self.x_reduced - x_, 2))
            self.training = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(self.cost, global_step=self.global_step)

            tf.summary.scalar("cost", self.cost)

        with tf.name_scope("cost_fc"):
            self.cost_fc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y))
            self.training_fc = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(self.cost_fc, var_list=self.train_list)

            tf.summary.scalar("cost_fc", self.cost_fc)

        with tf.name_scope("accuracy"):
            compare = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(compare, "float"))

            tf.summary.scalar("accuarcy", self.accuracy)

        # with tf.name_scope("accuracy"):
        #     compare = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(y_, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(compare, "float"))
        #
        #     tf.summary.scalar("accuarcy", self.accuracy)

    def run(self, step_limit):
        self.train()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            dataset = cifar.Cifar()
            train_data, train_label, test_data, test_label = dataset.getdata()

            train_reduced = np.array([misc.imresize(train_data[i], (16, 16, 3)) for i in range(len(train_data))])
            test_reduced = np.array([misc.imresize(test_data[i], (16, 16, 3)) for i in range(len(test_data))])

            test_indices = np.arange(len(test_data))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:1000]

            path = "ConvAE_reduced/"  + str(step_limit)
            saver = NNutils.save(path, sess)
            writer, writer_test, merged = NNutils.graph(path, sess)

            step = sess.run(self.global_step)
            while step < step_limit:
                print("step :", step)
                for start, end in zip(range(0, len(train_data), self.batch_size),
                                      range(self.batch_size, len(train_data), self.batch_size)):
                    summary, \
                    _, loss, \
                    _, \
                    step = sess.run([merged,
                                     self.training, self.cost,
                                     self.training_fc,
                                     self.global_step],
                                    feed_dict={self.x: train_data[start:end],
                                               self.x_reduced: train_reduced[start:end],
                                               self.y: train_label[start:end],
                                               self.dropout_conv: 0.8,
                                               self.dropout_normal: 0.9})

                    if step % 50 == 0:
                        writer.add_summary(summary, step)
                        print(step, datetime.now(), loss)

                summary, loss, loss_fc, acc = sess.run([merged, self.cost, self.cost_fc, self.accuracy],
                                                       feed_dict={self.x: test_data[test_indices],
                                                                  self.x_reduced: test_reduced[test_indices],
                                                                  self.y: test_label[test_indices],
                                                                  self.dropout_conv: 1.0,
                                                                  self.dropout_normal: 1.0})

                writer_test.add_summary(summary, step)
                print("test results : ", loss, loss_fc, acc)
                saver.save(sess, path + "/model.ckpt", step)

model = Network()
model.run(100000)