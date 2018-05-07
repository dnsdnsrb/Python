import tensorflow as tf
import numpy as np
import DataSetNp
import NNutils
from datetime import datetime


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

    def model(self, x, conv = [6, 12, 24, 48], fc = [50, 1]):
        for layer in range(10):
            output = x
            layer_num = 0
            image_size = 32
            for i in conv:
                with tf.variable_scope('conv'+ str(layer) + str(i)):
                    output = tf.layers.conv2d(output, i, [5 - layer_num, 5 - layer_num], padding='SAME',
                                              activation=tf.nn.relu)
                    output = tf.layers.max_pooling2d(output, [2, 2], [2, 2], padding='SAME')
                    output = tf.nn.dropout(output, self.dropout_conv)
                    image_size /= 2
                    layer_num += 1

            with tf.name_scope('reshape'+ str(layer)):
                reshape_size = int(image_size * image_size * conv[len(conv) - 1])
                output = tf.reshape(output, [-1, reshape_size])

            # layer_num += 1
            with tf.variable_scope('fc'+ str(layer) + "0"):
                output = tf.contrib.layers.fully_connected(output, fc[0], activation_fn=tf.nn.relu)
                output = tf.nn.dropout(output, self.dropout_normal)

            # layer_num += 1
            with tf.variable_scope('fc'+ str(layer) + "1"):
                if layer == 0:
                    output_result1 = tf.layers.dense(output, fc[1], activation=None)
                else:
                    output_result2 = tf.layers.dense(output, fc[1], activation=None)
                    output_result1 = tf.concat([output_result1, output_result2], 1)

        y = output_result1
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

            dataset = DataSetNp.Cifar()
            train_data, train_label, test_data, test_label = dataset.getdata()

            test_indices = np.arange(len(test_data))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:1000]

            path = "Conv/" + str(step_limit) + "cifar"
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