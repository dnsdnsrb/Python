import tensorflow as tf
import numpy as np
import mnist
import NNutils
import os
from datetime import datetime
from PIL import Image
from scipy import misc
from sklearn import svm

def save_image(im, path, step):
    path = path + "/image/"

    if not os.path.exists(path):
        os.makedirs(path)
    im = im.astype('uint8')
    im = im.reshape(-1, 21, 21)
    im = Image.fromarray(im[0])
    im.save(path + str(step) + ".jpg")

class Network():
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.x = tf.placeholder("float32", [None, 21 * 21 * 1])
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

    def model(self, x, layer = [21*21, 14*14, 7*7]):
        output = x

        layer_num = 0

        for i in range(2):
            layer_num += 1
            with tf.variable_scope('en' + str(layer_num)):
                output = tf.contrib.layers.fully_connected(output, layer[layer_num], activation_fn=tf.nn.relu)
                # output = tf.nn.dropout(output, self.dropout_conv)

        z = output

        for i in range(1):
            layer_num -= 1
            with tf.variable_scope('de' + str(layer_num)):
                output = tf.contrib.layers.fully_connected(output, layer[layer_num], activation_fn=tf.nn.relu)
                # output = tf.nn.dropout(output, self.dropout_conv)

        layer_num -= 1
        with tf.variable_scope('de' + str(layer_num)):
            output = tf.contrib.layers.fully_connected(output, layer[layer_num], activation_fn=None)

        x = output
        return x, z

    def model_fc(self, z, layer = [7*7, 6*6, 5*5, 10]):
        output = z

        layer_num = 0

        for i in range(2):
            layer_num += 1
            with tf.variable_scope('fc' + str(layer_num)):
                output = NNutils.create_layer('fc', output, layer[layer_num], var_list=self.train_list)
                # output = tf.nn.dropout(output, self.dropout_conv)

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

        self.im = x_
        self.z = z

        #cost and training
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.pow(self.x - x_, 2))
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

    def run(self, step_limit):
        self.train()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            dataset = mnist.read_data_sets("MNIST_data/", one_hot=True)
            train_data, train_label, test_data, test_label = dataset.train.images, dataset.train.labels, \
                                                             dataset.test.images, dataset.test.labels

            # 이 데이터셋은 기본적으로 0~1사이 값으로 정규화되어있어 이미지를 볼 수 없다. => 0~255로 변경
            train_data = train_data * 255
            test_data = test_data * 255

            # 21*21로 변형
            train_reduced = train_data.reshape(-1, 28, 28)
            test_reduced = test_data.reshape(-1, 28, 28)
            train_reduced = np.array([misc.imresize(train_reduced[i], (21, 21)) for i in range(len(train_data))])
            test_reduced = np.array([misc.imresize(test_reduced[i], (21, 21)) for i in range(len(test_data))])
            train_reduced = train_reduced.reshape(-1, 21 * 21)
            test_reduced = test_reduced.reshape(-1, 21 * 21)

            path = "AE_21/" + str(step_limit)
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
                                    feed_dict={self.x: train_reduced[start:end],
                                               self.y: train_label[start:end],
                                               self.dropout_conv: 0.8,
                                               self.dropout_normal: 0.5})

                    if step % 50 == 0:
                        writer.add_summary(summary, step)
                        print(step, datetime.now(), loss)

                summary, loss, loss_fc, im = sess.run([merged, self.cost, self.cost_fc, self.im],
                                                      feed_dict={self.x: test_reduced,
                                                                 self.y: test_label,
                                                                 self.dropout_conv: 1.0,
                                                                 self.dropout_normal: 1.0})

                writer_test.add_summary(summary, step)
                print("test results : ", loss, loss_fc)
                saver.save(sess, path + "/model.ckpt", step)
                # save_image(im, path, step)

            # #SVM 부분
            # z_train = sess.run(self.z, feed_dict={self.x: train_data,
            #                                   self.dropout_conv: 1.0,
            #                                   self.dropout_normal: 1.0})
            #
            # z_test = sess.run(self.z, feed_dict={self.x: test_data,
            #                                            self.dropout_conv: 1.0,
            #                                            self.dropout_normal: 1.0})
            #
            # accuracy = 0
            # iteration = 50
            # for i in range(iteration):
            #     print(i)
            #     clf = svm.LinearSVC(max_iter=100)
            #     clf.fit(z_train, train_label)
            #     acc = clf.score(z_test, test_label)
            #     print(acc)
            #     accuracy += acc
            # print(accuracy / iteration)


model = Network()
model.run(10000)