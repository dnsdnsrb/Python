import tensorflow as tf
import numpy as np
import input_data
import Imagenet
import os
import shutil
import NNutils

from tensorflow.core.framework import graph_pb2
from datetime import datetime

class ConvAE():
    def __init__(self):
        self.batch_size = 128
        self.x = tf.placeholder("float", [None, 32, 32, 3])  # [None, 32, 32, 3]
        self.y = tf.placeholder("float", [None, 10])  # [None, 10]
        self.x_nn = tf.placeholder("float", [None, 2, 2, 256])
        self.z = tf.Variable(0, trainable=False)
        self.dropout_conv = tf.placeholder("float")
        self.dropout_fc = tf.placeholder("float")
        self.global_step = tf.Variable(0, trainable=False)

    def conv_layer(self, name, X, shape):
        kernel = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        #kernel = tf.Variable(tf.random_normal(shape, stddev=0.01))
        bias = tf.Variable(tf.zeros([shape[3]]))
        conv = tf.nn.conv2d(X, kernel, strides=[1, 2, 2, 1], padding='SAME')
        Y = tf.nn.relu(tf.nn.bias_add(conv, bias))
        return Y

    def deconv_layer(self, name, X, shape, output_shape):
        kernel = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        #kernel = tf.Variable(tf.random_normal(shape, stddev=0.01))
        conv = tf.nn.conv2d_transpose(X, kernel, output_shape=output_shape, strides=[1, 2, 2, 1],padding='SAME')
        bias = tf.Variable(tf.zeros([shape[2]]))  #transpose가 되기때문에 [kernel, kernel, output, input]라서 [2]를 취한다.
        Y = tf.nn.relu(tf.nn.bias_add(conv, bias))
        return Y

    def noising(self, X, rate):
        rate = 255 * 0.5 * rate
        #size = X.get_shape().as_list()[0]
        mask = tf.random_uniform([self.batch_size, 32, 32, 3], rate, rate)
        X_noise = X + mask
        X_noise = tf.clip_by_value(X_noise, 0, 255)
        return X_noise

    def unsuper(self, X):                   #32 32 3             #in        out                                                              #28 28

        X_noise = self.noising(X, 0.2)

        with tf.variable_scope('en_conv1'):
            output = self.conv_layer('en_w1', X_noise, [5, 5, 3, 32])                  #16 16  14 14

        with tf.variable_scope('en_conv2'):
            output = self.conv_layer('en_w2', output, [4, 4, 32, 64])            #8  8   7   7

        with tf.variable_scope('en_conv3'):
            output = self.conv_layer('en_w3', output, [3, 3, 64, 128])           #4  4   4   4

        with tf.variable_scope('en_conv4'):
            output = self.conv_layer('en_w4', output, [2, 2, 128, 256])           #2  2   2   2

            Z = output

        with tf.variable_scope('de_conv1'):
            output = self.deconv_layer('de_w1', output, [2, 2, 128, 256], [self.batch_size, 4, 4, 128])

        with tf.variable_scope('de_conv2'):
            output = self.deconv_layer('de_w2', output, [3, 3, 64, 128], [self.batch_size, 8, 8, 64])  #내부에 transpose가 있다. => [5, 5, 64, 32]

        with tf.variable_scope('de_conv3'):
            output = self.deconv_layer('de_w3', output, [4, 4, 32, 64], [self.batch_size, 16, 16, 32])  #내부에 transpose가 있다. => [5, 5, 64, 32]

        with tf.variable_scope('de_conv4'):
            output = self.deconv_layer('de_w4', output, [5, 5, 3, 32], [self.batch_size, 32, 32, 3])
            X = output

        return X, Z


    def super(self, Z):
        with tf.variable_scope('reshape'):
            output = tf.reshape(Z, [-1, 2 * 2 * 256])

        with tf.variable_scope('fc1'):
            weight = tf.Variable(tf.random_normal([2 * 2 * 256, 700], stddev=0.1))
            bias = tf.Variable(tf.zeros([700]))
            output = tf.nn.relu(tf.matmul(output, weight) + bias) #Z
            output = tf.nn.dropout(output, self.dropout_fc)

        with tf.variable_scope('fc1'):
            weight = tf.Variable(tf.random_normal([700, 350], stddev=0.1))
            bias = tf.Variable(tf.zeros([350]))
            output = tf.nn.relu(tf.matmul(output, weight) + bias) #Z
            output = tf.nn.dropout(output, self.dropout_fc)

        with tf.variable_scope('fc2'):
            weight = (tf.Variable(tf.random_normal([350, 10], stddev=0.1)))
            bias = tf.Variable(tf.zeros([10]))
            output = tf.matmul(output, weight) + bias
            Y = output

        return Y

    def train_unsuper(self):
        with tf.name_scope("cost"):
            X_, self.z = self.unsuper(self.x)

            self.lr = tf.train.exponential_decay(0.001, self.global_step, (50000 / self.batch_size) * 10, 0.9, staircase=True)  # step은 batch마다 1씩 증가됨, 100, 0.96이므로 100단계에 96%로 줄어듦
            self.lr = tf.maximum(0.0001, self.lr)

            #cost_unsuper = tf.reduce_mean(tf.pow(X - X_, 2))   # (입력 - 네트워크 출력)^2
            Xsoftmax = tf.contrib.layers.softmax(self.x)
            self.cost_unsuper = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=X_, labels=Xsoftmax))
            self.trainop_unsuper = tf.train.AdamOptimizer(self.lr).minimize(self.cost_unsuper, global_step= self.global_step)

            tf.summary.scalar("cost_unsu", self.cost_unsuper)

    def train_super(self):
        with tf.name_scope("super"):
            Y_ = self.super(self.z)

            self.lr = tf.train.exponential_decay(0.001, self.global_step, (50000 / self.batch_size) * 10, 0.9, staircase=True)  # step은 batch마다 1씩 증가됨, 100, 0.96이므로 100단계에 96%로 줄어듦
            self.lr = tf.maximum(0.0001, self.lr)

            self.cost_super = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= Y_, labels= self.y))
            self.trainop_super = tf.train.AdamOptimizer(self.lr).minimize(self.cost_super, global_step= self.global_step)

            tf.summary.scalar("cost_su", self.cost_super)

        with tf.name_scope("accuracy"):
            predict_op = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(Y_, 1))
            acc_op = tf.reduce_mean(tf.cast(predict_op, "float"))

            tf.summary.scalar("accuarcy", acc_op)

    def run(self, epochs):
        self.train_unsuper()
        #self.train_super()
        with tf.Session() as sess:

            tf.global_variables_initializer().run()
            dataset = Imagenet.Cifar()
            trX, trY, teX, teY = dataset.getdata()

            filetime = datetime.now().strftime("%Y_%m_%d_%H_%M")
            path = "Networkfile/convAE_sep_class"
            #path = "Networkfile/convAENN_sep" + filetime
            saver = NNutils.save(path, sess)
            writer, merged = NNutils.graph(path, sess)

            test_indices = np.arange(len(teX))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:(self.batch_size)]

            st_time = datetime.now()
            for i in range(epochs):
                print(i, st_time)
                for start, end in zip(range(0, len(trX), self.batch_size), range(self.batch_size, len(trX), self.batch_size)):
                    _ = sess.run(self.trainop_unsuper,
                                             feed_dict={self.x: trX[start:end], self.dropout_conv: 0.8, self.dropout_fc : 0.6})

                    # graph_def = graph_pb2.GraphDef()
                    # output_names = ""
                    # tf.graph_util.convert_variables_to_constants(sess, graph_def, output_names)
                    #print(cost_ae)
                output = sess.run(self.z, feed_dict={self.x: trX[start:end], self.dropout_conv: 0.8, self.dropout_fc : 0.6})
                z = tf.constant(output)
                print(z.shape)
                    # cost_nn, _ = sess.run([self.cost_super, self.trainop_super],
                    #                             feed_dict={self.x_nn: self.z, self.y: trY[start:end],
                    #                                        self.dropout_conv: 0.8, self.dropout_fc : 0.6})
                    #


                    #_, loss_super, step = sess.run([trainop_super, cost_super, global_step], feed_dict={x:trX[0:0], y: trY[start:end]
                    #                                                                  ,dropout_conv: 0.8, dropout_fc : 0.6})

                    # if step % 50 == 0:
                    #     writer.add_summary(summary, step)
                    #     print(step, loss_super, loss_unsuper)

                    #print(np.shape(trX))
                    #summary, accuracy, loss = sess.run([merged, acc_op, cost], feed_dict={ X: teX[test_indices], Y: teY[test_indices]})
                    #print(step, datetime.now(), loss_unsuper, loss_super, learning_rate)

                # loss_su, loss_un, accuracy, step = sess.run([cost_super, cost_unsuper, acc_op, global_step], feed_dict={X: teX[test_indices], Y: teY[test_indices],
                #                                                           dropout_conv : 1.0, dropout_fc : 1.0})
                # print("test results : ", accuracy, loss_super, loss_unsuper)
                # saver.save(sess, path + "/model.ckpt", step)

            end_time = datetime.now()
            print("걸린 시간 = ", end_time - st_time)

            # test_loss, accuracy = sess.run([cost_super, acc_op], feed_dict={x: teX[test_indices], y: teY[test_indices]})
            # print("test results : ", accuracy, test_loss)

network = ConvAE()
network.run(800)