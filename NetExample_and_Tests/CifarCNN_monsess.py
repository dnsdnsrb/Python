import tensorflow as tf
import numpy as np
import input_data
import os
import shutil
import Imagenet
import random
import PIL
from PIL import Image
from datetime import datetime
import DataSet
import NNutils

#import DataSet

#batch_size = 128
#test_size = batch_size

class Network():

    def __init__(self):
        self.batch_size = 128

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def conv_layer(self, input, kernel_shape):
        kernel = self.init_weights(kernel_shape)
        conv = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME')  # (32, 32, 64)
        bias = self.init_weights([kernel.get_shape().as_list()[3]])
        output = tf.nn.relu(tf.nn.bias_add(conv, bias))
        return output

    def model(self, X ,p_keep_conv, p_keep_hidden):

        with tf.name_scope("conv1"):
            output = self.conv_layer(X, [5, 5, 3, 32])

        with tf.name_scope("maxpool1"):
            pool = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')          #(16, 16, 64)
            pool = tf.nn.dropout(pool, p_keep_conv)

        with tf.name_scope("conv2"):
            output = self.conv_layer(pool, [5, 5, 32, 64])

        with tf.name_scope("maxpool2"):
            pool = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')          #(8, 8, 128)
            pool = tf.nn.dropout(pool, p_keep_conv)

        with tf.name_scope("conv3"):
            output = self.conv_layer(pool, [5, 5, 64, 256])

        with tf.name_scope("maxpool3"):
            pool = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')          #(4, 4, 256)
            pool = tf.nn.dropout(pool, p_keep_conv)

        with tf.name_scope("reshape"):
            pool = tf.reshape(pool, [-1, 4 * 4 * 256])

        with tf.name_scope("fc4"):
            weight = self.init_weights([4 * 4 * 256, 2000])
            bias = self.init_weights([weight.get_shape().as_list()[1]])
            output = tf.nn.relu(tf.matmul(pool, weight) + bias)
            output = tf.nn.dropout(output, p_keep_hidden)

        with tf.name_scope("fc5"):
            weight = self.init_weights([2000, 375])
            bias = self.init_weights([weight.get_shape().as_list()[1]])
            output = tf.nn.relu(tf.matmul(output, weight) + bias)

        with tf.name_scope("fc6"):
            weight = self.init_weights([375, 10])
            bias = self.init_weights([weight.get_shape().as_list()[1]])
            output = tf.matmul(output, weight) + bias
            Y = output

        return Y

    def Train(self):
        self.p_keep_conv = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.
        limit = 0.0001
        py_x = self.model(self.trX, self.p_keep_conv, self.p_keep_hidden)

        with tf.name_scope("cost"):
            self.lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, self.global_step, (50000 / self.batch_size) * 3, 0.75, staircase=True) #step은 batch마다 1씩 증가됨, 100, 0.96이므로 100단계에 96%로 줄어듦
            self.lr = tf.maximum(limit, self.lr)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= py_x, labels= self.trY))
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost, global_step=self.global_step)  #학습이 훌륭하게 되진 않는다. learning rate를 adaptive하게 하지 않아서 그런가? 확실히 learning rate를 낮추니 효과가 있었음.
            tf.summary.scalar("cost", self.cost)

    def Test(self):

        Y = self.model(self.teX, self.p_keep_conv, self.p_keep_hidden)
        self.r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=self.trY))


        with tf.name_scope("accuracy"): #accuracy는 tf를 이용하는게 좋다. tensorboard로 볼 수 있기 때문
            self.predict_op = tf.equal(tf.argmax(self.trY, 1), tf.argmax(Y, 1))
            self.acc_op = tf.reduce_mean(tf.cast(self.predict_op, "float"))
            tf.summary.scalar("accuracy", self.acc_op)

    def graph(self, path, sess):
        graphPath = path
        if not os.path.exists(graphPath):
            os.makedirs(graphPath)
        shutil.rmtree(graphPath)
        self.writer = tf.summary.FileWriter(graphPath, sess.graph)
        self.merge = tf.summary.merge_all()

    def randomSample(self, data, number):
        test_indices = np.arange(len(data))
        np.random.shuffle(test_indices)
        return test_indices[0:number]

    def save(self, path, sess):
        ckpt_dir = path
        self.saver = tf.train.Saver()
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def run(self, epochs):
        #CIFAR


        dataset = DataSet.Cifar()
        self.trX, self.trY, self.teX, self.teY = dataset.create_sets()
        epochs = 50000 * epochs

        self.Train()
        self.Test()

        with tf.train.MonitoredTrainingSession(checkpoint_dir="Networkfile/tfconvtest",  # 세이브, 그래프, 초기화 모두 다 포함되있음.
                                               hooks=[tf.train.StopAtStepHook(last_step=epochs)],
                                               save_checkpoint_secs=300,
                                               save_summaries_steps=50) as sess:

            while not (sess.should_stop()):
                _, loss, step = sess.run([self.train_op, self.cost, self.global_step], feed_dict={self.p_keep_conv:0.8, self.p_keep_hidden:0.5})

                if(step % 1000 == 0):
                    accuracy = sess.run(self.acc_op, feed_dict={self.p_keep_conv:1.0, self.p_keep_hidden:1.0})
                    print(accuracy)




Net = Network()
Net.run(5)    #800번 10000번, 10시간 걸림