import tensorflow as tf
import numpy as np
import input_data
import Imagenet
import os
import shutil
import NNutils
from PIL import Image
from sklearn import svm
from datetime import datetime

dropout_conv = tf.placeholder("float")
dropout_fc = tf.placeholder("float")
global_step = tf.Variable(0, trainable=False)
cost_weight = tf.Variable(0, trainable=False)


batch_size = 128
layer = [32, 64, 128, 256, 375]
path = "ConvAENN/" + "FinalTest"
X = tf.placeholder("float", [None, 32, 32, 3], name='X')
Y = tf.placeholder("float", [None, 10])
dropout_value = [0.8, 0.8]

class Temp:
    list = []

def conv_layer(name, X, shape):
    kernel = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    bias = tf.Variable(tf.zeros([shape[3]]))
    conv = tf.nn.conv2d(X, kernel, strides=[1, 1, 1, 1], padding='SAME')
    Y = tf.nn.relu(tf.nn.bias_add(conv, bias))
    return Y

def deconv_layer(name, X, shape, output_shape):
    kernel = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv = tf.nn.conv2d_transpose(X, kernel, output_shape=output_shape, strides=[1, 2, 2, 1],padding='SAME')
    bias = tf.Variable(tf.zeros([shape[2]]))  #transpose가 되기때문에 [kernel, kernel, output, input]라서 [2]를 취한다.
    Y = tf.nn.relu(tf.nn.bias_add(conv, bias))
    return Y

def noising(X, rate):
    rate = 255 * 0.5 * rate
    mask = tf.random_uniform([batch_size, 32, 32, 3], rate, rate)
    X_noise = X + mask
    X_noise = tf.clip_by_value(X_noise, 0, 255)
    return X_noise

def model(X):                   #32 32 3             #in        out                                                              #28 28
    # weight1 = [5, 5, 3, 32]     #16 16 32
    # bias1 = [32]
    #
    # weight2 = [5, 5, 32, 64]    #8  8  64
    # bias2 = [64]
    #
    # reshape = [64 * 7 * 7, 1000]
    # bias_reshape = [1000]
    #
    # weight3 = [64 * 8 * 8, 300]
    # bias3 = [300]
    #                                                                     #32 32
    #encoder

    #X_noised = noising(X, 0.2)



    with tf.variable_scope('en_conv1'):
        output = conv_layer('en_w1', X, [5, 5, 3, layer[0]])                  #16 16  14 14

    with tf.name_scope("maxpool1"):
        pool = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')          #(16, 16, 64)
        output = tf.nn.dropout(pool, dropout_conv)



    with tf.variable_scope('en_conv2'):
        output = conv_layer('en_w2', output, [4, 4, layer[0], layer[1]])            #8  8   7   7

    with tf.name_scope("maxpool2"):
        pool = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')          #(16, 16, 64)
        output = tf.nn.dropout(pool, dropout_conv)

    with tf.variable_scope('en_conv3'):
        output = conv_layer('en_w3', output, [3, 3, layer[1], layer[2]])           #4  4   4   4

    with tf.name_scope("maxpool3"):
        pool = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')          #(16, 16, 64)
        output = tf.nn.dropout(pool, dropout_conv)

    with tf.variable_scope('en_conv4'):
        output = conv_layer('en_w4', output, [2, 2, layer[2], layer[3]])           #2  2   2   2

    with tf.name_scope("maxpool4"):
        pool = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (16, 16, 64)
        output = tf.nn.dropout(pool, dropout_conv)
    # with tf.variable_scope('en_reshape'):
    #     output = tf.reshape(output, [-1, 4 * 4 * 256])

    # with tf.variable_scope('en_fc'):
    #     weight = tf.Variable(tf.random_normal([4 * 4 * 256, 2000], stddev=0.1))
    #     bias = tf.Variable(tf.zeros([2000]))
    #     output = tf.nn.relu(tf.matmul(output, weight) + bias)
    #     #output = tf.nn.dropout(output, dropout_fc)
    #
    # with tf.variable_scope('en_fc2'):
    #     weight = tf.Variable(tf.random_normal([2000, 375], stddev=0.1))
    #     bias =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 loss_nn, loss_ae, \
                learning_rate, step = sess.run([merged, trainop,
                                                cost_NN, cost_AE,
                                                lr, global_step],
                                               feed_dict={ X: trX[start:end], Y: trY[start:end],
                                                           dropout_conv : dropout_value[0], dropout_fc : dropout_value[1]})
                if step % 50 == 0:
                    writer.add_summary(summary, step)
                    print(step, datetime.now(), loss_nn, loss_ae, learning_rate)


            summary, loss_nn, loss_ae, accuracy, results = sess.run([merged, cost_NN, cost_AE, acc_op, Y_],
                                                  feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                                  dropout_conv : 1.0, dropout_fc : 1.0})
            writer_test.add_summary(summary, step)
            print("test results : ", accuracy, loss_nn, loss_ae)
            saver.save(sess, path + "/model.ckpt", step)



            # im = im.astype('uint8')
            # im = Image.fromarray(im[0])
            # im.save('convAENN.jpg')

        end_time = datetime.now()
        print("걸린 시간 = ", end_time - st_time)

        print("retrain")
        #Retraining
        for i in range(0):
            print(i, st_time)
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                summary, _, loss_re, learning_rate, step = sess.run([merged, retrain_op, cost_retrain, lr, global_step],
                                                                          feed_dict={ X: trX[start:end], Y: trY[start:end],
                                                                                      dropout_conv : 0.8, dropout_fc : 0.5})
                if step % 50 == 0:
                    writer.add_summary(summary, step)
                    print(step, datetime.now(), loss_re, learning_rate)



            loss_nn, loss_ae, accuracy = sess.run([cost_NN, cost_AE, acc_op],
                                                  feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                                  dropout_conv : 1.0, dropout_fc : 1.0})

            print("test results : ", accuracy, loss_nn, loss_ae)


        # for i in range(epochs):
        #     for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        #         sess.run([trainop_b], feed_dict={X: trX[start:end], Y: trY[start:end],
        #                                          dropout_conv : 0.8, dropout_fc : 0.5})
        #
        #     image, test_loss, loss, accuracy = sess.run([X_, cost, cost_NN, acc_op],
        #                                                feed_dict={X: teX[test_indices], Y: teY[test_indices],
        #                                                           dropout_conv : 1.0, dropout_fc : 1.0})
        #     print("test results : ", accuracy, test_loss, loss)



        # test_loss, accuracy = sess.run([cost, acc_op], feed_dict={X: teX[test_indices], Y: teY[test_indices],
        #                                                           dropout_conv : 1.0, dropout_fc : 1.0})
        # print("test results : ", accuracy, test_loss)


run(200000)