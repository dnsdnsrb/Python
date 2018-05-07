import tensorflow as tf
import numpy as np
import input_data
import Imagenet
import os
import shutil
import NNutils
from PIL import Image

from datetime import datetime

batch_size = 128
lr = 0.001

def conv_layer(name, X, shape, strides = [1, 2, 2, 1]):
    kernel = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    bias = tf.Variable(tf.zeros([shape[3]]))
    conv = tf.nn.conv2d(X, kernel, strides=strides, padding='SAME')
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

    X_noised = noising(X, 0.2)

    with tf.variable_scope('en_conv1'):
        output = conv_layer('en_w1', X_noised, [5, 5, 3, 32])                  #16 16  14 14

    with tf.variable_scope('en_conv2'):
        output = conv_layer('en_w2', output, [4, 4, 32, 64])            #8  8   7   7

        Z = output

    with tf.variable_scope('de_conv3'):
        output = deconv_layer('de_w3', output, [4, 4, 32, 64], [batch_size, 16, 16, 32])  #내부에 transpose가 있다. => [5, 5, 64, 32]


    with tf.variable_scope('de_conv4'):
        output = deconv_layer('de_w4', output, [5, 5, 3, 32], [batch_size, 32, 32, 3])
        X = output



    with tf.variable_scope('conv1'):
        output = conv_layer('w1', Z, [3, 3, 64, 128])
        output = tf.nn.dropout(output, dropout_fc)

    with tf.variable_scope('conv2'):
        output = conv_layer('w2', output, [2, 2, 128, 256])
        output = tf.nn.dropout(output, dropout_fc)

    with tf.variable_scope('reshape'):
        output = tf.reshape(output, [-1, 2 * 2 * 256])
        output = tf.nn.dropout(output, dropout_conv)

    with tf.variable_scope('Supervise1'):
        weight = (tf.Variable(tf.random_normal([2 * 2 * 256, 500], stddev=0.1)))
        bias = tf.Variable(tf.zeros([500]))
        output = tf.matmul(output, weight) + bias
        Y = output

    with tf.variable_scope('Supervise2'):
        weight = (tf.Variable(tf.random_normal([500, 10], stddev=0.1)))
        bias = tf.Variable(tf.zeros([10]))
        output = tf.matmul(output, weight) + bias
        Y = output

    return X, Y, Z

def model_b(X):
    with tf.variable_scope('Supervise1'):
        weight = tf.Variable(tf.random_normal([2 * 2 * 256, 375], stddev=0.1))
        bias = tf.Variable(tf.zeros([375]))
        output = tf.nn.relu(tf.matmul(X, weight) + bias) #Z
        output = tf.nn.dropout(output, dropout_fc)

    with tf.variable_scope('Supervise2'):
        weight = (tf.Variable(tf.random_normal([375, 10], stddev=0.1)))
        bias = tf.Variable(tf.zeros([10]))
        output = tf.matmul(output, weight) + bias
        Y = output

        return Y

X = tf.placeholder("float", [None, 32, 32, 3], name='X')
Y = tf.placeholder("float", [None, 10])
dropout_conv = tf.placeholder("float")
dropout_fc = tf.placeholder("float")
global_step = tf.Variable(0, trainable=False)
cost_weight = tf.Variable(0, trainable=False)

X_, Y_ , Z = model(X)

with tf.name_scope("cost"):
    lr = tf.train.exponential_decay(0.001, global_step, (50000 / batch_size) * 10, 0.95, staircase=True)  # step은 batch마다 1씩 증가됨, 100, 0.96이므로 100단계에 96%로 줄어듦
    lr = tf.maximum(0.0001, lr)

    Xsoftmax = tf.contrib.layers.softmax(X)
    cost_AE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=X_, labels=Xsoftmax))
    cost_NN = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= Y_, labels= Y))
    cost = cost_NN + cost_AE

    trainop = tf.train.AdamOptimizer(lr).minimize(cost, global_step=global_step)
    tf.summary.scalar("cost_AE", cost_AE)
    tf.summary.scalar("cost_NN", cost_NN)

with tf.name_scope("accuracy"):
    predict_op = tf.equal(tf.arg_max(Y, 1), tf.arg_max(Y_, 1))
    acc_op = tf.reduce_mean(tf.cast(predict_op, "float"))
    tf.summary.scalar("accuarcy", acc_op)

def run(epochs):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        dataset = Imagenet.Cifar()
        trX, trY, teX, teY = dataset.getdata()

        filetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        #path = "Networkfile/" + "convAENN_noise"
        path = "Networkfile/convAENN" + filetime
        saver = NNutils.save(path, sess)
        writer, merged = NNutils.graph(path, sess)

        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:(batch_size)]

        st_time = datetime.now()

        for i in range(epochs):
            print(i, st_time)
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                summary, _, loss_nn, loss_ae, learning_rate, step = sess.run([merged, trainop, cost_NN, cost_AE, lr, global_step],
                                                                          feed_dict={ X: trX[start:end], Y: trY[start:end],
                                                                                      dropout_conv : 0.5, dropout_fc : 0.5})
                if step % 50 == 0:
                    writer.add_summary(summary, step)
                    print(step, datetime.now(), loss_nn, loss_ae, learning_rate)


            loss_nn, loss_ae, accuracy = sess.run([cost_NN, cost_AE, acc_op],
                                                  feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                                  dropout_conv : 1.0, dropout_fc : 1.0})
            print("test results : ", accuracy, loss_nn, loss_ae)
            saver.save(sess, path + "/model.ckpt", step)

            # im = im.astype('uint8')
            # im = Image.fromarray(im[0])
            # im.save('convAENN.jpg')

        end_time = datetime.now()
        print("걸린 시간 = ", end_time - st_time)

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

run(800)