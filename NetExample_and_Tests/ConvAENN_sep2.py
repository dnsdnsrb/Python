import tensorflow as tf
import numpy as np
import input_data
import Imagenet
import os
import shutil
import NNutils

from datetime import datetime

batch_size = 128

class Temp:
    list = []

def conv_layer(name, X, shape, strides = [1, 2, 2, 1]):
    kernel = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    bias = tf.Variable(tf.zeros([shape[3]]))
    conv = tf.nn.conv2d(X, kernel, strides=strides, padding='SAME')
    Y = tf.nn.relu(tf.nn.bias_add(conv, bias))
    return Y, kernel, bias

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

def unsuper(X):

    X_noise = noising(X, 0.2)

    with tf.variable_scope('en_conv1'):
        output, _, _ = conv_layer('en_w1', X_noise, [5, 5, 3, 32])                  #16 16  14 14

    with tf.variable_scope('en_conv2'):
        output, _, _ = conv_layer('en_w2', output, [4, 4, 32, 64])            #8  8   7   7

        Z = output

    with tf.variable_scope('de_conv3'):
        output = deconv_layer('de_w3', output, [4, 4, 32, 64], [batch_size, 16, 16, 32])  #내부에 transpose가 있다. => [5, 5, 64, 32]

    with tf.variable_scope('de_conv4'):
        output = deconv_layer('de_w4', output, [5, 5, 3, 32], [batch_size, 32, 32, 3])
        # kernel = tf.get_variable('de_w3', shape=weight1, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # conv = tf.nn.conv2d_transpose(output, kernel, output_shape=[batch_size, 32, 32, 3], strides=[1, 2, 2, 1], padding='SAME')
        # bias = tf.Variable(tf.zeros([1]))
        # output = tf.nn.relu(tf.nn.bias_add(conv, bias))
        X = output

    return X, Z


def super(Z):

    with tf.variable_scope('en_conv3'):
        output, weight, bias = conv_layer('en_w3', Z, [3, 3, 64, 128])           #4  4   4   4
        output = tf.nn.dropout(output, dropout_conv)
        Temp.list.append(weight)
        Temp.list.append(bias)

    with tf.variable_scope('en_conv4'):
        output, weight, bias = conv_layer('en_w4', output, [2, 2, 128, 256])
        output = tf.nn.dropout(output, dropout_conv)
        Temp.list.append(weight)
        Temp.list.append(bias)

    with tf.variable_scope('reshape'):
        output = tf.reshape(output, [-1, 2 * 2 * 256])

    with tf.variable_scope('fc1'):
        weight = tf.Variable(tf.random_normal([2 * 2 * 256, 500], stddev=0.1))
        bias = tf.Variable(tf.zeros([500]))
        output = tf.nn.relu(tf.matmul(output, weight) + bias) #Z
        output = tf.nn.dropout(output, dropout_fc)
        Temp.list.append(weight)
        Temp.list.append(bias)

    with tf.variable_scope('fc2'):
        weight = tf.Variable(tf.random_normal([500, 10], stddev=0.1), name='fc2_weight')
        bias = tf.Variable(tf.zeros([10]), name='fc2_bias')
        output = tf.matmul(output, weight) + bias
        Y = output
        Temp.list.append(weight)
        Temp.list.append(bias)

    return Y


x = tf.placeholder("float", [None, 32, 32, 3], name='X') #[None, 32, 32, 3]
y = tf.placeholder("float", [None, 10])   #[None, 10]
dropout_conv = tf.placeholder("float")
dropout_fc = tf.placeholder("float")
global_step = tf.Variable(0, trainable=False)



with tf.name_scope("cost"):
    X_, Z = unsuper(x)

    lr = tf.train.exponential_decay(0.001, global_step, (50000 / batch_size) * 10, 0.9, staircase=True)  # step은 batch마다 1씩 증가됨, 100, 0.96이므로 100단계에 96%로 줄어듦
    lr = tf.maximum(0.0001, lr)

    #cost_unsuper = tf.reduce_mean(tf.pow(x - X_, 2))   # (입력 - 네트워크 출력)^2
    Xsoftmax = tf.contrib.layers.softmax(x)
    cost_unsuper = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=X_, labels=Xsoftmax))
    trainop_unsuper = tf.train.AdamOptimizer(lr).minimize(cost_unsuper)
    tf.summary.scalar("cost_unsu", cost_unsuper)

    Y_ = super(Z)

    #train_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc1/fc1_weight')

    cost_super = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= Y_, labels= y))
    trainop_super = tf.train.AdamOptimizer(lr).minimize(cost_super, global_step=global_step,
                                                        var_list=Temp.list)
    tf.summary.scalar("cost_su", cost_super)
    tf.summary.scalar("cost_unsu", cost_unsuper)

with tf.name_scope("accuracy"):
    predict_op = tf.equal(tf.arg_max(y, 1), tf.arg_max(Y_, 1))
    acc_op = tf.reduce_mean(tf.cast(predict_op, "float"))

    tf.summary.scalar("accuarcy", acc_op)

def run(epochs):
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        dataset = Imagenet.Cifar()
        trX, trY, teX, teY = dataset.getdata()
        filetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        #path = "Networkfile/convAE_sep" + "2017_03_27_20_32"
        path = "Networkfile/convAENN_sep" + filetime
        saver = NNutils.save(path, sess)
        writer, merged = NNutils.graph(path, sess)

        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:(batch_size)]

        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()

        st_time = datetime.now()
        for i in range(epochs):
            print(i, st_time)
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                summary, _, _, cost_ae, cost_nn, step = sess.run([merged, trainop_unsuper, trainop_super,
                                                                  cost_unsuper, cost_super, global_step],
                                                                 feed_dict={x: trX[start:end],
                                                                            y: trY[start:end],
                                                                            dropout_conv: 1.0,
                                                                            dropout_fc : 0.8})
                if step % 50 == 0:
                    writer.add_summary(summary, step)
                    print(step, datetime.now(), cost_nn, cost_ae)

            accuracy, cost_nn, cost_ae = sess.run([acc_op, cost_super, cost_unsuper],
                                                  feed_dict={x: teX[test_indices], y: teY[test_indices],
                                                             dropout_conv: 1.0, dropout_fc: 1.0})

            saver.save(sess, path + "/model.ckpt", step)
            print("test results : ", accuracy, cost_nn, cost_ae)


                # _, loss_super, step = sess.run([trainop_super, cost_super, global_step], feed_dict={y: trY[start:end]
                #                                                                  ,dropout_conv: 0.8, dropout_fc : 0.6})

        #tf.graph_util.convert_variables_to_constants(sess, )
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

run(800)