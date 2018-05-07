import tensorflow as tf
import numpy as np
import input_data
import Imagenet
import os
import shutil
import NNutils
from PIL import Image

from datetime import datetime
from sklearn import svm

batch_size = 128
lr = 0.001

def conv_layer(name, X, shape):
    kernel = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    bias = tf.Variable(tf.zeros([shape[3]]))
    conv = tf.nn.conv2d(X, kernel, strides=[1, 2, 2, 1], padding='SAME')
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

    #X_noise = noising(X, 0.2)

    with tf.variable_scope('en_conv1'):
        output = conv_layer('en_w1', X, [5, 5, 3, 32])                  #16 16  14 14

    with tf.variable_scope('en_conv2'):
        output = conv_layer('en_w2', output, [4, 4, 32, 64])            #8  8   7   7

    with tf.variable_scope('en_conv3'):
        output = conv_layer('en_w3', output, [3, 3, 64, 128])           #4  4   4   4

    with tf.variable_scope('en_conv4'):
        output = conv_layer('en_w4', output, [2, 2, 128, 256])           #2  2   2   2

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
    #     bias = tf.Variable(tf.zeros([375]))
    #     output = tf.nn.relu(tf.matmul(output, weight) + bias) #Z

        Z = output

    #decoder
    # with tf.variable_scope('de_fc1'):
    #     weight = tf.transpose(tf.Variable(tf.random_normal([2000, 375], stddev=0.1)))
    #     bias = tf.Variable(tf.zeros([2000]))
    #     output = tf.nn.relu(tf.matmul(output, weight) + bias)
    #     #output = tf.nn.dropout(output, dropout_fc)
    #
    # with tf.variable_scope('de_fc2'):
    #     weight = tf.transpose(tf.Variable(tf.random_normal([4 * 4 * 256, 2000], stddev=0.1)))
    #     bias = tf.Variable(tf.zeros([4 * 4 * 256]))
    #     output = tf.nn.relu(tf.matmul(output, weight) + bias)
    #     #output = tf.nn.dropout(output, dropout_fc)

    # with tf.variable_scope('de_reshape'):
    #     output = tf.reshape(output, [-1, 4, 4, 256])
        #print(l6_output)



    with tf.variable_scope('de_conv1'):
        #output = deconv_layer('de_w1', output, [2, 2, 128, 256], [batch_size, 4, 4, 128])
        output = tf.layers.conv2d_transpose(output, 128, [2, 2], (2, 2), padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            bias_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('de_conv2'):
        #output = deconv_layer('de_w2', output, [3, 3, 64, 128], [batch_size, 8, 8, 64])
        output = tf.layers.conv2d_transpose(output, 64, [3, 3], (2, 2), padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            bias_initializer=tf.contrib.layers.xavier_initializer())

    with tf.variable_scope('de_conv3'):
        #output = deconv_layer('de_w3', output, [4, 4, 32, 64], [batch_size, 16, 16, 32])  #내부에 transpose가 있다. => [5, 5, 64, 32]
        output = tf.layers.conv2d_transpose(output, 32, [4, 4], (2, 2), padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            bias_initializer=tf.contrib.layers.xavier_initializer())

    with tf.variable_scope('de_conv4'):
        output = tf.layers.conv2d_transpose(output, 3, [5, 5], (2, 2), padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            bias_initializer=tf.contrib.layers.xavier_initializer())
        #output = deconv_layer('de_w4', output, [5, 5, 3, 32], [batch_size, 32, 32, 3])
        X = output

    return X, Z

X = tf.placeholder("float", [None, 32, 32, 3], name='X')
# Y = tf.placeholder("float", [None, 10])
dropout_conv = tf.placeholder("float")
dropout_fc = tf.placeholder("float")
global_step = tf.Variable(0, trainable=False)

X_, Z = model(X)
with tf.name_scope("cost"):
    lr = tf.train.exponential_decay(0.001, global_step, (50000 / batch_size) * 10, 0.9, staircase=True)  # step은 batch마다 1씩 증가됨, 100, 0.96이므로 100단계에 96%로 줄어듦
    lr = tf.maximum(0.0001, lr)

    # cost_AE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= X_, labels= X))
    cost = tf.reduce_mean(tf.pow(X - X_, 2))   # (입력 - 네트워크 출력)^2
    #Xsoftmax = tf.nn.softmax(X)
    #Xsoftmax = tf.contrib.layers.softmax(X)
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=X_, labels=Xsoftmax))

    trainop = tf.train.AdamOptimizer(lr).minimize(cost, global_step=global_step)
    tf.summary.scalar("cost_AE", cost)

# with tf.name_scope("modelb"):
#     X_, Y_, Z = model_b(X)
#
#     cost_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_, labels=Y))
#     trainop_b = tf.train.AdamOptimizer(lr).minimize(cost_b, global_step=global_step)

def result(sess, teX, path):
    loss, arr = sess.run([cost, X_],
                         feed_dict={X: teX[0:batch_size],
                                    dropout_conv: 1.0, dropout_fc: 1.0})

    NNutils.create_image(arr, path)

def run(epochs):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        dataset = Imagenet.Cifar()
        trX, trY, teX, teY = dataset.getdata()


        print(teY.shape)

        filetime = datetime.now().strftime("%Y_%m_%d_%H_%M")
        path = "convAE/" + "rgb2rgb"
        #path = "Networkfile/convAE" + filetime
        saver = NNutils.save(path, sess)
        writer, writer_test, merged = NNutils.graph(path, sess)

        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:batch_size]

        st_time = datetime.now()

        for i in range(epochs):
            print(i, st_time)
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                summary, _,\
                loss, learning_rate,\
                step = sess.run([merged, trainop,
                                 cost, lr,
                                 global_step],
                                feed_dict={ X: trX[start:end],
                                            dropout_conv : 0.8, dropout_fc : 0.5})
                if step % 50 == 0:
                    writer.add_summary(summary, step)
                    print(step, datetime.now(), loss, learning_rate)


            loss, results = sess.run([cost, Z], feed_dict={X: teX,
                                                           dropout_conv: 1.0,
                                                           dropout_fc: 1.0,
                                                           })
            print("test results : ", loss)
            saver.save(sess, path + "/model.ckpt", step)


            #
            # image = image.astype('uint8')
            # im = Image.fromarray(image[0])
            # im.show()

            # image = teX.astype('uint8')
            # im = Image.fromarray(image[0])
            # im.show()

        end_time = datetime.now()
        print("걸린 시간 = ", end_time - st_time)

        #평가
        # SVM 학습
        loss, x_train = sess.run([cost, Z], feed_dict={X: trX,
                                                       dropout_conv: 1.0,
                                                       dropout_fc: 1.0,
                                                       })
        x_train = x_train.reshape(len(x_train), -1)
        y_train = np.argmax(trY, 1)


        print(x_train.shape)
        clf = svm.LinearSVC(max_iter=500, random_state=2)
        clf.fit(x_train, y_train)

        # SVM 예측 정확도 계산
        loss, x_test = sess.run([cost, Z], feed_dict={X: teX,
                                                       dropout_conv: 1.0,
                                                       dropout_fc: 1.0,
                                                       })
        x_test = x_test.reshape(len(x_test), -1)
        y_test = np.argmax(teY, 1)

        accuracy = 0
        iteration = 50
        for i in range(iteration):
            print(i)
            clf = svm.LinearSVC(max_iter=200)
            clf.fit(x_train, y_train)
            acc = clf.score(x_test, y_test)
            print(acc)
            accuracy += acc

        accuracy /= iteration
        print(accuracy)

        # answer = clf.predict(x_test)
        # correct = 0
        # for i in range(len(answer)):
        #     if answer[i] == y_test[i]:
        #         correct += 1
        # accuracy = correct / len(answer)
        # print("정확도 ", accuracy)
        # print("ans :", answer)
        # print("y :", y_test[0:10])
        # for i in range(epochs):
        #     for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        #         sess.run([trainop_b], feed_dict={X: trX[start:end], Y: trY[start:end],
        #                                          dropout_conv : 0.8, dropout_fc : 0.5})
        #
        #     image, test_loss, loss, accuracy = sess.run([X_, cost, cost_NN, acc_op],
        #                                                feed_dict={X: teX[test_indices], Y: teY[test_indices],
        #                                                           dropout_conv : 1.0, dropout_fc : 1.0})
        #     print("test results : ", accuracy, test_loss, loss)

        #result(sess,teX, "Image_rgb2rgb")

        # test_loss, accuracy = sess.run([cost, acc_op], feed_dict={X: teX[test_indices], Y: teY[test_indices],
        #                                                           dropout_conv : 1.0, dropout_fc : 1.0})
        # print("test results : ", accuracy, test_loss)

run(0)