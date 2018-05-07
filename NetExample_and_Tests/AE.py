import tensorflow as tf
import numpy as np
import input_data
import Imagenet

imagenet = Imagenet.Imagenet()

mnist_width = 28    #글자 이미지 크기를 의미한다.
n_visible = mnist_width * mnist_width   #28*28 크기의 글자 이미지

input_size = imagenet.image_height * imagenet.image_width * imagenet.image_rgb
output_size = imagenet.output_size

def model(X):
                                        #in        out
    w1 = tf.Variable(tf.random_normal([input_size, 7500], stddev=0.1))
    b1 = tf.Variable(tf.zeros([7500]))  # bias는 0으로 초기화

    w2 = tf.Variable(tf.random_normal([7500, 5000], stddev=0.1))
    b2 = tf.Variable(tf.zeros([5000]))

    w3 = tf.Variable(tf.random_normal([5000, 2500], stddev=0.1))
    b3 = tf.Variable(tf.zeros([2500]))

    w4 = tf.Variable(tf.random_normal([2500, 1000], stddev=0.1))
    b4 = tf.Variable(tf.zeros([1000]))

    w5 = tf.transpose(w4)
    b5 = tf.Variable(tf.zeros([2500]))

    w6 = tf.transpose(w3)
    b6 = tf.Variable(tf.zeros([5000]))

    w7 = tf.transpose(w2)
    b7 = tf.Variable(tf.zeros([7500]))

    w8 = tf.transpose(w1)
    b8 = tf.Variable(tf.zeros([input_size]))

    l1 = tf.nn.relu(tf.matmul(X,  w1) + b1)
    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
    l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
    l5 = tf.nn.relu(tf.matmul(l4, w5) + b5)
    l6 = tf.nn.relu(tf.matmul(l5, w6) + b6)
    l7 = tf.nn.relu(tf.matmul(l6, w7) + b7)
    l8 = tf.nn.relu(tf.matmul(l7, w8) + b8)

    return  l4, l8


X = tf.placeholder("float", [None, input_size], name='X')

value, net = model(X)
cost = tf.reduce_mean(tf.pow(X - net, 2))   # (입력 - 네트워크 출력)^2
trainop = tf.train.GradientDescentOptimizer(0.02).minimize(cost)
acc = tf.argmax(net, 1)
output = value

train_length, test_length= imagenet.getnum()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10):
        for start, end in zip(range(0, train_length, 128), range(128, train_length, 128)):
            trX, trY = imagenet.getdata(start, end, train=True) #메모리에 다 올려놓고 하는게 아니라 배치를 일부 메모리에 올린다.

            input_ = trX
            loss, _ = sess.run([cost, trainop], feed_dict={ X: input_})
            print(loss)

        teX, teY = imagenet.getdata(0, test_length, train=False)
        loss = sess.run(cost, feed_dict={ X: teX})
        print(loss)