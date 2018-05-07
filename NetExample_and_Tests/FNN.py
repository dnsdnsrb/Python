import tensorflow as tf
import numpy
import input_data
import Imagenet
import NNutils

from datetime import datetime
def NN(X):
    w1 = tf.Variable(tf.random_normal([32*32*3, 2048], stddev=0.1))
    b1 = tf.Variable(tf.zeros([2048]))

    w2 = tf.Variable(tf.random_normal([2048, 1400], stddev=0.1)) #MNIST 숫자가 10개이므로(0~9) 10이다.
    b2 = tf.Variable(tf.zeros([1400]))

    w3 = tf.Variable(tf.random_normal([1400, 900], stddev=0.1))
    b3 = tf.Variable(tf.zeros([900]))

    w4 = tf.Variable(tf.random_normal([900, 600], stddev=0.1))
    b4 = tf.Variable(tf.zeros([600]))

    w5 = tf.Variable(tf.random_normal([600, 300], stddev=0.1))
    b5 = tf.Variable(tf.zeros([300]))

    w6 = tf.Variable(tf.random_normal([300, 10], stddev=0.1))
    b6 = tf.Variable(tf.zeros([10]))

    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
    l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
    l5 = tf.nn.relu(tf.matmul(l4, w5) + b5)
    Y = tf.matmul(l5, w6) + b6

    return Y

X = tf.placeholder("float", [None, 32 * 32 * 3])
Y = tf.placeholder("float", [None, 10])
global_step = tf.Variable(0, trainable=False)

with tf.name_scope("cost"):
    nn_Y = NN(X)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= nn_Y, labels= Y))
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost, global_step)
    tf.summary.scalar("cost", cost)

predict_op = tf.equal(tf.arg_max(Y, 1), tf.arg_max(nn_Y, 1))
acc_op = tf.reduce_mean(tf.cast(predict_op, "float"))

#데이터 관련
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
dataset = Imagenet.Cifar()
trX, trY, teX, teY = dataset.getdata()
trX = trX.reshape(-1, 32 * 32 * 3)
teX = teX.reshape(-1, 32 * 32 * 3)

print(tf.shape(cost))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    batch_size = 128
    st_time = datetime.now()
    print(st_time)

    savepath = "Networkfile/NN"
    saver = NNutils.save(savepath, sess)
    writer, merge = NNutils.graph("Networkfile/NN", sess)

    for i in range(1):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trY), batch_size)):
            summary, _, loss, step = sess.run([merge, train_op, cost, global_step], feed_dict={X: trX[start:end], Y: trY[start:end]})
            print(step, loss)
            writer.add_summary(summary, step)
        #saver.save(sess, savepath + "/model.ckpt", step)

    end_time = datetime.now()
    print("걸린 시간 = ", (end_time - st_time))

    #accuracy, loss = sess.run([acc_op, cost], feed_dict={X: teX, Y: teY})
    #print(datetime.now(), i, accuracy, loss)

