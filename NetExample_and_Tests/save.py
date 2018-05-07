import tensorflow as tf
import  numpy as np
import os
import  input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(inputData, dropout1, dropout2):

    w1 = init_weights([784, 625])
    w2 = init_weights([625, 500])
    w3 = init_weights([500, 10])

    inputData = tf.nn.dropout(inputData, dropout1)

    l1 = tf.nn.relu(tf.matmul(inputData, w1))
    l1 = tf.nn.dropout(l1, dropout2)

    l2 = tf.nn.relu(tf.matmul(l1, w2))
    l2 = tf.nn.dropout(l2, dropout2)

    l3 = tf.matmul(l2, w3)

    return l3

#변수 설정
inputData = tf.placeholder("float", [None, 784])
outputData = tf.placeholder("float", [None, 10])
dropout1 = tf.placeholder("float")
dropout2 = tf.placeholder("float")

#네트워크 설정
net = model(inputData, dropout1, dropout2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, outputData))
trainop = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predictop = tf.argmax(net, 1)

#데이터 설정
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trainInput, trainOutput = mnist.train.images, mnist.train.labels
testInput, testOutput = mnist.test.images, mnist.test.labels

#네트워크 저장 설정
netdir =  "./Networkfile"
if not os.path.exists(netdir):
    os.makedirs(netdir)

global_step = tf.Variable(0, name='global_step', trainable=False)   #trainable : 그래프에 표시 여부인 듯 GraphKeys.True(default)이면 GraphKeys.TRAINABLE_VARIABLES를 하면 그래프에 사용되는 듯 하다.

saver = tf.train.Saver()

#saver를 정의하고 나면 그 다음부터 선언하는 변수는 saver에 저장 안된다. 주의, 아래의 변수도 마찬가지로 저장안됨.
non_storable_variable = tf.Variable(777)

#실행
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    #모델이 이미 있으면 불러와서 사용한다.
    n = tf.train.get_checkpoint_state(netdir)
    if n and n.model_checkpoint_path:
        print(n.model_checkpoint_path)
        saver.restore(sess, n.model_checkpoint_path)

    start = global_step.eval()
    print("Start from : ", start)

    for i in range(start, 100):
        for start, end in zip( range(0, len(trainInput), 128), range(128, len(trainInput), 128) ):
            sess.run(trainop, feed_dict={inputData: trainInput[start:end], outputData: trainOutput[start:end],
                                         dropout1: 0.8, dropout2: 0.5})

        global_step.assign(i).eval()
        saver.save(sess, netdir + "/model.ckpt", global_step=global_step)
        print(i, np.mean(np.argmax(testOutput, axis=1) == sess.run(predictop, feed_dict={inputData: testInput,
                                                                                         outputData: testOutput,
                                                                                         dropout1: 1.0,
                                                                                         dropout2: 1.0})))
