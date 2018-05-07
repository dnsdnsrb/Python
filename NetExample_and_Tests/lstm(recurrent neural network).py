import tensorflow as tf
import numpy as np
import input_data

#a = [[1,2], [2, 3], [4, 5]]
#print(a[-1][1])

input_vec_size = lstm_size = 28
time_step_size = 28

batch_size = 128
test_size = 256

def init(shape):
    return  tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, lstm_size):

    W = init([lstm_size, 10])
    B = init([10])

    #이미 만들어져 있는 함수에 넣어서 돌리는데 거기에 맞게 자료형을 변형한다.
    # X = tf.transpose(X, [1, 0, 2]) #batch size, time step size, lstm size => time step size, batch size, lstm size
    X = tf.reshape(X, [[-1, 28*28]]) #열이 28개인 행렬로 바꾼다.
    # print(XR.)
    # print(X.shape)
    # XR = tf.as_dtype()

    # X = tf.split(X, lstm_size, 2) #1개의 행렬을 28(time_step_size)개로 나눈다.
    #spilt(분할할 축, 분할 크기 또는 수, 분할할 배열)
    #넣는 부분
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    print(X.shape)
    outputs, _states = tf.nn.dynamic_rnn(lstm, X, dtype=tf.float32)

    #마지막으로 연산?
    return tf.matmul(outputs[-1], W) + B, lstm.state_size

#MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY = mnist.train.images, mnist.train.labels
teX, teY = mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

#init
X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("float", [None, 10])

#Network Setup
net, state_size = model(X, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.equal(tf.argmax(Y, 1), tf.argmax(net, 1))
acc_op = tf.reduce_mean(tf.cast(predict_op, "float"))

#
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #tf.initialize_all_variables().run()

    for i in range(10):
        for start, end in zip(range(0, len(trX), batch_size),
                              range(batch_size, len(trX), batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y:trY[start:end]})

        #test data 순서를 섞어서 테스트에 이용
        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, sess.run(acc_op, feed_dict={X: teX[test_indices], Y:teY[test_indices]}))