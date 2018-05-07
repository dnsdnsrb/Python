import tensorflow as tf
import numpy as np
import input_data

mnist_width = 28    #글자 이미지 크기를 의미한다.
n_visible = mnist_width * mnist_width   #28*28 크기의 글자 이미지
n_hidden = 500
corruption_level = 0.3  #noise를 섞는다. 따라서 이 코드는 denoising stacked autoencoder로 보임.

def model(X, mask):

    tilde_X = mask * X
    W_init_noise = 4 * np.sqrt(6. / (n_visible + n_hidden))  # noise 최대 범위

    W_init = tf.random_uniform(shape=[n_visible, n_hidden], minval=W_init_noise, maxval=W_init_noise) #[784, 500] 행렬을 만들면서, 초기화를 한다.
    W2_init = tf.random_uniform(shape=[n_hidden, 250], minval=W_init_noise, maxval=W_init_noise)    #[500, 250] 행렬을 만들면서, 초기화를 한다.
                                        #in   out
    w1 = tf.Variable(W_init, name='W')  #784, 500    noise를 넣어 초기화
    b1 = tf.Variable(tf.zeros([500]), name='b')  # bias는 0으로 초기화

    w2 = tf.Variable(W2_init)           #500, 250
    b2 = tf.Variable(tf.zeros([250]))

    w3 = tf.transpose(w2)               #250, 500
    b3 = tf.Variable(tf.zeros([500]))

    w4 = tf.transpose(w1)               #500, 784
    b4 = tf.Variable(tf.zeros([28*28]))


    l1 = tf.nn.sigmoid(tf.matmul(tilde_X, w1) + b1)
    l2 = tf.nn.sigmoid(tf.matmul(l1,w2) + b2)
    l3 = tf.nn.sigmoid(tf.matmul(l2,w3) + b3)
    l4 = tf.nn.sigmoid(tf.matmul(l3,w4) + b4)
    return  l2, l4


X = tf.placeholder("float", [None, n_visible], name='X')
#Y = tf.placeholder("float", [None, n_visible], name='Y')
mask = tf.placeholder("float", [None, n_visible], name='mask')

#초기화 시 노이즈를 넣게되는데 그 범위를 적당히 집어넣어서 초기화하기 때문에 이렇게 생긴다.

#noise를 넣어 초기화하기 위한 변수

#마찬가지
#W_prime = tf.transpose(W)   #encoder의 반대는 decoder
#b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')    #마찬가지로 반대인데 여기선 n_visible로 해놓음

value, net = model(X, mask)
cost = tf.reduce_mean(tf.pow(X - net, 2))   # (입력 - 네트워크 출력)^2
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y))
trainop = tf.train.GradientDescentOptimizer(0.02).minimize(cost)
acc = tf.argmax(net, 1)
output = value


mnist = input_data.read_data_sets("MNIST_Data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            input_ = trX[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(trainop, feed_dict={ X: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np})) #여기 결과는 정확도가 아니라 cost이다. 이유는 비지도학습이기 때문
        #print(i, np.mean(np.argmax(teY, axis=1) == sess.run(acc, feed_dict={X: teX, mask: mask_np})))