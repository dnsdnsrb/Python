import tensorflow as tf
import numpy as np
"""
x = [[1, 1, 1, 1, 1],       #bias
     [0., 2., 0., 4., 0.],  #x2
     [1., 0., 3., 0., 5.]]  #x1
y = [1, 2, 3, 4, 5]
"""

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x = xy[0:-1]    #[행][열]에서 뒤의[열]만 건들이고 있다. 0~2열까지, 여기선 0~3열까지 있으므로 -1은 3열을 의미하고, 0:3는 0~2까지를 의미하므로 0~2까지이다.
y = xy[-1]      #2열만

X = tf.placeholder(tf.float32)  #입력
Y = tf.placeholder(tf.float32)  #출력

print('x', x)
print('y', y)
#W = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))   #[1,2] 배열
W = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))   #[1,3] 배열
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#hypothesis = tf.matmul(W, x) + b    #matmul이 중요, 행렬 곱이다.
hypothesis = tf.matmul(W, X)    #matmul이 중요, 행렬 곱이다. 행렬이 아닌 버전과 비교해봐라.
                                # 동시에 이렇게 표현되어있으면 바이어스가 포함된 것이라 보는게 좋다.
                                #bias는 입력(X)에 포함되어있다. 찾을 때 주의

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learningrate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learningrate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x, Y:y})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X:x, Y:y}), sess.run(W))

#이건 행렬에 bias, 입력1, 입력2를 모두 넣어놔서 이렇게 줘야한다. 매우 불편
print(sess.run(hypothesis, feed_dict={X: [[1], [2], [2]] }))