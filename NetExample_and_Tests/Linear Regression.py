import tensorflow as tf

x = [1, 2, 3]   #입력
y = [1, 2, 3]   #출력

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  #초기값을 랜덤으로 준다.
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W * x + b  #W = 가중치, b = bias

cost = tf.reduce_mean(tf.square(hypothesis - y))    #cost 혹은 loss 함수, 오차를 나타내는 함

a = tf.Variable(0.1)    #Learning rate
optimizer = tf.train.GradientDescentOptimizer(a)    #최적화 알고리즘
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()    #초기화법

sess = tf.Session()
sess.run(init)

#Graph  그래프라고 부르는 이유는 학습을 하면 선이 생겨서다(여기선 y = x) 분류를 나누는 것이기 때문인 듯하
for step in range(2001):
    #sess.run(train, feed_dict={X:x, Y:y})   #학습 부분
    sess.run(train) #학습 부분
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

#print(sess.run(hypothesis, 5))
#print(sess.run(hypothesis, 2.5))