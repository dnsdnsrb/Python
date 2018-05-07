import tensorflow as tf

x = [1, 2, 3]   #입력 데이터
y = [1, 2, 3]   #출력 데이터

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)  #입력
Y = tf.placeholder(tf.float32)  #출력

#model
hypothesis = W * X + b  #W = 가중치, b = bias X = 입력
#여기선 입력과 출력이 동일하기 때문에 y = x + 0 즉, y = x의 그래프가 형성된다.

#cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))    #cost 혹은 loss 함수, 오차를 나타내는 함

#train
a = tf.Variable(0.1)    #Learning rate
optimizer = tf.train.GradientDescentOptimizer(a)    #최적화 알고리즘

train = optimizer.minimize(cost)

init = tf.initialize_all_variables()    #초기화법

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x, Y:y})   #학습 부분
    #sess.run(train) #학습 부분
    if step % 100 == 0:
        print(step, sess.run(cost, feed_dict={X:x, Y:y}), sess.run(W), sess.run(b))

#python에선 placeholder로 입력을 줘야하는 듯 하다. 매우 불편한데?
print(sess.run(hypothesis, feed_dict={X:5}))
print(sess.run(hypothesis, feed_dict={X:2.5}))
