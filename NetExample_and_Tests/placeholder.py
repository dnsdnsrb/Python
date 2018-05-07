import tensorflow as tf

a = tf.placeholder(tf.int16)    #placeholder 나중에 값을 정한다.
b = tf.placeholder(tf.int16)    #함수 작동시킬 때 값을 집어넣는게 좋은 듯 하다.

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
    print("add : %i" % sess.run(add, feed_dict={a:2, b:3}))
    print("mul : %i" % sess.run(mul, feed_dict={a:2, b:3}))

c = 2
d = 2

sub = tf.sub(c, d)

with tf.Session() as sess:
    print("sub : %i" % sess.run(sub))