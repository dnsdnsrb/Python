import tensorflow as tf
import numpy as np
def unstack():  #제일 왼쪽의 차원이 사라짐, a가 분할되는 것
    a = tf.Variable([ ])
    b, c = tf.unstack(a)
    d, e, f = tf.unstack(a, axis=1)
    g = tf.unstack(a, 0)
    print("g", g)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(a.shape)
        print(b)
        print(c)
        print("1", sess.run((a)))
        print("2", sess.run((b)))
        print("3", sess.run((c)))
        print("4", sess.run((d)))

def stack():    #제일 왼쪽에 차원이 새로 생김, a,b,c를 합치는 것
    a = []
    b = tf.Variable([3, 4])
    c = tf.Variable([5, 6])
    d = tf.Variable([7, 8])

    a.append(b)
    a.append(c)

    d = tf.stack(a)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(d))

if __name__ == '__main__':
    stack()
