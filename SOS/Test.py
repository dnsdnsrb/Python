from sys import path
path.append('../Data/')

import numpy as np
import tensorflow as tf
import scipy
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_rcv1
import keyboard

def rcv1():
    rcv1 = fetch_rcv1(subset='train')
    # train_data = rcv1.data
    # train_label = rcv1.target

    train_data = csr_matrix(rcv1.data[0:1000]).toarray()
    # train_data2 = csr_matrix(rcv1.data).toarray()
    # print(train_data.shape)
    train_data = tf.train.batch([train_data], 128)

def macro():
    keyboard.press_and_release('space')
    keyboard.write('sss')

def sparse():
    rcv1 = fetch_rcv1(subset='train')

    train = rcv1.data

    print("들어감")
    a = scipy.sparse.coo_matrix(train[0:2])

    indices = np.array([a.row, a.col], dtype=np.int64).T
    # indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = a.data
    print(a.shape)


    # print(b.get_shape())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        b = tf.SparseTensor(indices=np.array([a.row, a.col]).T,
                            values=a.data,
                            dense_shape=a.shape)
        print(sess.run(b))


sparse()