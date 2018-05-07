import tensorflow as tf
import numpy as np
import os
from PIL import Image
import shutil


def onehot(x, size, list=False):
    if list == False:
        onehot = np.zeros(size)
        onehot[x] = 1
        return onehot
    else:
        onehot = np.zeros([len(x), size])
        onehot[range(len(x)), x] = 1
        return onehot



# def onehot(x, length):  # (onehot 배열 크기, 1이 되야하는 인덱스)
#
#     onehot = np.zeros([len(x), length], dtype=np.uint8)
#     onehot[range(len(x)), x] = 1
#
#     return onehot

def save(path, sess):
    saver = tf.train.Saver()
    if not os.path.exists(path):
        os.makedirs(path)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    return saver

def graph(path_train, sess):
    path_test = path_train + "/eval"

    if not os.path.exists(path_test):
        os.makedirs(path_test)
    #shutil.rmtree(path)
    writer = tf.summary.FileWriter(path_train, sess.graph)
    writer_test = tf.summary.FileWriter(path_test, sess.graph)
    merged = tf.summary.merge_all()

    return writer, writer_test, merged

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_image(image, path):
    im = image.astype('uint8')

    create_path(path)

    for i in range(128):
        image_path = path + "/" + str(i) + ".jpg"

        image = Image.fromarray(im[i])
        image.save(image_path)


# def conv_layer(x, kernel_size, output_num, name='var', var_list=None, strides=[1, 1, 1, 1], activation='relu', dropout=None):
#     kernel = tf.get_variable(name, [kernel_size[0], kernel_size[1], x.get_shape().as_list()[3], output_num], initializer=tf.contrib.layers.xavier_initializer_conv2d())
#     bias = tf.Variable(tf.zeros(output_num))
#     conv = tf.nn.conv2d(x, kernel, strides=strides, padding='SAME')
#
#     if not var_list is None:
#         var_list.list.append(kernel)
#         var_list.list.append(bias)
#
#     if activation=='relu':
#         y = tf.nn.relu(tf.nn.bias_add(conv, bias))
#     elif activation=='sigmoid':
#         y = tf.nn.sigmoid(tf.nn.bias_add(conv, bias))
#     elif activation=='none':
#         y = tf.nn.bias_add(conv, bias)
#
#     if not dropout == None:
#         y = tf.nn.dropout(y, dropout)
#
#     return y
#
# def create_layer(x, output_num, name='var', var_list=None, activation='relu', dropout=None):
#     weight = tf.get_variable(name, [x.get_shape().as_list()[1], output_num],
#                              initializer=tf.contrib.layers.xavier_initializer())
#     bias = tf.Variable(tf.zeros(output_num))
#
#     if not var_list is None:
#         var_list.list.append(weight)
#         var_list.list.append(bias)
#
#     if activation == 'relu':
#         y = tf.nn.relu(tf.matmul(x, weight) + bias)
#     elif activation == 'sigmoid':
#         y = tf.nn.sigmoid(tf.matmul(x, weight) + bias)
#     elif activation == 'none':
#         y = tf.matmul(x, weight) + bias
#     else:
#         y = -1
#
#     if not dropout is None:
#         y = tf.nn.dropout(y, dropout)
#
#     return y

def conv_layer(x, name, kernel, output_num, strides):
    kernel = tf.get_variable(name, [kernel[0], kernel[1], x.get_shape().as_list()[3], output_num])
    bias = tf.Variable(tf.zeros(output_num))

    y = tf.nn.conv2d(x, kernel, strides=strides, padding='SAME')
    y = tf.nn.bias_add(y, bias)
    return y, kernel, bias

def fc_layer(x, name, output_num):
    weight = tf.get_variable(name, [x.get_shape().as_list()[1], output_num],
                             initializer=tf.random_normal_initializer(stddev=0.01))
    bias = tf.Variable(tf.zeros(output_num))

    y = tf.matmul(x, weight) + bias

    return y, weight, bias

def deconv_layer(x, kernel_shape, output_num, strides, activation):
    y = tf.layers.conv2d_transpose(x, output_num, kernel_shape, [strides[1], strides[2]],
                                   padding='SAME',
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

    return y

def create_layer(layer, x, output_num, kernel_shape = [5, 5], strides = [1, 1, 1, 1],
                 name='var', var_list=None, activation='relu', dropout=None
                 ):    #output shape, batch는 나중에 클래스 형태로 하게되면 처리 가능할 듯.

    y, weight, kernel, bias = None, None, None, None

    if layer == 'conv':
        y, kernel, bias = conv_layer(x, name, kernel_shape, output_num, strides)
    elif layer == 'fc':
        y, weight, bias = fc_layer(x, name, output_num)
    elif layer == 'deconv':
        y = deconv_layer(x, kernel_shape, output_num, strides, activation)


    if not var_list is None:
        if layer == 'conv':
            var_list.append(kernel)
            var_list.append(bias)
        elif layer == 'fc':
            var_list.append(weight)
            var_list.append(bias)

    if activation == 'relu':
        y = tf.nn.relu(y)
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(y)
    elif activation == 'none':
        pass
    else:
        print("activation is not recognized")
        y = -1

    if not dropout is None:
        y = tf.nn.dropout(y, dropout)

    return y


