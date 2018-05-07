import tensorflow as tf
import numpy as np
#import sympy
import math
import PIL
from PIL import ImageOps
from PIL import Image
import random
import Imagenet
import DataSet
from datetime import datetime
import os

"""
print("d")

matrix = np.random.random([1024, 64])  # 64-dimensional embeddings

for i in range( 0, matrix.shape[0]*matrix.shape[1] ):
    matrix[int(i/64)][i%64] = i
    #print( int(i/64), i%64)

ids = np.array([0, 5, 17, 33])
#print(matrix)
np.set_printoptions(suppress=True)
#print(matrix[ids])  #0행, 5행, 17행, 33행을 출력하라는 의미
#print(matrix.size)
#print(ids.shape[0])
x, y, sigma = sympy.symbols('x y sigma')
formula = 1/(2 * 3.14 * sigma**2)**0.5 * sympy.exp( - (x**2 + y**2) / (2*sigma**2) )

expr = x**2 + 2 * y
print(expr)

print(sympy.diff(formula, x))
print(sympy.diff(sympy.diff(formula, x), x))
print(sympy.diff(sympy.diff(formula, y), y))
fx, fy, a = 0.0, 0.0, 1.4
ddx = -0.399043442233811*(a**2)**(-0.5)*math.exp((-fx**2 - fy**2)/(2*a**2))/a**2 + 0.399043442233811*fx**2*(a**2)**(-0.5)*math.exp((-fx**2 - fy**2)/(2*a**2))/a**4
ddy = -0.399043442233811*(a**2)**(-0.5)*math.exp((-fx**2 - fy**2)/(2*a**2))/a**2 + 0.399043442233811*fy**2*(a**2)**(-0.5)*math.exp((-fx**2 - fy**2)/(2*a**2))/a**4
formula = ddx + ddy

print(formula)

#3차원 배열이 2차원 배열로 바뀐다.
a = np.random.random([6, 2, 4])
print(a.shape[0])
b = a.reshape( (a.shape[0], a.shape[1]*a.shape[2]) )  #둘다 똑같은 걸 봐선 알아서 처리해버리는 모양 (6, 8)로 알아서 인식
#b = a.reshape(a.shape[0], a.shape[1]*a.shape[2])
print(a.shape[0], a.shape[1], a.shape[2])
print(b.shape[0], b.shape[1])

#a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
sess = tf.Session()
a = tf.random_uniform((2, 2, 3), minval= 0, maxval= 9, dtype=tf.int32)
#b = tf.transpose(a, perm=[0, 2, 1])
b = tf.split(0, 2, a)
#기본적으로 perm[차원 번호, ...]이다. 만약 3차원이라면 반대로 뒤집기 때문에 perm은 기본값으로 [2, 1, 0]이 된다.
#ex> [3][2][1] 행렬은 기본적으로 [1][2][3] 행렬로 치환됨( [0차원][1차원][2차원]이기 때문).
#perm = [1, 2, 0]인 경우, [2][1][3] 행렬로 치환됨.
print(sess.run([a,b]))
print(tf.shape(a))
print(tf.shape(b))

print(a.shape[0])
a = a.flat[2]
print(a)
print(a.ravel())
print(np.arange(5) * 10)
"""

# im = Image.open("b.jpg")
# im =ImageOps.fit(im, (5, 5))
# #im.save("compressed.jpg", quality=10)
# im.show()

#with tf.Session() as sess:
#    tf.max
dataset = Imagenet.Cifar()
trX, trY, teX, teY = dataset.getdata()
#dataset = DataSet.Cifar()
#trX, trY, teX, teY = dataset.create_sets()


def shuffle(images, labels):
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    return images, labels

def distort():
    number = random.randrange(1, 3)
    image = np.flip(trX, number)
    image = trX[:]
    image = image.astype(np.uint8)
    image = Image.fromarray(image[0])
    image.show()

def brightness():
    array = trX
    array = array * 1.5
    array = np.clip(array, 0, 255)
    array = array.astype(np.uint8)
    image = Image.fromarray(array[0])
    image.show()

def standard():
    array = trX[0]
    array = (array - np.mean(array)) / np.std(array)
    array = array.astype(np.uint8)
    print(array)
    image = Image.fromarray(array)
    image.show()

def denosing(arr, percent):
    #mask = np.arange(arr.shape[0])
    #np.random.shuffle(mask)

    # batch = np.arange(arr.shape[0])
    # np.random.shuffle(batch)
    # row = np.random.randint(0, arr.shape[1], num * arr.shape[0])
    # col = np.random.randint(0, arr.shape[2], num * arr.shape[0])
    #
    # arr[, row, col, :] = 255
    #
    # batch = np.arange(arr.shape[0])
    # np.random.shuffle(batch)
    # row = np.random.randint(0, arr.shape[1], num * arr.shape[0])
    # col = np.random.randint(0, arr.shape[2], num * arr.shape[0])
    #
    # arr[batch, row, col, :] = 0
    num = int(arr.shape[1] * percent) * int(arr.shape[2] * percent)

    for batch in range(arr.shape[0]):
        row = np.random.randint(0, arr.shape[1], num)
        col = np.random.randint(0, arr.shape[2], num)
        arr[batch, row, col, :] = 255

    for batch in range(arr.shape[0]):
        row = np.random.randint(0, arr.shape[1], num)
        col = np.random.randint(0, arr.shape[2], num)
        arr[batch, row, col, :] = 0

    arr = arr[0].astype(np.uint8)
    print(arr)
    image = Image.fromarray(arr)

    image.show()

def tfnoise(origin, rate):
    rate = 255 * 0.5 * rate
    mask = tf.random_uniform([128, 32, 32, 3], -rate, rate)
    print(rate)
    arr = origin + mask
    arr = tf.clip_by_value(arr, 0, 255)
    with tf.Session() as sess:
        arr = sess.run(arr)

        arr = arr[0].astype(np.uint8)
        image = Image.fromarray(arr)

        image.show()

        arr = sess

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

def image():
    print('what')
    #denosing(trX[0:128], 0.2)
    #tfnoise(trX[0:128], 0.2)

    # st_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    # str = "name" + st_time
    # print(str)
        #print(trX[0], trY[0])
    #list = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    #print( random.sample(list, 5) )
    #print(list[][0])
    #print( 1 in list)
    # class get():
    #     pass
    # get = get()
    # def te2():  #이름에 test라는 게 들어가면 unit test로 실행되버림.
    #     with tf.variable_scope('c'):
    #         get.a = tf.Variable(2, name='a')
    #         b = tf.get_variable('b', [1])
    #     with tf.variable_scope('d'):
    #         get.d = tf.Variable(2, name='a')
    #         get.e = tf.Variable(2, name='b')
    #
    # te2()
    #
    # with tf.variable_scope("e"):
    #     #name = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd/a:0')
    #     name = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
    #     #print(name)
    #     print(name)
    # list = []
    # output = get.a
    # list.append(output)
    # output = get.d
    # list.append(output)
    # print(list)

    #list = [int(random.random() * 10)] * 5
    #list[1] = 1
    #print(list)
    #list = np.array(
    #    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15],
    #)
    #x = 1
    #y = 2

    # list = [1, 2, 3, 4, 5]
    # print(list[1:5])

def conv_layer(x, size, shape, name='var', strides=[1, 1, 1, 1], activation='relu',dropout=None):
    kernel = tf.get_variable(name, [size[0], size[1], x.get_shape().as_list[3], shape], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    bias = tf.Variable(tf.zeros(shape))
    conv = tf.nn.conv2d(x, kernel, strides=strides, padding='SAME')

    if activation=='relu':
        y = tf.nn.relu(tf.nn.bias_add(conv, bias))
    elif activation=='sigmoid':
        y = tf.nn.sigmoid(tf.nn.bias_add(conv, bias))
    elif activation=='none':
        y = tf.nn.bias_add(conv, bias)

    if not dropout == None:
        y = tf.nn.dropout(y, dropout)
    return y

def create_layer(x, shape, var_list=False, name='var',activation='relu', dropout=None):
    weight = tf.get_variable(name, [x.get_shape().as_list[1], shape], initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.Variable(tf.zeros(shape))

    # if var_list==True:
    #     Temp.list.append(weight)
    #     Temp.list.append(bias)

    if activation=='relu':
        y = tf.nn.relu(tf.matmul(x, weight) + bias)
    elif activation=='sigmoid':
        y = tf.nn.sigmoid(tf.matmul(x, weight) + bias)
    elif activation=='none':
        y = tf.matmul(x, weight) + bias
    else:
        y = -1

    if not dropout == None:
        y = tf.nn.dropout(y, dropout)

    return y

def onehot():
    data =[[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]
    data = np.argmax(data, 1)
    print(data)


onehot()