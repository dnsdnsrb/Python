import tensorflow as tf
import numpy as np
import NNutils
from datetime import datetime
import Input
import output
import model_img
import model_ir
from PIL import Image
def fc_layer(name, x, output_num, activation='none', dropout=None):
    weight = tf.get_variable(name, [x.get_shape().as_list()[1], output_num],
                             initializer=tf.random_normal_initializer(stddev=0.01))
    bias = tf.Variable(tf.zeros(output_num))

    y = tf.matmul(x, weight) + bias

    if activation == 'relu':
        y = tf.nn.relu(y)
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(y)
    elif activation == 'none':
        pass

    if not dropout is None:
        y = tf.nn.dropout(y, dropout)

    return y

class Network():
    def __init__(self, image=[960, 540, 3], actions=20):
        # self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.im_size = [image[0], image[1], image[2]]
        self.actions = actions  #행동가능 경우의 수
        # self.ir = ir            #inner representation, 내부 표현

        self.x = tf.placeholder(tf.float32, [None, image[0], image[1], image[2]])
        self.obj = tf.placeholder(tf.float32, [None, image[0], image[1], image[2]])
        self.y = tf.placeholder(tf.float32)

        self.dropout_conv = tf.placeholder("float")
        self.dropout_normal = tf.placeholder("float")



    def model(self, x, conv_layers = [6, 12, 24, 48], fc_layers = [200, 100]):#[28*28, 450, 300, 200, 100, 50, 10]
        image_size = self.im_size[0]    #248 * 248 * 3
        kernel_size = max([int(image_size / 5), 2])
        pool_size = max([int(image_size / 10), 2])
        output = x

        for i, layer in enumerate(conv_layers):
            with tf.variable_scope('conv' + str(i)):
                output = tf.layers.conv2d(output, layer, kernel_size, padding='SAME', activation=tf.nn.relu)
                output = tf.layers.max_pooling2d(output, pool_size, pool_size, padding='SAME')
                output = tf.nn.dropout(output, self.dropout_conv)

            image_size = output.get_shape().as_list()[1]
            # print(image_size)
            kernel_size = max([int(image_size / 5), 2])
            pool_size = max([int(image_size / 10), 2])

        with tf.name_scope('reshape'):
            reshape_size = image_size * image_size * conv_layers[-1]    #-1은 제일 끝을 나타낸다.
            output = tf.reshape(output, [-1, reshape_size])
            # print(output)

        for i, layer in enumerate(fc_layers):
            with tf.variable_scope('fc' + str(i)):
                output = tf.layers.dense(output, layer, activation=tf.nn.relu)
                output = tf.nn.dropout(output, self.dropout_normal)

        with tf.variable_scope('fc'):
            output = tf.layers.dense(output, self.actions, activation=None)

        y_ = output
        return y_

    def train(self):
        #model
        self.y_ = self.model(self.x)
        act = output.Action()
        #cost and training
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.pow(self._ - self.y, 2))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.cost, global_step=self.global_step)

            tf.summary.scalar("cost", self.cost)

def show_img(img):

    img = np.uint8(img)
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img[0], 'RGB')
    img.show()

if __name__ == '__main__':
    sight = Input.Sight()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        info_type = 'image'

        step = 0
        while step < 1000:
            step += 1
            #환경 정보 -> 내부 표현1, 동시에 들어오는 정보는? 지금은 시각 정보만 처리된다.
            origin = sight.get_image()
            y, cost, img, img2 = sess.run([imgnet.opt, imgnet.cost, imgnet.y_de, imgnet.y],
                                    feed_dict={imgnet.x: origin,
                                               imgnet.y: origin,
                                               imgnet.dropout_conv: 1.0,
                                               imgnet.dropout_normal: 1.0}) #먼저 출력을 내고(학습은 안됨), 다른 정보 시스템은?

            print("step : ", step, cost)

        print(img)
        img = img * 255
        img2 = img2 * 255

        show_img(img)
        show_img(img2)