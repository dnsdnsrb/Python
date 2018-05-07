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

    # def model_ir(self, x, layers=[40, 30, 20]):
    #     with tf.variable_scope('ir'):
    #         output = x
    #
    #         for i, layer in enumerate(layers):
    #             with tf.variable_scope('fc' + str(i)):
    #                 output = tf.layers.dense(output, layer, activation=tf.nn.relu)
    #
    #         with tf.variable_scope('fc'):
    #             output = tf.layers.dense(output, self.actions, activation=None)
    #
    #         y_ = output
    #
    #         return y_

    def train(self):
        #model
        self.y_ = self.model(self.x)
        act = output.Action()
        #cost and training
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=self.y))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.cost, global_step=self.global_step)

            tf.summary.scalar("cost", self.cost)

        # with tf.name_scope("accuracy"):
        #     compare = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(compare, "float"))
        #
        #     tf.summary.scalar("accuarcy", self.accuracy)

    # def run(self, step_limit):
    #     with tf.Session() as sess:
    #         tf.global_variables_initializer().run()
    #
    #         dataset = DataSetNp.Cifar()
    #         train_data, train_label, test_data, test_label = dataset.getdata()
    #
    #         path = "AI/" + str(step_limit) + ""
    #         saver = NNutils.save(path, sess)
    #         writer, writer_test, merged = NNutils.graph(path, sess)
    #
    #         step = sess.run(self.global_step)
    #         while step < step_limit:
    #             print("step :", step)
    #             for start, end in zip(range(0, len(train_data), self.batch_size),
    #                                   range(self.batch_size, len(train_data), self.batch_size)):
    #                 summary, \
    #                 _, loss, \
    #                 step = sess.run([merged,
    #                                  self.training, self.cost,
    #                                  self.global_step],
    #                                 feed_dict={self.x: train_data[start:end],
    #                                            self.y: train_label[start:end],
    #                                            self.dropout_conv: 1.0,
    #                                            self.dropout_normal: 0.75})
    #
    #                 if step % 50 == 0:
    #                     writer.add_summary(summary, step)
    #                     print(step, datetime.now(), loss)
    #
    #             summary, \
    #             loss, \
    #             accuracy = sess.run([merged, self.cost, self.accuracy],
    #                                 feed_dict={self.x: test_data,
    #                                            self.y: test_label,
    #                                            self.dropout_conv: 1.0,
    #                                            self.dropout_normal: 1.0})
    #
    #             writer_test.add_summary(summary, step)
    #             print("test results : ", accuracy, loss)
    #             saver.save(sess, path + "/model.ckpt", step)

def show_img(img):

    img = np.uint8(img)
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img[0], 'RGB')
    img.show()

if __name__ == '__main__':
    sight = Input.Sight()
    imgnet = model_img.Network(image=sight.im_size)
    irnet = model_ir.Network()

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


        # img = np.uint8(img)
        # img = Image.fromarray(img[0], 'RGB')
        # img.show()
        # img = np.uint8(img2)
        # img = Image.fromarray(img[0], 'RGB')
        # img.show()

            # #목표 정보 -> 내부 표현2, 마찬가지로 시각 정보만 된다.
            # y = sess.run(imgnet.y_, feed_dict={imgnet.x: sight.get_object(),
            #                                    imgnet.dropout_conv: 1.0,
            #                                    imgnet.dropout_normal: 1.0})

            #y_와 y를 이어붙인다.
            # x = np.concatenate([y_, y])
            # print(x.shape)

            # #
            # #내부 표현1 + 내부표현2 -> 행동
            # #수정이 필요
            # print(y_)
            # y_, z = sess.run([irnet.y_, irnet.z], feed_dict={irnet.x: x,
            #                                    irnet.dropout_normal:1.0})
            # print(y_)
            #
            # #행동
            # y_ = np.reshape(y_, (2, 10))
            # act = output.Action()
            # act.act(y_)
            #
            # #확인 및 업데이트
            # y_, step = sess.run([imgnet.opt, imgnet.global_step], feed_dict={imgnet.x: sight.get_image(),
            #                                                                 imgnet.y: y,
            #                                                                 imgnet.dropout_conv: 1.0,
            #                                                                 imgnet.dropout_normal: 1.0})
            #
            # if(step % 50 == 0):
            #     acc = sess.run([imgnet.accuracy], feed_dict={imgnet.x: sight.get_image(),
            #                                                  imgnet.y: y,
            #                                                  imgnet.dropout_conv: 1.0,
            #                                                  imgnet.dropout_normal: 1.0})
            #     print("test :", datetime.now(), "step :", step, acc)

            #학습이 제대로 안됨, 여러 네트워크가 있는데 1개만 학습이 되는 것으로 보임, 대책이 필요
            #train()함수 내에서 처리하도록 해볼 것
            #



        # elif 텍스트:
        #  = sess.run(txtnet., feed_dict={net.obj: 목표 텍스트})
        # elif 소리:
        #  = sess.run(vocnet., feed_dict={net.obj: 목표 소리})
        # else:
        #  = sess.run(?net.y_, feed_dict={net.obj: 목표 ?})

        #  = sess.run(act, feed_dict={net.?: 내부표현})
        # act
        #
        # = sess.run(net.y, net.cost, feed_dict={net.x: 행동 후 이미지}) #학습용, 다른 정보 시스템은?
        #  = sess.run(act, feed_dict={net.?: 내부표현})   #학습용


        #그걸로 행동을 하고
        #그 동작 후 나온 이미지를 관측하여 주면, 학습이 된다.



    # model.run(1000)