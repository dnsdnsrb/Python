import tensorflow as tf

class Network():
    def __init__(self, image=[512, 512, 3], ir=100):
        # self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.im_size = [image[0], image[1], image[2]]
        self.ir = ir            #inner representation, 내부 표현

        self.x = tf.placeholder(tf.float32, [None, image[0], image[1], image[2]])
        self.y = tf.placeholder(tf.float32, [None, image[0], image[1], image[2]])

        self.dropout_conv = tf.placeholder("float")
        self.dropout_normal = tf.placeholder("float")

        self.train()

    # 현재 코드가 딱 2로 나누어 떨어져야 되게 되있다. => pool size가 2로 고정되서 그럼
    # 아니면 원본과 decode 결과 이미지 크기가 달라짐
    # 가로 세로가 동일한 그림만 됨. (이미지 크기를 가로만 보고 세로를 판단하고 있음)
    def encode(self, x, conv_layers = [12, 12, 12, 12], fc_layers = [100], kernel = [5, 5, 5, 5]):
        image_size = self.im_size[0]    #248 * 248 * 3
        output = x
        # print(output)
        with tf.variable_scope('encode'):

            for i, layer in enumerate(conv_layers):                         #Conv layers 구축부분
                # kernel_size = max([int(image_size / 5), 2])
                # pool_size = max([int(image_size / 5), 2])
                kernel_size = [5, 5]
                pool_size = 2
                with tf.variable_scope('conv' + str(i)):
                    output = tf.layers.conv2d(output, layer, kernel_size, padding='SAME', activation=tf.nn.relu)
                    output = tf.nn.dropout(output, self.dropout_conv)

                image_size = output.get_shape().as_list()[1]
                # print(image_size)

            with tf.name_scope('reshape'):
                reshape_size = image_size * image_size * conv_layers[-1]    #-1은 제일 끝을 나타낸다.
                output = tf.reshape(output, [-1, reshape_size])
                # print(output)

            for i, layer in enumerate(fc_layers):                           #fc layers 구축 부분
                with tf.variable_scope('fc' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)
                    output = tf.nn.dropout(output, self.dropout_normal)

            with tf.variable_scope('fc'):                                   #img -> ir
                output = tf.layers.dense(output, self.ir, activation=None)

        y_ = output
        return y_, image_size, reshape_size

    def decode(self, x, image_size, reshape_size, deconv_layers = [12, 12, 12, 12], fc_layers = [100]):
        # image_size = self.im_size[0]    #248 * 248 * 3
        output = x
        with tf.variable_scope('decode'):

            for i, layer in enumerate(fc_layers):                                       #ir -> img
                with tf.variable_scope('fc' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)
                    output = tf.nn.dropout(output, self.dropout_normal)

            with tf.name_scope('fc'):
                output = tf.layers.dense(output, reshape_size, activation=tf.nn.relu)
                output = tf.nn.dropout(output, self.dropout_normal)

            with tf.name_scope('reshape'):                              #reshape layers
                output = tf.layers.dense(output, reshape_size, activation=tf.nn.relu)
                output = tf.reshape(output, [-1, image_size, image_size, deconv_layers[0]])
                deconv_layers.pop(0)    #1번째 원소 제거, 그래야 encode랑 맞음

            for i, layer in enumerate(deconv_layers):                                   #Conv layers
                # kernel_size = max([int(image_size / 5), 2])
                kernel_size = [5, 5]
                pool_size = 2

                with tf.variable_scope('conv' + str(i)):
                    output = tf.layers.conv2d_transpose(output, layer, kernel_size,
                                                        (pool_size, pool_size),
                                                        padding='SAME',
                                                        activation=tf.nn.relu)
                    output = tf.nn.dropout(output, self.dropout_conv)
                    # print(output)

                image_size = output.get_shape().as_list()[1]
                # print(image_size)

            with tf.variable_scope('output'):   #마지막은 활성함수가 없어서 따로 만들어 놓음
                output = tf.layers.conv2d_transpose(output, 3, kernel_size, (2, 2),
                                                    padding='SAME')

        y_ = output
        # print(y_)
        return y_

    def train(self):
        #model
        self.y_en, size, reshape_size = self.encode(self.x)
        self.y_de = self.decode(self.y_en, size, reshape_size)
        print(self.y_de)
        print(self.y)

        #cost and training
        with tf.name_scope("cost"):
            self.cost = tf.reduce_sum(tf.pow(self.y_de - self.y, 2))
            # self.cost = tf.reduce_mean( -tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.y_de, 1e-5, 1e10))))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.cost, global_step=self.global_step)

            tf.summary.scalar("cost", self.cost)
        #
        # with tf.name_scope("accuracy"):
        #     compare = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_de, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(compare, "float"))
        #
        #     tf.summary.scalar("accuarcy", self.accuracy)


if __name__ == '__main__':
    net = Network()
