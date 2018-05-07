import tensorflow as tf

class Network():
    def __init__(self, image=[128, 128, 3], actions=20, ir=250):
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        #image
        self.im_size = [image[0], image[1], image[2]]

        #inner representation
        self.act_num = actions  #행동가능 경우의 수
        self.ir_num = ir            #inner representation, 내부 표현


        self.env = tf.placeholder(tf.float32)   #환경
        self.obj = tf.placeholder(tf.float32)   #목표 환경
        self.ir_ = tf.placeholder(tf.float32)   #결과
        self.ir = tf.placeholder(tf.float32)  # 결과

        self.dropout_fc = tf.placeholder("float")
        self.dropout_conv = tf.placeholder("float")

    def model_img(self, img, conv_layers = [12, 12, 12, 12], fc_layers = [1000, 500], kernel = [5, 5, 5, 5]):
        image_size = self.im_size[0]
        output = img

        with tf.variable_scope('img'):

            for i, layer in enumerate(conv_layers):                         #Conv layers 구축부분
                with tf.variable_scope('conv' + str(i)):
                    output = tf.layers.conv2d(output, layer, kernel[0], padding='SAME', activation=tf.nn.relu)
                    output = tf.nn.dropout(output, self.dropout_conv)

                image_size = output.get_shape().as_list()[1]

            with tf.name_scope('reshape'):
                reshape_size = image_size * image_size * conv_layers[-1]    #-1은 제일 끝을 나타낸다.
                output = tf.reshape(output, [-1, reshape_size])

            for i, layer in enumerate(fc_layers):                           #fc layers 구축 부분
                with tf.variable_scope('fc' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)
                    output = tf.nn.dropout(output, self.dropout_fc)

            with tf.variable_scope('fc'):
                output = tf.layers.dense(output, self.ir, activation=None)

        ir = output
        return ir

    def model_obj(self, img, conv_layers = [12, 12, 12, 12], fc_layers = [1000, 500], kernel = [5, 5, 5, 5]):
        image_size = self.im_size[0]
        output = img

        with tf.variable_scope('img'):

            for i, layer in enumerate(conv_layers):                         #Conv layers 구축부분
                with tf.variable_scope('conv' + str(i)):
                    output = tf.layers.conv2d(output, layer, kernel[0], padding='SAME', activation=tf.nn.relu)
                    output = tf.nn.dropout(output, self.dropout_conv)

                image_size = output.get_shape().as_list()[1]

            with tf.name_scope('reshape'):
                reshape_size = image_size * image_size * conv_layers[-1]    #-1은 제일 끝을 나타낸다.
                output = tf.reshape(output, [-1, reshape_size])

            for i, layer in enumerate(fc_layers):                           #fc layers 구축 부분
                with tf.variable_scope('fc' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)
                    output = tf.nn.dropout(output, self.dropout_fc)

            with tf.variable_scope('fc'):
                output = tf.layers.dense(output, self.ir, activation=None)

        ir = output
        return ir

    def model_ir(self, ir_concat, layers=[300, 200, 100]):

        with tf.variable_scope('ir'):
            output = ir_concat

            for i, layer in enumerate(layers):
                with tf.variable_scope('fc_en' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_en'):
                output = tf.layers.dense(output, self.act_num, activation=None)

            act = output  #actions

            for i, layer in enumerate(reversed(layers)):
                with tf.variable_scope('fc_de' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_de'):
                output = tf.layers.dense(output, self.ir, activation=None)  #예측 상황 1개만 낸다.(완전한 AE는 아닌셈)

            ir_ = output #ir

            return ir_, act

    def model(self, img, obj):
        ir1 = self.model_img(img)
        ir2 = self.model_obj(obj)

        ir_concat = tf.concat([ir1, ir2], 0)

        ir_, act = self.model_ir(ir_concat)

        return ir_, act

    def train(self):

        ir_, act = self.model(self.env, self.obj)

        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.square(ir_ - self.ir))
            # self.cost = tf.reduce_mean( -tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.y_de, 1e-5, 1e10))))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.cost, global_step=self.global_step)

