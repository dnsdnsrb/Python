import tensorflow as tf

class Network():
    def __init__(self, actions=20, ir=50):
        # self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.actions = actions  #행동가능 경우의 수
        self.ir = ir            #inner representation, 내부 표현

        self.x = tf.placeholder(tf.float32, [None, ir*2])   #환경과 목표 2개가 같이 들어오기때문
        self.y = tf.placeholder(tf.float32)     #예상 환경
        self.y_obj = tf.placeholder(tf.float32) #목표 환경

        self.dropout_normal = tf.placeholder("float")

        self.train()
    #Ir -> act -> Ir => 학습을 위해서임.
    def model(self, x, layers=[300, 200, 100]):
        with tf.variable_scope('ir'):
            output = x

            for i, layer in enumerate(layers):
                with tf.variable_scope('fc_en' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_en'):
                output = tf.layers.dense(output, self.actions, activation=None)

            z = output  #actions

            for i, layer in enumerate(reversed(layers)):
                with tf.variable_scope('fc_de' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_de'):
                output = tf.layers.dense(output, self.ir, activation=None)  #예측 상황 1개만 낸다.(완전한 AE는 아닌셈)

            y_ = output #ir

            return y_, z

    #학습을 위해선 2번 돌려야함. 1 => act 생성(z), 2 => 예측환경과 act로 나온 환경, 예측환경과 목표환경 비교(opt)
    def train(self):
        #model
        self.y_, self.z = self.model(self.x)

        #cost and training
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_, labels=self.y)
                                       + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y, labels=self.y_obj))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.cost, global_step=self.global_step)

            tf.summary.scalar("cost", self.cost)

if __name__ == '__main__':
    net = Network()