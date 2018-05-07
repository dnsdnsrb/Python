import tensorflow as tf
import gym

from gym.envs.registration import register


env = gym.make('FrozenLake-v0')
init_state = env.reset()

class Network():
    def __init__(self, name, x_size, y_size):
        self.name = name
        self.x_size = x_size
        self.y_size = y_size
        self.x = tf.placeholder(tf.float32, [None, x_size])
        self.y = tf.placeholder(tf.float32, [None, y_size])

        self.global_step = tf.Variable(0, trainable=False)

    def model_cleaner(self, x, layers=[50, 50], reuse=False):
        #AutoEncoder
        #입력 : 현재 상황
        #출력 : 현재 상황'
        with tf.variable_scope(self.name + "model", reuse=reuse):
            output = x

            for i, layer in enumerate(layers):
                with tf.variable_scope('dense' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('dense_out'):
                output = tf.layers.dense(output, self.y_size, activation=None)

            y = output
            return y

    def model_distrupter(self, x, layers=[50, 50], reuse=False):
        #Dense
        #입력 : cleaner의 가중치들
        #출력 : 이번에 할 행동

        with tf.variable_scope(self.name + "model", reuse=reuse):
            output = x

            for i, layer in enumerate(layers):
                with tf.variable_scope('dense' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('dense_out'):
                output = tf.layers.dense(output, self.y_size, activation=None)

            y = output
            return y

    def train_cleaner(self):
        #loss = 현재 상황 - 현재 상황'
        self.x_ = self.model(self.x)

        with tf.name_scope(self.name + "cost"):
            self.loss = tf.reduce_sum(tf.square(self.x_ - self.x))
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate). \
                minimize(self.loss, global_step=self.global_step)

    def train_distrupter(self):
        #loss = action-cleaner_loss - max(actions-cleaner_loss) = 현재 행동의 loss - 가능한 행동의 가장 높은 loss
        #max(actions-cleaner_loss)가 계속 변한다고 생각할 수 있으나, cleaner의 가중치를 고려하기 때문에 실제론 그렇지 않다.
        self.y_ = self.model(self.x)

        with tf.name_scope(self.name + "cost"):
            # self.loss = tf.reduce_sum(tf.square(self.y_ - self.y))
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate). \
                minimize(self.loss, global_step=self.global_step)

    def predict(self, session, x1, x2):      #model 전체 실행
        x = self.concat(x1, x2)

        act = session.run(self.y_, feed_dict={self.x: x})
        return act