import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque

class Network():    #env net이 될수도, obj net이 될수도, env net이 될수도 있다.
    buffer = deque()
    buffer_max = 50000
    def __init__(self, name, session, x_size, y_size, z_size):
        with tf.variable_scope(name):
            self.name = name
            self.session = session

            self.learning_rate = 0.0001
            self.global_step = tf.Variable(0, trainable=False)

            self.x_size = x_size
            self.y_size = y_size
            self.z_size = z_size
            #
            self.x = tf.placeholder(tf.float32, [None, x_size])
            self.y = tf.placeholder(tf.float32, [None, y_size])
            self.z = tf.placeholder(tf.float32, [None, z_size])
            # self.model = tf.placeholder(tf.int32)

            self.train()

    def model(self, x, layers=[50, 50], reuse=False):
        with tf.variable_scope("present2predict", reuse=reuse):
            output = x
            for i, layer in enumerate(layers):
                with tf.variable_scope('denseLayer' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('outputLayer'):
                output = tf.layers.dense(output, self.y_size, activation=None)

            y = output
            return y

    def model2(self, y, layers=[50, 50], reuse=False):
        with tf.variable_scope("predict2act", reuse=reuse):
            output = y

            for i, layer in enumerate(layers):
                with tf.variable_scope('denseLayer' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('outputLayer'):
                output = tf.layers.dense(output, self.z_size, activation=None)

            z = output
            return z

    def train(self):
        self.y_ = self.model(self.x)

        with tf.name_scope("lossObserve"):
            self.loss1 = tf.reduce_sum(tf.square(self.y_ - self.y))
            self.opt1 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).\
                minimize(self.loss1, global_step=self.global_step)

        self.z_ = self.model2(self.y_)

        with tf.name_scope("lossAct"):
            self.loss2 = tf.reduce_sum(tf.square(self.z_ - self.z))
            self.opt2 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).\
                minimize(self.loss2, global_step=self.global_step)

    def predictState(self, x1, x2):      #model 전체 실행
        x = self.concat(x1, x2)

        act = self.session.run(self.y_, feed_dict={self.x: x})
        return act

    def predictAct(self, x):

        act = self.session.run(self.z_, feed_dict={self.x: x})
        return act

    def concat(self, x1, x2):
        try:
            x = np.concatenate([x1, x2])
            x = np.reshape(x, [1, self.x_size])
        except:
            x1 = self.reshape(x1)
            x2 = self.reshape(x2)

            x = np.concatenate([x1, x2], axis=1)    #이 경우, axis 0은 배치이므로, axis 1을 합쳐야함
            x = np.reshape(x, [len(x1), self.x_size])

        return x

    def reshape(self, x):
        try:
            x = np.reshape(x, [1, len(x)])
        except:
            pass

        return x

    def updateState(self, statePresent, stateNext):   #model 전체 실행 후, opt까지 작동
        stateNext = self.reshape(stateNext)

        loss, opt = self.session.run([self.loss1, self.opt1], feed_dict={self.x: statePresent, self.y: stateNext})
        return loss

    def updateAct(self, statePresent, act):   #model 전체 실행 후, opt까지 작동
        act = self.reshape(act)

        loss, opt = self.session.run([self.loss2, self.opt2], feed_dict={self.x: statePresent, self.z: act})
        return loss

class State:
    def __init__(self, present=[], done=False):
        self.present = present
        self.next = []
        self.action = -1
        self.reward = 0
        self.done = done

EPISODE_MAX = 1000
STEP_MAX = 1500

if __name__ == '__main__':
    with tf.Session() as sess:
        env = gym.make('CartPole-v1')
        init_state = env.reset()

        num_epi = 100000

        batch = 50
        env_size = env.observation_space.shape[0]
        act_size = env.action_space.n

        net = Network("main", sess, env_size, env_size, act_size)
        state = State()

        observe = True
        observe_max = int(EPISODE_MAX * 0.5)


        for episode in range(EPISODE_MAX):
            step = 0
            while not state.done:
                step += 1
                state.present = env.reset()

                if observe:
                    state.action = env.action_space.sample()
                else:
                    state.action = net.predictAct(state.present)

                state.action, state.reward, state.done, _ = env.step(state.action)

                # if state.done:

                net.buffer.append(state)  #exp_replay를 위해 buffer 넣어둠
                if len(net.buffer) > net.buffer_max:
                    net.buffer.popleft()

                state = State(state.next, state.done)

                if step > STEP_MAX:
                    break

                if episode % 100 == 0:
                    env.render()


            if episode < observe_max:
                state_stack = np.empty(0).reshape(0, env_size)
                act_stack = np.empty(0).reshape(0, act_size)
                minibatch = random.sample(net.buffer.present, 10)

                net.updateState(net, present, next)
            else:
                net.updateAct(net, present, act)
