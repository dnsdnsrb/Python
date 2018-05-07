import gym
import numpy as np
import random
import matplotlib.pyplot as plt

from collections import deque
from gym.envs.registration import register
# import matplotlib.pyplot as plt
# import time
# import os
# from gym import wrappers

import tensorflow as tf

#이 dqn은 오차함수에 같은 망을 이용(제일 최근은 2개의 망을 사용)
#tf.assign(target_weight, main_weight)

def copy_weights(source, target):
    weights = []
    sources = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=source)
    targets = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target)


    for source, target in zip(sources, targets):
        weights.append(tf.assign(target, source.value()))

    return weights

def onehot(x, size):
    onehot = np.zeros(size)
    onehot[x] = 1
    return onehot

class DQN:
    def __init__(self, inputSize, outputSize, name='net'):
        with tf.variable_scope(name):
            self.input_size = inputSize
            self.output_size = outputSize
            self.learning_rate = 0.1
            self.x = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
            self.y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            self.discount = 0.99
            self.replay_memory = 50000
            self.rewardList = []

            self.train()

    def model(self, x, layers=[50, 50]):
        output = x

        for i, layer in enumerate(layers):
            with tf.variable_scope('DenseLayer' + str(i)):
                output = tf.layers.dense(output, layer, activation=tf.nn.tanh)

        with tf.variable_scope('OutputLayer'):
            output = tf.layers.dense(output, self.output_size)

        y = output

        return y

    def train(self):
        self.y_ = self.model(self.x)
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))
        self.loss = tf.reduce_sum(tf.square(self.y - self.y_))
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def update(self, session, state, action):
        return session.run([self.loss, self.opt], feed_dict={self.x: state, self.y: action})

    def predict(self, session, state, target=None):
        if target is not None:
            state = np.concatenate((state, target))

        state = np.reshape(state, [1, self.input_size])
        return session.run(self.y_, feed_dict={self.x: state})

    def exp_replay(self, sess, targetDQN, batch):  #
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)

        for state in batch:    #batch를 만들어냄
            x = np.concatenate((state.present, state.target))
            q = self.predict(sess, x)
            if state.done:
                q[0, state.action] = state.reward
            else:
                q[0, state.action] = state.reward + self.discount * \
                                     np.max(targetDQN.predict(sess, state.next, state.target))
            x_stack = np.vstack([x_stack, x])  # state를 쌓음
            y_stack = np.vstack([y_stack, q])      # q를 쌓음

        return self.update(sess, x_stack, y_stack)  # 쌓은 걸로 학습시킴.

class State:
    def __init__(self, present=None, done=False):
        self.present = present
        self.target = []
        self.next = []
        self.action = -1
        self.reward = 0
        self.done = done

STEP_MAX = 100
EPISODE_MAX = 2000

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    with tf.Session() as sess:
        rList = []

        inputSize = env.observation_space.n
        outputSize = env.action_space.n
        net = DQN(inputSize * 2, outputSize, name='main')
        target = DQN(inputSize * 2, outputSize, name='target')

        tf.global_variables_initializer().run()
        replay_buffer = deque()

        weights = copy_weights('main', 'target')
        sess.run(weights)

        # 무작위 목표를 할 때는 이 부분을 for문에 넣고 수정해야 한다.
        stateTarget = np.zeros((16))
        stateTarget[15] = 1

        for episode in range(EPISODE_MAX):
            rAll = 0
            state = State()
            # reset parameters
            state.present = onehot(env.reset(), inputSize)
            state.done = False

            step = 0
            while not state.done:
                state.target = stateTarget
                state.action = np.argmax(net.predict(sess, state.present, state.target) + \
                                   np.random.randn(1, outputSize) / (episode + 1))

                state.next, state.reward, state.done, _ = env.step(state.action)
                state.next = onehot(state.next, inputSize)

                if np.array_equal(state.next, state.target):
                    state.reward = 1
                if state.done and state.reward == 0:    #실패 시
                    state.reward = -1

                replay_buffer.append(state)  #exp_replay를 위해 buffer 넣어둠
                if len(replay_buffer) > net.replay_memory:
                    replay_buffer.popleft()

                step += 1 #게임에서 버틴 시간을 의미
                if episode % 100 == 0:
                    env.render()
                if state.reward == -1:
                    pass
                else:
                    rAll += state.reward

                state = State(state.next, state.done)
            rList.append(rAll)

            print("Ep : ", episode, "steps : ", step)
            if step > STEP_MAX:
                pass

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, min(step, 10))
                    loss, _ = net.exp_replay(sess, target, minibatch)
                print("Loss : ", loss)
                sess.run(weights)
        #wrapper는 win10에서는 안 되는 듯
        print("Success rate: " + str(sum(rList) / EPISODE_MAX))
        plt.bar(range(len(rList)), rList, color="blue")
        plt.show()
        # net.play(sess, env)