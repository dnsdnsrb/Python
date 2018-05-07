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


env = gym.make('FrozenLake-v0')

def fc_layer(x, output_num, activation='none', dropout=None):
    #random_uniform_initializer보다 random_uniform가 더 잘 나옴. => 이유는 모름 seed가 다른가?
    w = tf.get_variable(name='weight',
                        shape=[x.get_shape().as_list()[1], output_num],
                        initializer=tf.contrib.layers.xavier_initializer())
    # w =  tf.Variable(tf.random_uniform([x.get_shape().as_list()[1], output_num], 0, 0.01))

    y = tf.matmul(x, w)

    if activation == 'tanh':
        y = tf.nn.tanh(y)
    elif activation == 'relu':
        y = tf.nn.relu(y)
    elif activation == 'none':
        pass

    return y

def onehot(x, size):
    onehot = np.zeros(size)
    onehot[x] = 1
    return onehot

class Network:
    def __init__(self, name='net'):
        self.input_size = env.observation_space.n
        self.output_size = env.action_space.n
        self.learning_rate = 0.1
        self.x = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self.discount = 0.99
        self.replay_memory = 50000
        self.rewardList = []
        self.name = name

        self.train()

    def model(self, x, layer=[50, 50]):
        output = x

        layer_num = 0

        with tf.variable_scope(self.name):
            for i in layer:
                layer_num += 1
                with tf.variable_scope('fc' + str(layer_num)):
                    output = fc_layer(output, i, activation='tanh')

            layer_num += 1
            with tf.variable_scope('fc' + str(layer_num)):
                output = fc_layer(output, self.output_size)

        y = output

        return y

    def train(self):
        self.y_ = self.model(self.x)
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_, labels=self.y))
        # self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.loss = tf.reduce_sum(tf.square(self.y - self.y_))
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def update(self, session, state, action):
        return session.run([self.loss, self.opt], feed_dict={self.x: state, self.y: action})

    def predict(self, session, state):
        state = np.reshape(state, [1, self.input_size])
        return session.run(self.y_, feed_dict={self.x: state})



    def play(self, sess, env):
        state = env.reset()
        reward_sum = 0

        while True:
            env.render()
            action = np.argmax(self.predict(sess, state))

            state, reward, done, _ = env.step(action)
            reward_sum += reward

            if done:
                print("Total : ", reward_sum)
                break

if __name__ == '__main__':
    with tf.Session() as sess:
        step_limit = 1500
        num_episodes = 2000
        rList = []

        net = DQN(name='main')
        target = DQN(name='target')

        tf.global_variables_initializer().run()
        replay_buffer = deque()

        weights = copy_weights('main', 'target')
        sess.run(weights)

        for episode in range(num_episodes):
            rAll = 0
            #reset parameters
            state = onehot(env.reset(), net.input_size)
            e = 1. / ((episode / 10) + 1)    #e-greedy
            done = False
            step_count = 0

            while not done:
                # if np.random.rand(1) < e:   #explore & exploit
                #     action = env.action_space.sample()
                # else:
                #     action = np.argmax(net.predict(sess, state))
                action = np.argmax(net.predict(sess, state) + \
                                   np.random.randn(1, env.action_space.n) / (episode + 1))

                state_new, reward, done, _ = env.step(action)
                state_new = onehot(state_new, net.input_size)
                # action = onehot(action, net.output_size)

                if done and reward == 0:    #실패 시
                    reward = -1

                replay_buffer.append((state, action, reward, state_new, done))  #exp_replay를 위해 buffer 넣어둠
                if len(replay_buffer) > net.replay_memory:
                    replay_buffer.popleft()

                state = state_new
                step_count += 1 #게임에서 버틴 시간을 의미
                if step_count > step_limit:
                    break

                if episode % 100 == 0:
                    env.render()
                if reward == -1:
                    pass
                else:
                    rAll += reward
            rList.append(rAll)

            print("Ep : ", episode, "steps : ", step_count)
            if step_count > step_limit:
                pass

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, min(step_count, 10))
                    loss, _ = net.exp_replay(sess, target, minibatch)
                print("Loss : ", loss)
                sess.run(weights)
        #wrapper는 win10에서는 안 되는 듯
        print("Success rate: " + str(sum(rList) / num_episodes))
        plt.bar(range(len(rList)), rList, color="blue")
        plt.show()
        # net.play(sess, env)