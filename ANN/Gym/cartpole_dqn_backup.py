import gym
import numpy as np
import random

from collections import deque
from gym.envs.registration import register
# import matplotlib.pyplot as plt
# import time
# import os
# from gym import wrappers

import tensorflow as tf

#이 dqn은 오차함수에 같은 망을 이용(제일 최근은 2개의 망을 사용)
#tf.assign(target_weight, main_weight)
# register(
#     id='CartPole-v2',
#     entry_point='gym.envs.classic_control:CartPoleEnv',
#     tags={'wrapper_config.TimeLimit.max_episode_steps':10000},
#     reward_threshold=10000.0,
# )

env = gym.make('CartPole-v0')

def fc_layer(x, output_num, activation='none', dropout=None):
    #random_uniform_initializer보다 random_uniform가 더 잘 나옴. => 이유는 모름 seed가 다른가?
    w = tf.get_variable(name='weight',
                        shape=[x.get_shape().as_list()[1], output_num],
                        initializer=tf.contrib.layers.xavier_initializer())
    # w =  tf.Variable(tf.random_uniform([x.get_shape().as_list()[1], output_num], 0, 0.01))

    y = tf.matmul(x, w)

    if activation == 'tanh':
        y = tf.nn.tanh(y)
    elif activation == 'none':
        pass

    return y

def copy_weights(source, target):
    weights = []
    sources = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=source)
    targets = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target)


    for source, target in zip(sources, targets):
        weights.append(tf.assign(target, source.value()))

    return weights

class DQN:
    def __init__(self, name='net'):
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.learning_rate = 0.1
        self.x = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self.discount = 0.9
        self.replay_memory = 50000
        self.rewardList = []
        self.name = name

        self.train()

    def model(self, x, layer=[50]):
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
        self.loss = tf.reduce_sum(tf.square(self.y - self.y_))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def update(self, session, state, action):
        return session.run([self.loss, self.opt], feed_dict={self.x: state, self.y: action})

    def predict(self, session, state):
        state = np.reshape(state, [1, self.input_size])
        return session.run(self.y_, feed_dict={self.x: state})

    def exp_replay(self, sess, targetDQN, batch):  #
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)

        for state, action, reward, state_new, done in batch:    #batch를 만들어냄
            q = self.predict(sess, state)

            if done:
                q[0, action] = reward
            else:
                # q[0, action] = reward + self.discount * \
                #                         targetDQN.predict(sess, state_new)[0, np.argmax(self.predict(sess, state_new))]
                # print("q", "action", q[0, action], action)
                # print("q1 = ", q[0, action])
                q[0, action] = reward + self.discount * np.max(targetDQN.predict(sess, state_new))
                print("q2 = ", q[0, action])
            x_stack = np.vstack([x_stack, state])  # state를 쌓음
            y_stack = np.vstack([y_stack, q])      # q를 쌓음

        return self.update(sess, x_stack, y_stack)  # 쌓은 걸로 학습시킴.

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

def run():
    with tf.Session() as sess:
        step_limit = 1500
        num_episodes = 5000

        net = DQN(name='main')
        target = DQN(name='target')

        tf.global_variables_initializer().run()
        replay_buffer = deque()

        weights = copy_weights('main', 'target')
        sess.run(weights)

        for episode in range(num_episodes):
            #reset parameters
            state = env.reset()
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
                print("state_new = ", state_new)
                if done:    #실패 시
                    reward = -100

                replay_buffer.append((state, action, reward, state_new, done))  #exp_replay를 위해 buffer 넣어둠
                if len(replay_buffer) > net.replay_memory:
                    replay_buffer.popleft()

                state = state_new
                step_count += 1 #게임에서 버틴 시간을 의미
                if step_count > step_limit:
                    break

                if episode % 100 == 0:
                    env.render()

            print("Ep : ", episode, "steps : ", step_count)
            if step_count > step_limit:
                pass

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = net.exp_replay(sess, target, minibatch)
                print("Loss : ", loss)
                sess.run(weights)
        #wrapper는 win10에서는 안 되는 듯
        net.play(sess, env)


run()