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
env = env.unwrapped


def denseLayer(x, output_num, activation='none', name='denseLayer'):
    #random_uniform_initializer보다 random_uniform가 더 잘 나옴. => 이유는 모름 seed가 다른가?
    w = tf.get_variable(name=name + 'weight',
                        shape=[x.get_shape().as_list()[1], output_num],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name=name + 'bias',
                        shape=output_num,
                        initializer=tf.zeros_initializer())

    # w =  tf.Variable(tf.random_uniform([x.get_shape().as_list()[1], output_num], 0, 0.01))

    y = tf.matmul(x, w) + b

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
        with tf.variable_scope(name):
            self.input_size = env.observation_space.shape[0]
            self.output_size = env.action_space.n
            self.learning_rate = 0.01
            self.x = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
            self.y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            self.discount = 0.9
            self.replay_memory = 50000
            self.rewardList = []
            # self.name = name

            self.train()

    def model(self, x, layers=[100, 100, 100]):
        #어이가 없네
        #tf.layers.dense로 하니까, scope 문제로 weight 복사가 안됨.
        #해결 => init에 variable scope를 넣으니 됨. (A3C 예에서 보고 배움)
        output = x

        for i, layer in enumerate(layers):
            output = tf.layers.dense(output, layer, tf.nn.relu, name='denseLayer' + str(i))
                # output = denseLayer(output, layer, activation='tanh', name='denseLayer' + str(i))

        output = tf.layers.dense(output, self.output_size, name='outputLayer')
        # output = denseLayer(output, self.output_size, name='outputLayer')

        y = output

        return y

    def train(self):
        self.y_ = self.model(self.x)
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def update(self, session, state, action):
        return session.run([self.loss, self.opt], feed_dict={self.x: state, self.y: action})

    def predict(self, session, state):
        state = np.reshape(state, [1, self.input_size])
        return session.run(self.y_, feed_dict={self.x: state})

    def exp_replay(self, sess, targetDQN, batch):  #
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)

        for state in batch:    #batch를 만들어냄
            q = self.predict(sess, state.present)

            if state.done:
                q[0, state.action] = state.reward
            else:
                q[0, state.action] = state.reward + self.discount * np.max(targetDQN.predict(sess, state.next))
                # print("q", q[0, state.action])
            x_stack = np.vstack([x_stack, state.present])  # state를 쌓음
            y_stack = np.vstack([y_stack, q])      # q를 쌓음

        return self.update(sess, x_stack, y_stack)  # 쌓은 걸로 학습시킴.

def exploreAndExploit(episode, net, sess, state):
    action = np.argmax(net.predict(sess, state) + \
                       np.random.randn(1, env.action_space.n) / (episode + 1))

    return action

class State:
    def __init__(self, present=[], done=False):
        self.present = present
        self.next = []
        self.action = -1
        self.reward = 0
        self.done = done

if __name__ == '__main__':
    # 알고리즘
    # state에 따른 action을 생성
    # action을 실행해봄
    # 결과를 buffer에 저장
    # episode가 끝날 때까지, 위를 반복
    # episode가 끝난 후 학습
    # 위를 반복
    with tf.Session() as sess:
        step_limit = 1000
        num_episodes = 5000

        net = DQN(name='main')
        target = DQN(name='target')

        tf.global_variables_initializer().run()
        replay_buffer = deque()

        weights = copy_weights('main', 'target')
        sess.run(weights)

        stateTarget = np.array([0, 0, 0, 0])


        for episode in range(num_episodes):
            #reset parameters
            state = State()
            state.present = env.reset()
            state.done = False
            step = 0

            while not state.done:
                step += 1
                state.action = exploreAndExploit(episode, net, sess, state.present)
                state.next, state.reward, state.done, _ = env.step(state.action)

                # 0에 가까울 수록 점수가 높도록 해봐 아니면 0이면 1 그 외에는 점차 낮게 하던가
                if state.next[0] == 0 and state.next[2] == 0:  # 걍 이래도 됨
                    reward = 1
                else:
                    reward = 0
                if state.done:    #실패 시
                    reward = -1

                #buffer 관리
                replay_buffer.append(state)  #exp_replay를 위해 buffer 넣어둠
                if len(replay_buffer) > net.replay_memory:
                    replay_buffer.popleft()

                state = State(state.next, state.done)
                # state.present = state.next
                # print("here = ", state.present, state.next)


                # step_count += 1 #게임에서 버틴 시간을 의미
                if step > step_limit:
                    break

                if episode % 100 == 0:
                    env.render()

            print("Ep : ", episode, "step : ", step)
            if step > step_limit:
                pass

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = net.exp_replay(sess, target, minibatch)
                print("Loss : ", loss)
                sess.run(weights)
        # #wrapper는 win10에서는 안 되는 듯
        # net.play(sess, env)