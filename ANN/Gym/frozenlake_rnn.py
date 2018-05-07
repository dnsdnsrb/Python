import tensorflow as tf
import gym
import numpy as np
import random
import time
from collections import deque
import matplotlib.pyplot as plt
from gym.envs.registration import register

env = gym.make('FrozenLake-v0')
init_state = env.reset()


temp = 0  # 테스트용 전역변수


class Network():  # env net이 될수도, obj net이 될수도, env net이 될수도 있다.
    replay_buffer = 100

    def __init__(self, name, x_size, y_size, lr=0.1):
        self.name = name
        self.global_step = tf.Variable(0, trainable=False)

        self.x_size = x_size
        self.y_size = y_size
        #
        self.max_length = 100
        self.x = tf.placeholder(tf.float32, [None, self.max_length, x_size])
        self.y = tf.placeholder(tf.float32, [None, y_size])

        self.lr = lr
        # self.model = tf.placeholder(tf.int32)

        self.train()

    def lstm(self, num, activation='relu'):
        if activation == 'relu':
            return tf.nn.rnn_cell.BasicLSTMCell(num, activation=tf.nn.relu)
        elif activation == 'none':
            return tf.nn.rnn_cell.BasicLSTMCell(num, activation=None)

    def length(self, x):
        used = tf.sign(tf.reduce_max(tf.abs(x), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def cost(self, y_, y):
        print(y.shape, y_.shape)
        cross_entropy = -tf.reduce_sum(y * tf.log(y_), 2)

        mask = tf.sign(tf.reduce_max(tf.abs(y), 2))

        cross_entropy = tf.reduce_sum(cross_entropy, 1)
        cross_entropy /= tf.reduce_sum(mask, 1)

        return tf.reduce_mean(cross_entropy)


    def model(self, x, layers=[50, 50], reuse=False):
        x = tf.reshape(x, [1, 1, self.x_size])
        with tf.variable_scope(self.name + "model", reuse=reuse):
            output = x

            # for i, layer in enumerate(layers):
            #     with tf.variable_scope('rnn' + str(i)):
            #         cell = tf.nn.rnn_cell.BasicLSTMCell(i, activation=tf.nn.relu)
            #         output, state = tf.nn.dynamic_rnn(cell, output, dtype=tf.float32)
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm(i, 'relu') for i in layers])
            output, state = tf.nn.dynamic_rnn(cell, output, dtype=tf.float32, sequence_length=self.length(output))

            with tf.variable_scope('fc'):
                output = tf.layers.dense(output, self.y_size, activation=None)

            y = output
            return y

    def train(self):
        self.y_ = self.model(self.x)

        with tf.name_scope(self.name + "cost"):
            # self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
            self.loss = tf.reduce_sum(tf.square(self.y_ - self.y))
            # self.loss = self.cost(self.y_, self.y)
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr). \
                minimize(self.loss, global_step=self.global_step)

    def predict(self, session, x):  # model 전체 실행
        # x = self.concat(x1, x2)

        act = session.run(self.y_, feed_dict={self.x: x})
        return act

    # def concat(self, x1, x2):
    #     try:
    #         x = np.concatenate([x1, x2])
    #         x = np.reshape(x, [1, self.x_size])
    #     except:
    #         x1 = self.reshape(x1)
    #         x2 = self.reshape(x2)
    #
    #         x = np.concatenate([x1, x2], axis=1)  # 이 경우, axis 0은 배치이므로, axis 1을 합쳐야함
    #         x = np.reshape(x, [len(x1), self.x_size])
    #
    #     return x

    # def reshape(self, x):
    #     try:
    #         x = np.reshape(x, [1, len(x)])
    #     except:
    #         pass
    #
    #     return x

    def update(self, session, x, y):  # model 전체 실행 후, opt까지 작동
        # x = self.concat(x1, x2)
        # y = self.reshape(y)

        # ir = session.run([self.irs], feed_dict={self.env_now: state, self.ir: action})
        loss, opt = session.run([self.loss, self.opt], feed_dict={self.x: x, self.y: y})
        return loss

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

def show(sess, anet):
    table = np.identity(16)
    act = np.identity(4)

    obj = np.zeros_like(table)
    obj[:, 15] = 1
    # print(obj)
    np.set_printoptions(suppress=True)
    print("act")
    print(anet.predict(sess, table, obj))

def onehot(x, size):
    onehot = np.zeros(size)
    onehot[x] = 1
    return onehot


def stack(sess, batch, obj, anet, success, size=16):
    now_stack = np.empty(0).reshape(0, size)
    obj_stack = np.empty(0).reshape(0, size)
    act_stack = np.empty(0).reshape(0, anet.y_size)

    for now, act in batch:
        # print(act)
        if success:
            act[np.argmax(act)] = 1
        else:
            act[np.argmax(act)] = 0

        now_stack = np.vstack([now_stack, now])
        obj_stack = np.vstack([obj_stack, obj])
        act_stack = np.vstack([act_stack, act])

    now_stack = np.expand_dims(now_stack, axis=0)
    obj_stack = np.expand_dims(obj_stack, axis=0)
    act_stack = np.expand_dims(act_stack, axis=0)

    loss = anet.update(sess, now_stack, obj_stack, act_stack)

    return loss

def buffer_append(buffer, env):
    try:
        buffer = np.append(buffer, env, axis=1)
    except:
        buffer = np.append(buffer, env, axis=2)
    return buffer

if __name__ == '__main__':
    num_epi = 2000
    state_size = env.observation_space.n
    env_size = state_size  # state + old + dead

    act_size = env.action_space.n
    anet = Network("act", env_size, act_size, 0.1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        batch_size = 50
        rList = []

        obj = np.zeros(env_size)
        obj[env_size - 1] = 1

        for episode in range(num_epi):
            now = env.reset()
            now = onehot(now, env_size)
            now = now.reshape(1, 1, 16)
            rAll = 0

            done = False
            step_count = 0

            # init_act = np.array([1, 0, 0, 0])

            buffer = now
            stack_x = np.empty(0).reshape(0, 1, env_size)
            stack_y = np.empty(0).reshape(0, act_size)
            while not done:
                # 행동 및 결과 관찰
                # random.shuffle(init_act)
                # act_onehot = init_act
                act_onehot = anet.predict(sess, buffer)[0]

                # 행동 후처리
                act = np.argmax(act_onehot)

                # 환경에 적용
                state, reward, done, _ = env.step(act)
                # x = np.concatenate([new, act])

                # 환경 후처리
                state = onehot(state, env_size)
                state = env.reshape([1, 1, env_size])

                # # 행동 후처리
                # act = np.zeros(act_size)
                # act[y] = 1

                # hole에 빠질 경우, 움직일 수 없는 상태 데이터 생성
                # if done == True and reward == 0:
                #     buffer_append(buffer, state)
                #
                #     # buffer.append((now, act_onehot))
                #     # now = new
                #     random.shuffle(act_onehot)


                # replay buffer 추가 관련
                buffer_append(buffer, state)
                # buffer.append((now, act_onehot))
                # 시각화 및 확인
                # if episode % 100 == 0:
                #     env.render()

                # 다음 루프 관련
                # now = new
                rAll += reward

            #결과를 보고 y값 생성, 해냈다면 1을 줘서 추구하게 하고, 아니면 0을 줘서 다른 선택을 하게 만든다.
            if rAll >= 1:
                state[np.argmax(state)] = 1

            else:
                state[np.argmax(state)] = 0
            y = state
            # y = y.reshape( 1, act_size)
            #결과에 따른 act 수정, batch 데이터를 vstack으로 변형하여 학습
            stack_x = np.vstack([stack_x, buffer])
            stack_y = np.vstack([stack_y, y])
            loss = 0
            # if rAll >= 1:
            #     loss = anet.update(sess, now_stack, obj_stack, act_stack)
            # else:
            #     loss = stack(sess, replay_buffer, obj, anet, False)

            # 100회마다 결과 확인
            if episode % 50 == 0:
                print(stack_x.shape, stack_y.shape)
                loss = anet.update(sess, stack_x, stack_y)
                print("Ep : ", episode, "steps : ", step_count)

            rList.append(rAll)
        show(sess, anet)
        print("Success rate: " + str(sum(rList) / num_epi))
        plt.bar(range(len(rList)), rList, color="blue")
        plt.show()
