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

    def __init__(self, name, x_size, y_size):
        self.name = name
        self.global_step = tf.Variable(0, trainable=False)

        self.x_size = x_size
        self.y_size = y_size
        #
        self.x = tf.placeholder(tf.float32, [None, x_size])
        self.y = tf.placeholder(tf.float32, [None, y_size])
        self.lr = tf.placeholder(tf.float32)
        # self.model = tf.placeholder(tf.int32)

        self.train()

    def model(self, x, layers=[500, 500], reuse=False):

        with tf.variable_scope(self.name + "model", reuse=reuse):
            output = x

            for i, layer in enumerate(layers):
                with tf.variable_scope('fc' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope('fc'):
                output = tf.layers.dense(output, self.y_size, activation=None)

            y = output
            return y

    def train(self):
        self.y_ = self.model(self.x)

        with tf.name_scope(self.name + "cost"):
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))
            self.loss = tf.reduce_sum(tf.square(self.y_ - self.y))
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr). \
                minimize(self.loss, global_step=self.global_step)

    def predict(self, session, x1, x2):  # model 전체 실행
        x = self.concat(x1, x2)

        act = session.run(self.y_, feed_dict={self.x: x})
        return act

    def concat(self, x1, x2):
        try:
            x = np.concatenate([x1, x2])
            x = np.reshape(x, [1, self.x_size])
        except:
            x1 = self.reshape(x1)
            x2 = self.reshape(x2)

            x = np.concatenate([x1, x2], axis=1)  # 이 경우, axis 0은 배치이므로, axis 1을 합쳐야함
            x = np.reshape(x, [len(x1), self.x_size])

        return x

    def reshape(self, x):
        try:
            x = np.reshape(x, [1, len(x)])
        except:
            pass

        return x

    def update(self, session, x1, x2, y, lr):  # model 전체 실행 후, opt까지 작동
        x = self.concat(x1, x2)
        y = self.reshape(y)

        # ir = session.run([self.irs], feed_dict={self.env_now: state, self.ir: action})
        loss, opt = session.run([self.loss, self.opt], feed_dict={self.x: x, self.y: y, self.lr: lr})
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

class Buffer():
    def __init__(self, state_size, target_size, act_size):
        self.state = np.empty(0).reshape(0, state_size)
        self.target = np.empty(0).reshape(0, target_size)
        self.act = np.empty(0).reshape(0, act_size)

    def append(self, now, new, act):
        # buffer에서 동일한 state-act 쌍이 이미 존재하는지 확인
        for i in range(len(self.state)):
            if np.array_equal(self.state[i], now) and np.array_equal(self.act[i], act):
                return

        self.state = np.vstack([self.state, now])
        self.target = np.vstack([self.target, new])
        self.act = np.vstack([self.act, act])

    #성공 여부에 따라 act 값 수정
    def act_update(self, success):
        step_all = len(self.act)

        if success:
            value = 1
        else:
            value = 0

        for step, act in enumerate(self.act):
            state = self.act[step][np.argmax(act)]
            self.act[step][np.argmax(act)] += (value - state) * step + 1 / step_all

    def clear(self):
        self.state = np.empty(0).reshape(0, self.state.shape[-1])
        self.target = np.empty(0).reshape(0, self.target.shape[-1])
        self.act = np.empty(0).reshape(0, self.act.shape[-1])


def buffer_append(buffer, now, new, act):
    # buffer에서 동일한 state-act 쌍이 이미 존재하는지 확인
    for i in range(len(buffer.state)):
        if np.array_equal(buffer.state[i], now) and np.array_equal(buffer.act[i], act):
            return

    # 없으면 추가
    buffer.state = np.vstack([buffer.state, now])
    buffer.target = np.vstack([buffer.target, new])
    buffer.act = np.vstack([buffer.act, act])

    return buffer

def stack(sess, batch, obj, anet, success, size=16):
    now_stack = np.empty(0).reshape(0, size)
    obj_stack = np.empty(0).reshape(0, size)
    act_stack = np.empty(0).reshape(0, anet.y_size)

    step = 0
    total_step = len(batch)
    #동일한 상태-액션 쌍이 여러번 들어갈 경우, 업데이트 후 발산하는 경향이 발생한다.
    # print(batch.shape)
    act_b = np.zeros(anet.y_size)
    state_b = np.zeros(size)
    for now, new, act in batch:
        # print(act)
        step += 1
        best = np.argmax(act)
        if success:
            act[best] = 1.

        else:
            act[best] = 0.

        act_p = np.zeros_like(act)
        act_p[np.argmax(act)] = 1

        now_stack = np.vstack([now_stack, now])
        obj_stack = np.vstack([obj_stack, obj])
        act_stack = np.vstack([act_stack, act])

    # print(now_stack.shape)
    loss = anet.update(sess, now_stack, obj_stack, act_stack, 0.1)

    return loss

if __name__ == '__main__':
    num_epi = 2000
    state_size = env.observation_space.n
    env_size = state_size  # state + old + dead

    act_size = env.action_space.n
    anet = Network("act", env_size * 2, act_size)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        batch_size = 50
        rList = []

        obj = np.zeros(env_size)
        obj[env_size - 1] = 1
        buffer = Buffer(state_size, state_size, act_size)

        for episode in range(num_epi):
            now = env.reset()
            now = onehot(now, env_size)
            rAll = 0

            done = False
            step_count = 0

            #버퍼 관련
            buffer.clear()
            while not done:
                # 행동 및 결과 관찰
                # random.shuffle(init_act)
                # act_onehot = init_act
                act_onehot = anet.predict(sess, now, obj)[0]

                # 행동 후처리
                act = np.argmax(act_onehot)

                # 환경에 적용
                new, reward, done, _ = env.step(act)

                # 환경 후처리
                new = onehot(new, env_size)

                # # 행동 후처리
                # act = np.zeros(act_size)
                # act[y] = 1

                # hole에 빠질 경우, 움직일 수 없는 상태 데이터 생성
                if done == True and reward == 0:
                    buffer.append(now, new, act_onehot)
                    now = new
                    random.shuffle(act_onehot)


                # replay buffer 추가 관련
                buffer.append(now, new, act_onehot)


                # 다음 루프 관련
                now = new
                rAll += reward
                step_count += 1

                # if episode % 100 == 0:
                #     env.render()

            #결과에 따른 act 수정, batch 데이터를 vstack으로 변형하여 학습
            loss = 0
            if rAll >= 1:
                buffer.act_update(True)
            else:
                buffer.act_update(False)

            anet.update(sess, buffer.state, buffer.target, buffer.act, 0.1)

            # 100회마다 결과 확인
            if episode % 100 == 0:
                print("Ep : ", episode, "steps : ", step_count)
                show(sess, anet)

            rList.append(rAll)
        show(sess, anet)
        print("Success rate: " + str(sum(rList) / num_epi))
        plt.bar(range(len(rList)), rList, color="blue")
        plt.show()
