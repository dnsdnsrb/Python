import tensorflow as tf
import gym
import numpy as np
import random
import NNutils
import pickle
import os
import matplotlib.pyplot as plt
from shutil import copyfile

class Memory:
    def __init__(self, state_num, act_num, data_dir='data.pickle'):
        self.buffer = {}
        self.data = data_dir
        self.score_num = 3
        self.state_num = state_num
        self.act_num = act_num


    def save(self):
        if os.path.exists(self.data):
            copyfile(self.data, self.data + '.bak')

        with open(self.data, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load(self):
        with open(self.data, 'rb') as file:
            self.buffer = pickle.load(file)

    def add(self, state, act, score):
        #data[state][action][score]
        #state = dynamic
        #action = LEFT DOWN RIGHT UP = 4
        #score = trys, total_score, avg_score = 3
        if state in self.buffer:
            value = self.buffer.get(state)
        else:
            value = np.ones((self.act_num, self.score_num))

        value[act][0] += 1
        if score > 0:
            value[act][1] += score
        value[act][2] = value[act][1] / value[act][0]

        # print(total)

        # #총 시도횟수 증가
        # total[0] += 1
        # #성공(1) 시 1 추가, 실패 시 냅둠
        # if value > 0:
        #     total[1] += value
        #
        # total[2] = total[1] / total[0]

        # print(key, total)
        self.buffer.update({state: value})

class Network:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.x = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)
        self.learning_rate = 0.1

        self.train()

    def model(self, x, layers=[50, 50]):
        output = x

        for i, layer in enumerate(layers):
            with tf.variable_scope('dense' + str(i)):
                output = tf.layers.dense(output, layer, activation=tf.nn.tanh)

        with tf.variable_scope('output'):
            output = tf.layers.dense(output, self.output_size)

        y = output

        return y

    def train(self):
        self.y_ = self.model(self.x)
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_))
        self.opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def update(self, session, x, y):
        return session.run([self.opt, self.loss], feed_dict={self.x: x, self.y: y})

    def predict(self, session, x):
        x = np.reshape(x, [1, self.input_size])
        return session.run([self.y_], feed_dict={self.x: x})


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    net = Network(env.observation_space.n, env.action_space.n)
    mem = Memory(env.observation_space.n, env.action_space.n)
    buffer = []
    graph = []
    num_episodes = 10000

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for episode in range(num_episodes):
            if episode % 100 == 0:
                print(episode)
            #초기화
            state = env.reset()

            # state = NNutils.onehot(state, env.observation_space.n)
            # print(state)
            # print(state_init.shape)
            buffer = []
            done = False
            step_count = 0
            e = 1. / ((episode / 10) + 1)  # e-greedy
            e = 0.2
            reward_all = 0

            #def run()
            while not done:

                # print("state :", state)
                # action = net.predict(sess, state)
                # action_rand = np.random.randn(1, env.action_space.n) / (episode + 1)
                # action = np.argmax(action + action_rand)

                # if episode < num_episodes * 0.75:  # explore & exploit
                #     action = env.action_space.sample()
                # else:
                try:
                    y = np.array(list(mem.buffer.get(state)))
                    # print("y", y[:, 2])
                    action = np.argmax(y[:, 2])
                    print("did", action)
                except:
                    action = env.action_space.sample()
                    print("nop", state)
                # action = net.predict(sess, state)
                # action = np.argmax(action)

                # print(action)


                # print(action)
                state, reward, done, _ = env.step(action)
                buffer.append([state, action, reward])
                # print("state before :", state)
                # state = NNutils.onehot(state, env.observation_space.n)
                # print("state after :", state)
                if done and reward > 0:
                    print("success", buffer)
                    for i in range(len(buffer)):
                        buffer[i][2] = 1    #점수 줌

                if episode % 100 == 0:
                    env.render()

                reward_all += reward
            #

            for key, action, value in buffer:
                mem.add(key, action, value)

            # print(mem.buffer)
            x = np.array(list(mem.buffer.keys()))
            y = np.array(list(mem.buffer.values()))
            y = y[:, :, 2]
            x = NNutils.onehot(x, env.observation_space.n, list=True)

            # print("input & label",x.shape, y.shape)
            # print("x, y :", x, y)
            _, loss = net.update(sess, x, y)
            # print("loss", loss)

            graph.append(reward_all)

        x = np.array(list(mem.buffer.keys()))
        y = np.array(list(mem.buffer.values()))
        y = y[:, :, 2]
        print(x, y)
        print("Success rate: " + str(sum(graph) / num_episodes))
        plt.bar(range(len(graph)), graph, color="blue")
        plt.show()
        #데이터 형태가 잘못됨.
        # data[][3][4]
        # {state:
        # [
        # [left trys, right trys, top trys, bottom trys]
        # [left totalscore, right totalscore, top totalscore, bottom totalscore]
        # [left avgscore, right avgscore, top avgscore, bottom avgscore]
        # ]} 형태 비슷하게 되야함.
        # 이 형태로 만들 때 문제는 state, act 페어가 안된다는 점이다.
        # act를 받아서 data의 act형태로 변환해야할 필요가 있다.
        # 여기서 act는 (LEFT DOWN RIGHT UP)형태.
        # 변경하려면 방향을 체크하고, mem data[i][]에서 i값을 조정하여 처리한다.
        # 따라서 mem add 시 들어가게되는 것은 onehot이 아니다.
        # mem add(key, action, value) action은 뭐라 칭할지 의문이긴하다.

        # y = np.array(y)
        # print(y)
        # # print(y[:, :, 2])
        # print(x)
        # print(y[0])
        # print(x.shape)
           #batch가 문제가 됨. 행을 열로 바꾸고 onehot으로 변경해야한다.
        #y에서 avg_score만 받아와야함.

            # keys = []
            # values = []
            # for key, value in mem.buffer.items():
            #     print("key :", key, "value :", value)
            #     keys.append(key)
            #
            #     net.update(sess, key, value)


        # print(mem.buffer)



            # reset parameters

