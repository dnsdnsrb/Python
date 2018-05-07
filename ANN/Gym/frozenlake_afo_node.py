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

class Node:
    def __init__(self, name=None, link={}):
        self.name = name
        self.link = link

temp = 0    #테스트용 전역변수
class Network():    #env net이 될수도, obj net이 될수도, env net이 될수도 있다.
    replay_buffer = 50000
    def __init__(self, name, x_size, y_size, learning_rate=0.001):
        self.name = name
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False)

        self.x_size = x_size
        self.y_size = y_size
        #
        self.x = tf.placeholder(tf.float32, [None, x_size])
        self.y = tf.placeholder(tf.float32, [None, y_size])
        # self.model = tf.placeholder(tf.int32)

        self.train()

    def model(self, x, layers=[20, 20], reuse=False):

        with tf.variable_scope(self.name + "model", reuse=reuse):
            output = x

            for i, layer in enumerate(layers):
                with tf.variable_scope('fc_en' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_en'):
                output = tf.layers.dense(output, self.y_size, activation=None)

            y = output
            return y

    def train(self):
        self.y_ = self.model(self.x)

        with tf.name_scope(self.name + "cost"):
            # self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
            self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.loss, global_step=self.global_step)

    def predict(self, session, x1, x2):      #model 전체 실행
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

            x = np.concatenate([x1, x2], axis=1)    #이 경우, axis 0은 배치이므로, axis 1을 합쳐야함
            x = np.reshape(x, [len(x1), self.x_size])

        return x

    def reshape(self, x):
        try:
            x = np.reshape(x, [1, len(x)])
        except:
            pass

        return x

    def update(self, session, x1, x2, y):   #model 전체 실행 후, opt까지 작동
        x = self.concat(x1, x2)
        y = self.reshape(y)

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


def replay(sess, batch, obj, anet, onet, mnet):
    #stack 생성
    now_stack = np.empty(0).reshape(0, len(obj))
    new_stack = np.empty(0).reshape(0, len(obj))
    obj_stack = np.empty(0).reshape(0, len(obj))
    act_p_stack = np.empty(0).reshape(0, anet.y_size)
    act_stack = np.empty(0).reshape(0, anet.y_size)

    for now, new, act in batch:
        #최적 행동 예측
        global temp
        temp = 0

        nows = np.vstack([now, now, now, now])
        make_obj(sess, nows, obj, onet)

        y = predict(sess, now, obj, anet, onet)
        act_p = onehot(y, anet.y_size)

        #데이터 배치용 스택 쌓기
        now_stack = np.vstack([now_stack, now])
        new_stack = np.vstack([new_stack, new])
        act_stack = np.vstack([act_stack, act])

        obj_stack = np.vstack([obj_stack, obj])
        act_p_stack = np.vstack([act_p_stack, act_p])

    #업데이트
    loss1 = anet.update(sess, now_stack, new_stack, act_stack)
    loss2 = onet.update(sess, now_stack, act_stack, new_stack)
    loss3 = mnet.update(sess, new_stack, act_stack, now_stack)

    return loss1, loss2, loss3


#onet 동작
def make_obj(sess, now, obj, onet, act = np.identity(4), count = 0):

    now = onet.predict(sess, now, act)

    loss = float('inf')
    for i in range(len(now)):
        loss = min(np.sum(np.square(now[i] - obj)), loss)
        
    if loss > 1 and count < 4:
        act = np.vstack([act, act, act, act])
        now = np.vstack([now, now, now, now])
        now = make_obj(sess, now, obj, onet, act, count + 1)


    return now



def predict(sess, now, obj, anet, onet, count = 0):
    global temp
    cost_best = float('inf')
    act_best = -1
    i = 0
    act_init = anet.predict(sess, now, obj)[0]
    # act_init = np.array([1, 0.75, 0.5, 0.25])
    # np.random.shuffle(act_init)

    if count == 0:
        length = len(act_init)
    else:
        length = 1

    while i < length:
        i += 1
        index_act = act_init.argmax()
        act = onehot(index_act, anet.y_size)

        #다음 act 생성을 위한 후처리
        act_init[index_act] = 0  #다음을 위해 사용한 act는 acts에서 날림

        #new state 생성
        new = onet.predict(sess, now, act)[0]

        if count < 1:
            cost_next = predict(sess, new, obj, anet, onet, count + 1) #살면 더 먼 미래로
        else:
            cost_next = np.mean(np.square(new - obj))   #말단 함수이면 판단

        if cost_next < cost_best:
            cost_best = cost_next
            act_best = np.argmax(act)
        # elif cost_next == cost_best:
        #     np.random.shuffle(act)
        #     act_best = np.argmax(act)


    if count == 0:
        return act_best #count가 0인 처음 함수에선 cost 비교부분이 반드시 1번 이상 작동하므로 문제가 없다.
    else:
        return cost_best

# def explore(sess, now, obj, anet, onet, count = 0):
#     act_init = anet.predict(sess, now, obj)[0]

#mnet 동작
def revert(sess, new, obj, act, mnet):
    old = mnet.predict(sess, obj, act)


def select_less(a, b):
    if a[0] < b[0]:
        return a
    elif a[0] == b[0]:
        return random.choice([a, b])
    else:
        return b


def show(sess, anet, onet):
    table = np.identity(16)
    act = np.identity(4)

    obj = np.zeros_like(table)
    obj[:, 15] = 1
    # print(obj)
    print(anet.predict(sess, table, obj))
    print(onet.predict(sess, table[15], act[0]))


def onehot(x, size):
    onehot = np.zeros(size)
    onehot[x] = 1
    return onehot


if __name__ == '__main__':
    num_epi = 2000
    step_limit = 1500
    batch = 50
    state_size = env.observation_space.n
    env_size = state_size  #state + old + dead
    # env_size = 1
    act_size = env.action_space.n
    anet = Network("act", env_size*2, act_size, 0.1)
    onet = Network("obj", env_size+act_size, env_size)
    mnet = Network("mem", env_size+act_size, env_size)
    # dnet = Network("dead", env_size+act_size, 1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        replay_buffer = deque()
        batch_size = 50
        rList = []

        obj = np.zeros(env_size)
        obj[env_size - 1] = 1

        for episode in range(num_epi):
            now = env.reset()
            now = onehot(now, env_size)
            rAll = 0

            done = False
            step_count = 0

            while not done:
                #행동 및 결과 관찰
                y = anet.predict(sess, now, obj)

                #행동 후처리
                y = np.argmax(y)

                #환경에 적용
                new, reward, done, _ = env.step(y)

                #환경 후처리
                new = onehot(new, env_size)

                #행동 후처리
                act = np.zeros(act_size)
                act[y] = 1

                #hole에 빠질 경우, 움직일 수 없는 상태 데이터 생성
                if done == True and reward == 0:
                    replay_buffer.append((now, new, act))
                    now = new
                    random.shuffle(act)


                # temp buffer 로 바꾸고, 성공 시 기억 x, 실패 시 기억하게 만들자.
                # replay buffer 추가 관련
                replay_buffer.append((now, new, act))
                if len(replay_buffer) > anet.replay_buffer:
                    replay_buffer.popleft() #이 경우, 잘되는 상황이 반복되면 기억하는 데이터가 비슷비슷해져 성능이 떨어지게 된다. 수정 필요 or early stopping을 적용
                    #Asynchronous method
                # 시각화 및 확인
                if episode % 100 == 0:
                    env.render()

                #종료 조건 추가
                step_count += 1
                if step_count > step_limit:
                    print("over step")
                    break

                #다음 루프 관련
                now = new
                rAll += reward

            #100회마다 결과 확인
            if episode % 100 == 0:
                print("Ep : ", episode, "steps : ", step_count)
                show(sess, anet, onet)
            if step_count > step_limit:
                pass
            rList.append(rAll)

            if episode % 10 == 1:
                loss1, loss2, loss3 = 0, 0, 0
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, step_count)
                    loss1, loss2, loss3 = replay(sess, minibatch, obj, anet, onet, mnet)
                print("Loss : ", loss1, loss2, loss3)
        plt.bar(range(len(rList)), rList, color="blue")
        plt.show()
