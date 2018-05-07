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

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

def join(a, b):
    c = np.concatenate((a, b))
    return c

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

    def model(self, x, layers=[20], reuse=False):
        with tf.variable_scope(self.name + "model", reuse=reuse):
            output = x

            for i, layer in enumerate(layers):
                with tf.variable_scope('dense' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('output'):
                output = tf.layers.dense(output, self.y_size, activation=None)

            y = output
            return y

    def train(self):
        self.y_ = self.model(self.x)

        with tf.name_scope(self.name + "cost"):
            # self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
            self.loss = tf.reduce_sum(tf.square(self.y_ - self.y))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.loss, global_step=self.global_step)

    def predict(self, session, x):      #model 전체 실행
        act = session.run(self.y_, feed_dict={self.x: x})
        return act

    def update(self, session, x, y):   #model 전체 실행 후, opt까지 작동
        loss, opt = session.run([self.loss, self.opt], feed_dict={self.x: x, self.y: y})
        return loss

def replay(sess, batch, obj, anet, onet):
    #stack 생성
    now_stack = np.empty(0).reshape(0, len(obj))
    new_stack = np.empty(0).reshape(0, len(obj))
    obj_stack = np.empty(0).reshape(0, len(obj))
    act_p_stack = np.empty(0).reshape(0, anet.y_size)
    act_stack = np.empty(0).reshape(0, anet.y_size)

    for now, new, act, done in batch:
        #최적 행동 예측
        global temp
        temp = 0
        y = predict(sess, now, obj, anet, onet)
        act_p = onehot(y, anet.y_size)

        #최적 행동 후처리
        # act_p = np.zeros(anet.y_size)
        # act_p[np.argmax(y)] = 1.
        #경험한 것 기억


        #데이터 배치용 스택 쌓기
        now_stack = np.vstack([now_stack, now])
        new_stack = np.vstack([new_stack, new])
        act_stack = np.vstack([act_stack, act])

        obj_stack = np.vstack([obj_stack, obj])
        act_p_stack = np.vstack([act_p_stack, act_p])

    #업데이트
    loss1 = anet.update(sess, now_stack, obj_stack, act_p_stack)
    loss2 = onet.update(sess, now_stack, act_stack, new_stack)

    return loss1, loss2

def predict(sess, now, obj, anet, onet, count = 0):
    global temp
    cost_best = float('inf')
    act_best = -1
    old_best = 2
    i = 0
    acts = anet.predict(sess, now, obj)[0]

    if count == 0:
        length = len(acts)
    else:
        length = 1


    while i < length:
        i += 1
    # for i in range(length):  #range는 줄일 필요가 있다.(가장 확률이 높은 act만 해본다.)
        #가장 높은 확률대로 act 생성

        index_act = acts.rargmax()
        act = onehot(index_act, anet.y_size)
        # act = np.zeros_like(acts)  # acts와 같은 크기이며, 0으로 초기화되는 act를 만들고
        # act[index_act] = 1  # acts에서 가장 높은 act를 표시하기 위해 1로 만든다.

        #다음 act 생성을 위한 후처리
        acts[index_act] = 0  #다음을 위해 사용한 act는 acts에서 날림

        #new state 생성
        new = onet.predict(sess, now, act)[0]

        old_new = new[old_size - 1]

        #dead, old 보존
        dead = new[dead_size - 1]
        old = new[old_size - 1]

        #new state 후처리
        # index_new = new[0:-1].argmax()
        # new = onehot(index_new, len(new))
        # new[index_new] = 1

        #dead, old 후처리
        new[dead_size - 1] = dead
        new[old_size - 1] = old

        # if new[8] == 1:
        #     print(act

        if count < 1:
            cost_next = predict(sess, new, obj, anet, onet, count + 1) #살면 더 먼 미래로
        else:
            cost_next = np.sum(np.square(new - obj))   #말단 함수이면 판단


        if cost_next < cost_best:
            cost_best = cost_next
            old_best = old_new
            act_best = np.rargmax(act)

        # 목표 지향 확인 불가 시 작동, 탐험 선택
        elif cost_next == cost_best:
            old_best, act_best = select_less([old_best, act_best], [old_new, rargmax(act)])

            #선택지를 늘린다.
            length += 1
            length = np.minimum(length, len(acts))



    if count == 0:
        return act_best #count가 0인 처음 함수에선 cost 비교부분이 반드시 1번 이상 작동하므로 문제가 없다.
    else:
        return cost_best



        # if np.round(dead[0][0]) != 1 and reward < 2:                #안 죽으면 재귀적 호출
        #     act_reward[i] = predict_re(next, anet, onet, dnet, reward + 1) #이번 for문 act의 최종 예측결과 평가가 저장됨.
        # else:
        #     act_reward[i] = reward

        #count가 0인 것은 제일 처음 함수이다 => 재귀가 종료되었다. 예측을 바탕으로 가장 좋다고 생각하는 행동을 반환한다.t

# def rec(sess, anet, onet, state, obj, act):
#     # 알고리즘
#     # while(true)
#     # anet : state, obj -> act
#     # onet : state, act -> target
#     # if target == obj
#     #     true -> break
#     #     false -> state = target
#     # state = target
#
#     while(count != 100):
#         x = join(state, obj)
#         act = anet.predict(x)
#
#         x = join(state, act)
#         target = onet.predict(x)
#
#         process(target)
#
#         if same(target, obj) == True:
#             return
#         else:
#             state = target
#
#     return random

class Predictor():
    def __init__(self, act_size):
        self.act_size = act_size
        self.acts = np.empty(0).reshape(0, act_size)

    def process(self, target):


    def predict(self, sess, onet, state, obj):
        #알고리즘
        # for(count++)
        # onet : state, acts -> target

        # process(target)

        # if target == obj
        # true : return act
        # false : predict(state = targets)

        for act in range(self.act_size):
            act = onehot(act, self.act_size)
            x = join(state, act)
            target = onet.predict(x)

            target = process(target)

            if target == obj:
                return np.vstack((self.acts, act))
            else:
                self.predict(sess, onet, target, obj)


#전체 알고리즘
# 1. 무작위로 움직여서 예측망(state, act -> target)을 학습함.
# 2. 예측망으로 목표로 가는 act를 찾아냄. (이 시점에서 충분히 사용가능하게 만들어야 한다.)
#
# 3. 예측망으로 만든 데이터로 행동망을 학습함. (덤?)
# 연산이 많이 필요하다. => 이래서 안쓰기로 했구만


def select_less(a, b):
    if a[0] < b[0]:
        return a
    elif a[0] == b[0]:
        return random.choice([a, b])
    else:
        return b


def onehot(x, size):
    onehot = np.zeros(size)
    onehot[x] = 1
    return onehot

if __name__ == '__main__':
    num_epi = 2000
    step_limit = 1500
    batch = 50
    state_size = env.observation_space.n
    old_size = state_size + 1
    dead_size = old_size + 1
    env_size = dead_size  #state + old + dead
    # env_size = 1
    act_size = env.action_space.n
    anet = Network("act", env_size*2, act_size, 0.1)
    onet = Network("obj", env_size+act_size, env_size)
    # mnet = Network("mem", env_size, 1)
    # dnet = Network("dead", env_size+act_size, 1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        replay_buffer = deque()
        batch_size = 50
        rList = []

        obj = np.zeros(env_size)
        obj[dead_size - 1] = 0
        obj[old_size - 1] = 1

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
                y = np.rargmax(y)

                #환경에 적용
                new, reward, done, _ = env.step(y)

                #환경 후처리
                new = onehot(new, env_size)

                if done == True and reward == 0:
                    done = 1
                else:
                    done = 0
                new[dead_size - 1] = done

                # new[old_size - 1] = np.minimum(1, now[old_size - 1] + 0.1)


                #행동 후처리
                act = np.zeros(act_size)
                act[y] = 1


                # temp buffer 로 바꾸고, 성공 시 기억 x, 실패 시 기억하게 만들자.
                # replay buffer 추가 관련
                replay_buffer.append((now, new, act, done))
                if len(replay_buffer) > anet.replay_buffer:
                    replay_buffer.popleft() #이 경우, 잘되는 상황이 반복되면 기억하는 데이터가 비슷비슷해져 성능이 떨어지게 된다. 수정 필요 or early stopping을 적용
                    #Asynchronous method
                # 시각화 및 확인
                if episode % 50 == 0:
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
            if step_count > step_limit:
                pass
            rList.append(rAll)

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, step_count)
                    loss1, loss2 = replay(sess, minibatch, obj, anet, onet)
                print("Loss : ", loss1, loss2)
        plt.bar(range(len(rList)), rList, color="blue")
        plt.show()
