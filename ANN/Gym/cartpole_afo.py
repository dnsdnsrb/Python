import tensorflow as tf
import gym
import numpy as np
import random
import time
from collections import deque
from gym.envs.registration import register

# register(
#     id='CartPole-v2',
#     entry_point='gym.envs.classic_control:CartPoleEnv',
#     tags={'wrapper_config.TimeLimit.max_episode_steps':10000},
#     reward_threshold=10000.0,
# )
env = gym.make('CartPole-v1')
init_state = env.reset()

temp = 0    #테스트용 전역변수
class Network():    #env net이 될수도, obj net이 될수도, env net이 될수도 있다.
    replay_buffer = 50000
    def __init__(self, name, x_size, y_size):
        self.name = name
        self.learning_rate = 0.0001
        self.global_step = tf.Variable(0, trainable=False)

        self.x_size = x_size
        self.y_size = y_size
        #
        self.x = tf.placeholder(tf.float32, [None, x_size])
        self.y = tf.placeholder(tf.float32, [None, y_size])
        # self.model = tf.placeholder(tf.int32)

        self.train()

    def model(self, x, layers=[50, 50, 50], reuse=False):

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
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
            self.loss = tf.reduce_sum(tf.square(self.y_ - self.y))
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).\
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
        # print(y.shape)

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

# def create_batch(batch, input_size):    #배치가 여러항목을 포함한 경우, 다음과 같이 항목을 뽑아냄 [[x, y] for x, y, _ in a]
#     # stack = np.empty(0).reshape(0, input_size)
#
#     stack = np.vstack(batch)
#
#     for data in batch:    #batch를 만들어냄
#         stack = np.vstack([stack, data])  # state를 쌓음
#
#     return stack

def replay(sess, batch, obj, anet, onet):
    #stack 생성
    now_stack = np.empty(0).reshape(0, len(obj))
    new_stack = np.empty(0).reshape(0, len(obj))
    obj_stack = np.empty(0).reshape(0, len(obj))
    act_p_stack = np.empty(0).reshape(0, anet.y_size)
    act_stack = np.empty(0).reshape(0, anet.y_size)

    for now, new, act in batch:
        #최적 행동 예측

        y = predict(sess, now, obj, anet, onet)
        #최적 행동 후처리
        act_p = [0., 0.]
        act_p[np.argmax(y)] = 1.

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

def predict(sess, now, obj, anet, onet, cost_best = float('inf'), count = 0):
    # 원래 dnet으로 죽음여부를 확인하려하였으나 0 or 1만 판단할 수 있어, 죽음에 가까운 상황을 인지하지 못했다.
    # act를 내서 가장 높은 것을 먼저 따라감
    #
    # act_best = -1
    acts = anet.predict(sess, now, obj)
    # act_best = [0, 0]

    if count == 0:
        length = len(acts[0])
    else:
        length = 1

    for i in range(length):  #range는 줄일 필요가 있다.(가장 확률이 높은 act만 해본다.)
        index_max = acts.argmax()
        act = np.zeros_like(acts)  # acts와 같은 크기이며, 0으로 초기화되는 act를 만들고
        act[0][index_max] = 1  # acts에서 가장 높은 act를 표시하기 위해 1로 만든다.
        acts[0][index_max] = 0  #다음을 위해 사용한 act는 acts에서 날림

        #next 생성하여
        new = onet.predict(sess, now, act)

        #예측의 오차를 계산
        cost_next = np.mean(np.square(new - obj))
        # print(new)
        # print(new[0][4])
        #
        if round(new[0][4]) == 1:
            return float('inf')

        #예측을 바탕으로 더 먼 미래를 예측
        #종료 조건, 5번 이상 내려갔거나, cost가 충분히 낮다면 종료. 아니면 count를 올리고 다음을 진행
        if not (count > 5):
            cost_next = predict(sess, new, obj, anet, onet, cost_next, count + 1) #재귀적으로 들어가 더 좋은게 있으면 그걸로 바뀜.

        # print(cost_next, cost_best)
        if cost_next <= cost_best:
            cost_best = cost_next
            act_best = act

    # print(count)
    if count == 0:
        return act_best #count가 0인 처음 함수에선 cost 비교부분이 반드시 1번 이상 작동하므로 문제가 없다.
    else:
        return cost_best



        # if np.round(dead[0][0]) != 1 and reward < 2:                #안 죽으면 재귀적 호출
        #     act_reward[i] = predict_re(next, anet, onet, dnet, reward + 1) #이번 for문 act의 최종 예측결과 평가가 저장됨.
        # else:
        #     act_reward[i] = reward

    #count가 0인 것은 제일 처음 함수이다 => 재귀가 종료되었다. 예측을 바탕으로 가장 좋다고 생각하는 행동을 반환한다.t

# def compare(a, b):
#     if a == None:
#         return b
#     elif b == None:
#         return a
#     elif a < b:
#         return b
#     elif a > b:
#         return a
#     elif a == b:
#         return a
#     else:
#         print("error")
#         return -1



if __name__ == '__main__':
    num_epi = 100000
    step_limit = 1500
    batch = 50
    env_size = env.observation_space.shape[0] + 1
    # env_size = 1
    act_size = env.action_space.n
    anet = Network("act", env_size*2, act_size)
    onet = Network("obj", env_size+act_size, env_size)
    nnet = Network("now", env_size+act_size, env_size)
    # dnet = Network("dead", env_size+act_size, 1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        replay_buffer = deque()
        batch_size = 50

        for episode in range(num_epi):
            now = env.reset()
            now = np.append(now, 0)
            obj = np.array([0, 0, 0, 0, 0])
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
                if done == True:
                    done = 1
                    # print(done)
                if done == False:
                    done = 0
                new = np.append(new, done)

                #행동 후처리
                act = [0, 0]
                act[y] = 1

                #temp buffer로 바꾸고, 성공 시 기억 x, 실패 시 기억하게 만들자.
                #replay buffer 추가 관련
                replay_buffer.append((now, new, act))
                if len(replay_buffer) > anet.replay_buffer:
                    replay_buffer.popleft() #이 경우, 잘되는 상황이 반복되면 기억하는 데이터가 비슷비슷해져 성능이 떨어지게 된다. 수정 필요 or early stopping을 적용
                                            #Asynchronous method
                #시각화 및 확인
                if episode % 50 == 0:
                    print("cost :", np.mean(np.square(new - obj)))
                    print("act :", act)

                    env.render()

                #종료 조건 추가
                step_count += 1
                if step_count > step_limit:
                    break

                #다음 루프 관련
                now = new

            #100회마다 결과 확인
            if episode % 100 == 0:
                print("Ep : ", episode, "steps : ", step_count)
            if step_count > step_limit:
                pass

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss1, loss2 = replay(sess, minibatch, obj, anet, onet)
                print("Loss : ", loss1, loss2)

        # wrapper는 win10에서는 안 되는 듯
        net.play(sess, env)