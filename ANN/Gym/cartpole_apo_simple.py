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
        self.learning_rate = 0.1
        self.global_step = tf.Variable(0, trainable=False)

        self.x_size = x_size
        self.y_size = y_size
        #
        self.x = tf.placeholder(tf.float32, [None, x_size])
        self.y = tf.placeholder(tf.float32, [None, y_size])
        # self.model = tf.placeholder(tf.int32)

        self.train()

    def model(self, x, layers=[50, 50], reuse=False):

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
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
            self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).\
                minimize(self.loss, global_step=self.global_step)

    def predict(self, session, x):      #model 전체 실행
        act = session.run(self.y_, feed_dict={self.x: x})
        return act

    def update(self, session, x, y):   #model 전체 실행 후, opt까지 작동

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

    x_stack_anet = np.concatenate([now_stack, obj_stack], 1)
    x_stack_onet = np.concatenate([now_stack, act_stack], 1)

    #업데이트
    loss1 = anet.update(sess, x_stack_anet, act_p_stack)
    loss2 = onet.update(sess, x_stack_onet, new_stack)

    return loss1, loss2

def predict(sess, now, obj, anet, onet, cost_all = np.array(0), count = 0):
    cost_best = float('inf')
    # 원래 dnet으로 죽음여부를 확인하려하였으나 0 or 1만 판단할 수 있어, 죽음에 가까운 상황을 인지하지 못했다.
    # act를 내서 가장 높은 것을 먼저 따라감
    #
    # act_best = -1
    x = getinput(now, obj)
    acts = anet.predict(sess, x)[0]
    # act_best = [0, 0]

    if count == 0:
        length = len(acts)
    else:
        length = len(acts)

    for i in range(length):  #range는 줄일 필요가 있다.(가장 확률이 높은 act만 해본다.)
        cost = cost_all

        index_max = acts.argmax()
        act = np.zeros_like(acts)  # acts와 같은 크기이며, 0으로 초기화되는 act를 만들고
        act[index_max] = 1  # acts에서 가장 높은 act를 표시하기 위해 1로 만든다.

        acts[index_max] = float('-inf')  #다음을 위해 사용한 act는 acts에서 날림

        #next 생성하여
        x = getinput(now, act)
        new = onet.predict(sess, x)[0]

        cost_next = np.sum(np.square(new - obj))

        if count < 3:
            cost = predict(sess, new, obj, anet, onet, cost + cost_next, count + 1) #살면 더 먼 미래로
        # else:
        #     cost_next = np.sum(np.square(new - obj))   #말단 함수이면 판단
        #

        # print(i, count, cost_best, cost_next, cost)

        if new[-1] > 10:
            print(count, i)

        if cost < cost_best:
            cost_best = cost
            act_best = np.argmax(act)
            # print(acts.argmax(), act_best)
    if count == 0:
        return act_best #count가 0인 처음 함수에선 cost 비교부분이 반드시 1번 이상 작동하므로 문제가 없다.
    else:
        return cost_best

def onehot(x, size):
    onehot = np.zeros(size)
    onehot[x] = 1
    return onehot

def getinput(x1, x2):
    x = np.concatenate([x1, x2])
    x = np.expand_dims(x, 0)

    return x

if __name__ == '__main__':
    num_epi = 2000
    step_limit = 1500
    batch = 50
    dead_size = 1
    state_size = env.observation_space.shape[0]
    env_size = state_size + dead_size

    act_size = env.action_space.n
    anet = Network("act", env_size*2, act_size)
    onet = Network("obj", env_size+act_size, env_size)

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

            x = getinput(now, obj)

            while not done:
                #행동 및 결과 관찰
                act_onehot = anet.predict(sess, x)

                #행동 후처리
                act = np.argmax(act_onehot)

                #환경에 적용
                new, reward, done, _ = env.step(act)

                #환경 후처리
                if done == True:
                    done = 100
                    # print(done)
                if done == False:
                    done = 0
                new = np.append(new, done)

                #행동 후처리
                act_processed = [0, 0]
                act_processed[act] = 1

                #temp buffer로 바꾸고, 성공 시 기억 x, 실패 시 기억하게 만들자.
                #replay buffer 추가 관련
                replay_buffer.append((now, new, act_processed))
                if len(replay_buffer) > anet.replay_buffer:
                    replay_buffer.popleft() #이 경우, 잘되는 상황이 반복되면 기억하는 데이터가 비슷비슷해져 성능이 떨어지게 된다. 수정 필요 or early stopping을 적용
                                            #Asynchronous method
                #시각화 및 확인
                if episode % 100 == 0:
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
        # net.play(sess, env)