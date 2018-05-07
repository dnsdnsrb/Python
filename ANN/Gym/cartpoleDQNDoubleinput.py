import gym
import numpy as np
import random
import tensorflow as tf
from collections import deque

def copy_weights(source, target):
    weights = []
    sources = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=source)
    targets = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target)


    for source, target in zip(sources, targets):
        weights.append(tf.assign(target, source.value()))

    return weights

def exploreAndExploit(episode, net, sess, state):
    action = np.argmax(net.predict(sess, state.present, state.target) + \
                       np.random.randn(1, env.action_space.n) / (episode + 1))

    return action

def normalize(value, min, max):
    value = (value - min) / abs(max - min)

    return value

def rewardCalculate(state, target, mask):
    #표준화
    state[0] = normalize(state[0], -2.4, 2.4)
    target[0] = normalize(target[0], -2.4, 2.4)
    state[2] = normalize(state[2], -41.8, 41.8)
    target[2] = normalize(target[2], -41.8, 41.8)

    #계산
    reward = 0
    maskNum = 0

    for i in range(len(target)):
        if mask[i] == True:
            maskNum += 1
            reward += (state[i] - target[i]) ** 2
    rewardAvg = reward / maskNum

    if rewardAvg == 1:
        print("Avg가 1")
    return rewardAvg

class DQN:
    def __init__(self, inputSize, outputSize, name='net'):
        with tf.variable_scope(name):
            self.input_size = inputSize
            self.output_size = outputSize
            self.learning_rate = 0.01
            self.x = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
            self.y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            self.discount = 0.9
            self.replay_memory = 50000
            self.rewardList = []
            # self.name = name

            self.train()

    def model(self, x,layers=[100, 100, 100]):
        #해결 => init에 variable scope를 넣으니 됨. (A3C 예제 코드에서 보고 배움)
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
        feed_dict = {self.x: state, self.y: action}
        return session.run([self.loss, self.opt], feed_dict=feed_dict)

    def predict(self, session, state, target=None):
        if target is not None:
            state = np.concatenate((state, target))

        state = np.reshape(state, [1, self.input_size])
        return session.run(self.y_, feed_dict={self.x: state})


    def exp_replay(self, sess, targetDQN, batch):  #
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)

        for state in batch:    #batch를 만들어냄
            x = np.concatenate((state.present, state.target))
            q = self.predict(sess, x)

            if state.done:
                q[0, state.action] = state.reward
            else:
                q[0, state.action] = state.reward + self.discount *\
                                     np.max(targetDQN.predict(sess, state.next, state.target))
                # print("q", q[0, state.action])

            x_stack = np.vstack([x_stack, x])  # state를 쌓음
            y_stack = np.vstack([y_stack, q])      # q를 쌓음

        return self.update(sess, x_stack, y_stack)  # 쌓은 걸로 학습시킴.

class State:
    def __init__(self, present=[], done=False):
        self.present = present
        self.target = []
        self.next = []
        self.action = -1
        self.reward = 0
        self.done = done

STEP_MAX = 1000
EPISODE_MAX = 5000

if __name__ == '__main__':
    # 알고리즘
    # state에 따른 action을 생성
    # action을 실행해봄
    # 결과를 buffer에 저장
    # episode가 끝날 때까지, 위를 반복
    # episode가 끝난 후 학습
    # 위를 반복
    with tf.Session() as sess:


        env = gym.make('CartPole-v0')
        env = env.unwrapped

        inputSize = env.observation_space.shape[0] * 2
        outputSize = env.action_space.n


        mask = np.array([False, False, True, False])

        net = DQN(name='main', inputSize=inputSize, outputSize=outputSize)
        target = DQN(name='target', inputSize=inputSize, outputSize=outputSize)

        tf.global_variables_initializer().run()
        replay_buffer = deque()

        weights = copy_weights('main', 'target')
        sess.run(weights)

        for episode in range(EPISODE_MAX):
            #reset parameters
            state = State()

            state.present = env.reset()
            stateTarget = np.array([0., 0., 0., 0.])
            state.done = False

            step = 0
            while not state.done:
                step += 1

                state.target = stateTarget
                state.action = exploreAndExploit(episode, net, sess, state)
                state.next, state.reward, state.done, _ = env.step(state.action)

                # copy()를 사용하지 않으면 array 값이 바뀌어 버린다.
                state.reward = rewardCalculate(state.next.copy(), state.target.copy(), mask)

                if state.done:    #실패 시
                    state.reward = -1

                # buffer 관리
                replay_buffer.append(state)  #exp_replay를 위해 buffer 넣어둠
                if len(replay_buffer) > net.replay_memory:
                    replay_buffer.popleft()

                state = State(state.next, state.done)
                # state.present = state.next
                # print("here = ", state.present, state.next)

                # step_count += 1 #게임에서 버틴 시간을 의미
                if step > STEP_MAX:
                    break

                if episode % 100 == 0:
                    env.render()

            print("Ep : ", episode, "step : ", step)
            if step > STEP_MAX:
                pass

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = net.exp_replay(sess, target, minibatch)
                print("Loss : ", loss)
                sess.run(weights)
        # #wrapper는 win10에서는 안 되는 듯
        # net.play(sess, env)