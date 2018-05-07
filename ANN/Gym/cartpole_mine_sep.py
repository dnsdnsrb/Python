import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
from gym.envs.registration import register


register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps':10000},
    reward_threshold=10000.0,
)

env = gym.make('CartPole-v2')
init_state = env.reset()
class Network():
    def __init__(self, image=[128, 128, 3], actions=20, ir=100):
        self.learning_rate = 0.1
        self.global_step = tf.Variable(0, trainable=False)

        #input
        self.env_size = env.observation_space.shape[0]

        #output(inner representation)
        self.ir_size = ir            #inner representation, 내부 표현

        self.act_num = env.action_space.n

        self.env_now = tf.placeholder(tf.float32, [None, self.env_size])   #환경
        self.env_new = tf.placeholder(tf.float32)  # 행동 후 환경
        # self.env_obj = tf.placeholder(tf.float32)   #목표 환경
        # self.done = tf.placeholder(tf.float32)

        self.train()

    def model_ir(self, env, layers=[200, 150, 100], reuse=False):

        with tf.variable_scope('ir', reuse=reuse):
            output = env

            for i, layer in enumerate(layers):
                with tf.variable_scope('fc_en' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_en'):
                output = tf.layers.dense(output, self.act_num, activation=None)

            ir = output  #actions

            for i, layer in enumerate(reversed(layers)):
                with tf.variable_scope('fc_de' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_de'):
                output = tf.layers.dense(output, self.env_size, activation=None)  #예측 상황 1개만 낸다.(완전한 AE는 아닌셈)

            env_ = output  # ir

            return env_, ir

    def model_pred(self, irs, layers=[200, 150, 100], reuse=False):

        with tf.variable_scope('pred', reuse=reuse):
            output = irs

            for i, layer in enumerate(layers):
                with tf.variable_scope('fc_en' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_en'):
                output = tf.layers.dense(output, self.act_num, activation=None)

            act = output  #actions
            output = tf.concat([act, self.env_now], -1)  #act + now => predict env

            for i, layer in enumerate(reversed(layers)):
                with tf.variable_scope('fc_de' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_de'):
                output = tf.layers.dense(output, self.env_size, activation=None)  #예측 상황 1개만 낸다.(완전한 AE는 아닌셈)

            pred = output  # ir

            return pred, act

    def model_act(self, irs, layers=[200, 150, 100], reuse=False):

        with tf.variable_scope('act', reuse=reuse):
            output = irs

            for i, layer in enumerate(layers):
                with tf.variable_scope('fc_en' + str(i)):
                    output = tf.layers.dense(output, layer, activation=tf.nn.relu)

            with tf.variable_scope('fc_en'):
                output = tf.layers.dense(output, self.act_num, activation=None)

            act = output  # actions
            return  act


    def train(self):

        self.pred, self.act = self.model_ir(self.env_now)

        with tf.name_scope("cost"):
            self.loss = tf.reduce_mean(tf.square(self.pred - self.env_new) + tf.square(self.pred - init_state))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.loss, global_step=self.global_step)

    def predict(self, session, state):      #model 전체 실행
        state = np.reshape(state, [1, self.env_size])
        act = session.run(self.act, feed_dict={self.env_now: state, self.env_new: None})
        return act

    def update(self, session, state, done):   #model 전체 실행 후, opt까지 작동
        # ir = session.run([self.irs], feed_dict={self.env_now: state, self.ir: action})
        pred, loss, opt = session.run([self.pred, self.loss, self.opt], feed_dict={self.env_now: state, self.env_new: done})
        return loss, opt

    def batch_train(self, session, net, batch):
        env_now_stack = np.empty(0).reshape(0, self.env_size)
        env_new_stack = np.empty(0).reshape(0, self.env_size)
        # irs_new_stack = np.empty(0).reshape(0, self.ir_size)

        for state, action, reward, state_new, done in batch:    #batch를 만들어냄

            env_now_stack = np.vstack([env_now_stack, state])  # state를 쌓음
            env_new_stack = np.vstack([env_new_stack, state_new])

        distance_stack = np.empty(0).reshape(0, 1)
        for start in range(len(env_new_stack)):
            distance = 0
            for stack in env_new_stack[start:]:
                distance += sum(np.square(init_state - stack))
            # print(distance)
            distance_stack = np.vstack([distance_stack, distance])

        return net.update(session, env_now_stack, env_new_stack)  # 쌓은 걸로 학습시킴.

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

if __name__ == '__main__':
    num_epi = 100000
    step_limit = 1500
    batch = 50
    net = Network()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        buffer = deque()

        for epi in range(num_epi):
            state = env.reset()
            done = False
            step_count = 0
            buffer.clear()  #buffer는 도전할 때마다 비우고 다시 함.
            while not done:
                act = net.predict(sess, state)
                action = np.argmax(act)
                # action = np.argmax(act + np.random.randn(1, env.action_space.n) / (epi + 1))
                # if epi % batch:
                #     print("act :", act)

                state_new, reward, done, _ = env.step(action)

                buffer.append((state, action, reward, state_new, done))  # exp_replay를 위해 buffer 넣어둠
                # if len(buffer) > batch:
                #     buffer.popleft()

                state = state_new
                step_count += 1  # 게임에서 버틴 시간을 의미
                if step_count > step_limit:
                    break

            print("Ep : ", epi, "steps : ", step_count)
            if step_count > step_limit:
                pass

            if epi % batch == 1:
                minibatch = buffer
                loss, _ = net.batch_train(sess, net, minibatch)
                print("Loss : ", loss)
                # sess.run(weights)
        # wrapper는 win10에서는 안 되는 듯
        net.play(sess, env)