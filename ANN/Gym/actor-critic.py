import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)

# Superparameters
OUTPUT_GRAPH = True
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000             # maximum time step in one episode
RENDER = False                  # rendering
GAMMA = 0.9                     # reward discount in TD(Temporal difference) error
LR_A = 0.001                    # learning rate for actor
LR_C = 0.01                     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")    #critic에서 나온다.

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(self.s, 20, tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name='l1')

            self.acts_prob = tf.layers.dense(l1, n_actions, tf.nn.softmax,
                                             kernel_initializer=tf.random_normal_initializer(0., .1),
                                             bias_initializer=tf.constant_initializer(0.1),
                                             name='acts_prob')

            with tf.variable_scope('exp_v'):    #critic에서 나온 loss를 log(지수) 확률? 형태로 변경하는 듯
                log_prob = tf.log(self.acts_prob[0, self.a])    #acts_prob [[]] 형태, 여기서 act 1개만 빼내려고 [0, a]를 한거임.
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)   # advantage (TD_error) guided loss?

            with tf.variable_scope('train'):
                # minimize(-self.exp_v) = maximize(self.exp_v)
                self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]    #[[]]형태로 만들려고 그러는 듯(배치를 쓰지 않지만, tf가  배치형태만 받아서 그런듯?)
        feed_dict = {self.s: s, self.a: a, self.td_error: td}   # 여긴 feed_dict를 따로 만들어서 넣었다. 색다르군
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

    def choose_action(self, s):
        # 행동 확률에 따라 확률적으로 선택하도록 만드는 것으로 보임.
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   #return a int

class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(self.s, 20, tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name='l1')

            self.v = tf.layers.dense(l1, 1, None,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(0.1),
                                     name='V')

        with tf.variable_scope('squared_TD_error'): #Q learning인가 아닌가, 확인하려면 on off policy를 판단하면 될듯?
            self.td_error = self.r + GAMMA * self.v_ - self.v   #loss
            self.loss = tf.square(self.td_error)    #TD_error = (r+gamma*V_next) - V_eval

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})

        return td_error

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C) # teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    # 알고리즘
    # actor에서 action이 나옴
    # 그 action에 따라 움직임(env에 적용)
    # 그에 따른 reward를 받고 저장함
    # critic을 학습시킴
    # actor를 학습시킴
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER:
            env.render()

        a = actor.choose_action(s)
        s_, r, done, info = env.step(a)

        if done:
            r = -20

        track_r.append(r)

        #on policy 학습인 듯
        td_error = critic.learn(s, r, s_)   # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)         # true_gradient = grad[logPi(s, a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True
            else:
                RENDER = False
            print("epiode:", i_episode, " reward:", int(running_reward))
            break

