import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.signal
import scipy.misc
import os

from vizdoom import *
from time import sleep

#가중치 복사
def update_target_graph(source, target):
    weights = []
    sources = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=source)
    targets = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target)

    for source, target in zip(sources, targets):
        weights.append(tf.assign(target, source.value()))

    return weights

#화면 관련 처리
def process_frame(frame):
    #crop
    screen = frame[10:-10, 30:-30]

    #resize
    screen = scipy.misc.imresize(screen, [84, 84])

    #값 범위 조절(0~1)
    screen = np.reshape(screen, [np.prod(screen.shape)]) / 255.0

    return screen

#discount
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#가중치 초기화 함수
def normalized_columns_initializer(std=1.0):
    def _initalizer(shape, dtype=None, partition_info=None):
        out = np.random.rand(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initalizer

class ACNetwork():
    def __init__(self, state_size, act_size, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            self.scope = scope
            self.trainer = trainer

            self.act_size = act_size

            self.train()

    def conv(self, x):
        output = x
        # Conv
        output = tf.layers.conv2d(output, 16, [8, 8], [4, 4], activation=tf.nn.elu)
        output = tf.layers.conv2d(output, 32, [4, 4], [2, 2], activation=tf.nn.elu)

        output = tf.layers.flatten(output)

        output = tf.layers.dense(output, 256, activation=tf.nn.elu)

        y = output
        return y

    def lstm(self, x):
        #배치 크기를 sequence length로 변환하여 lstm을 작동한다.

        output = x
        #LSTM
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256)

        #LSTM에 들어갈 변수 설정
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = [c_in, h_in]

        rnn_in = tf.expand_dims(output, [0])    # [batch_size, step_size, data]로 만든다.
        # 여기서 batch_size를 전부 step_size로 만들며, expand_dims로 생긴 batch_size는 그냥 1이 된다.)
        step_size = tf.shape(self.imageIn)[:1]  # batch_size만큼을 step_size로 만들기위해 batch_size를 뽑아냄.
        # [:1]는 [0:1]이며 [0]와 유사, [[]]인게 []가 되냐, 그대로냐의 차이가 있다.
        # 여기선 행렬형태를 유지하면서 뽑아내기 위해 [:1]를 사용
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)    #걍 초기값만드는 과정인가? 모르겠네
        #

        output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in,
                                               initial_state=state_in,
                                               sequence_length=step_size)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])

        output = tf.reshape(output, [-1, 256])

        y = output
        return y

    def model(self):
        output = self.imageIn

        output = self.conv(output)

        output = self.lstm(output)

        #정책, 가치 생성
        # act_distribution(possibility)가 나옴
        self.policy = tf.layers.dense(output, self.act_size, activation=tf.nn.softmax,
                                      kernel_initializer=normalized_columns_initializer(0.01),
                                      bias_initializer=None)
        # value가 나옴(advantages에 사용)
        self.value = tf.layers.dense(output, 1, activation=None,
                                     kernel_initializer=normalized_columns_initializer(1.0),
                                     bias_initializer=None)

    def train(self):

        self.model()

        if self.scope == 'global':
            return

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.act_size, dtype=tf.float32)

        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)\

        # 해당 상황에서 평균적으로 받을 점수가 baseline, 이것을 빼서 분산을 낮춘다. => advantages
        # ex> left 1점, right 2점 => left -0.5점, right 0.5점 (baseline=1.5)
        # 다만, 따로 baseline을 구하는 것은 비효율적 => 현재 상태의 점수를 adv로 사용
        # => r_t+1 + gamma * V(s_t+1) - V(S_t)
        # r_t+1 + gamma * V(s_t+1) = Q function, V(s_t) = baseline
        # V는 critic에서 나온다.
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)    # 분산을 낮추기 위한 변수?

        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

        # loss 계산
        # advantages가 critic에서 나온 거다. Q - V = td_error
        # value_loss는 critic을 업데이트하기 위함인 듯
        # policy_loss는 policy를 업데이트하기 위함인 듯
        # 왜 loss로 둘을 합쳐놨지? => actor critic을 한 클래스로 합쳐놔서 꼼수 부린건가?
        self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
        self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
        # (advantage = Q - baseline) => 분산이 낮아짐.

        self.loss = self.value_loss * 0.5 + self.policy_loss - self.entropy * 0.01
        # self.loss = self.policy_loss - self.entropy * 0.01

        #gradient 계산, 지역망에서 구한 값으로 전역망을 업데이트하기 때문에 기본 함수를 사용못함.?
        locals_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.gradients = tf.gradients(self.loss, locals_vars)

        #gradient 정규화?
        self.var_norms = tf.global_norm(locals_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

        #지역망에서 구한 값으로 전역망을 업데이트
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads = self.trainer.apply_gradients(zip(grads, global_vars))

class Worker():
    def __init__(self, game, name, state_size, act_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        self.local_AC = ACNetwork(state_size, act_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        game = self.game_setting(game)
        self.actions = self.actions = np.identity(act_size, dtype=bool).tolist()
        self.env = game


    def game_setting(self, game):
        game.set_doom_scenario_path("basic.wad")  #"../../vizdoom/scenarios/basic.wad"
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)    #화면 크기
        game.set_screen_format(ScreenFormat.GRAY8)                  #색깔
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.set_window_visible(True)
        game.init()

        return game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout) #각종 상태가 1개의 배열(rollout)로 되어있다.

        #rollout에서 정보 추출
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        self.reward_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.reward_plus, gamma)[: -1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[: -1]
        advantages = discount(advantages, gamma)

        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}

        value_loss, policy_loss, entropy, \
        grad_norms, var_norms, \
        self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                            self.local_AC.policy_loss,
                                            self.local_AC.entropy,
                                            self.local_AC.grad_norms,
                                            self.local_AC.var_norms,
                                            self.local_AC.state_out,
                                            self.local_AC.apply_grads],
                                           feed_dict=feed_dict)

        return value_loss / len(rollout), \
               policy_loss / len(rollout), \
               entropy / len(rollout), \
               grad_norms, var_norms

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker" + str(self.number))

        with sess.as_default(), sess.graph.as_default():

            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                done = False

                self.env.new_episode()
                state = self.env.get_state().screen_buffer

                episode_frames.append(state)
                state = process_frame(state)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                while self.env.is_episode_finished() == False:
                    # policy에 넣어서 policy loss 나오는 부분?
                    act_dist, value, rnn_state = sess.run([self.local_AC.policy,
                                                           self.local_AC.value,
                                                           self.local_AC.state_out],
                                                          feed_dict={self.local_AC.inputs:[state],
                                                                     self.local_AC.state_in[0]:rnn_state[0],
                                                                     self.local_AC.state_in[1]:rnn_state[1]})

                    act = np.random.choice(act_dist[0], p=act_dist[0])
                    act = np.argmax(act_dist == act)

                    reward = self.env.make_action(self.actions[act]) / 100.0
                    done = self.env.is_episode_finished()

                    if done == False:
                        state1 = self.env.get_state().screen_buffer
                        episode_frames.append(state1)
                        state1 = process_frame(state1)
                    else:
                        state1 = state

                    episode_buffer.append([state, act, reward, state1, done, value[0, 0]])
                    episode_values.append(value[0 ,0])

                    episode_reward += reward
                    state = state1
                    total_steps += 1
                    episode_step_count += 1

                    if len(episode_buffer) >= 30 and done == False and episode_step_count != max_episode_length - 1:
                        # Critic에 넣어서 V(가치)가 나온다.
                        value1 = sess.run(self.local_AC.value,
                                          feed_dict={self.local_AC.inputs:[state],
                                                     self.local_AC.state_in[0]:rnn_state[0],
                                                     self.local_AC.state_in[1]:rnn_state[1]})[0, 0]

                        value_loss, policy_loss, entropy, \
                        grad_norms, var_norms = self.train(episode_buffer, sess, gamma, value1) #value1 = bootstrap 값
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                    if done == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0:    # bootstrap 값이 0이다.
                    value_loss, policy_loss, entropy, grad_norms, var_norms = self.train(episode_buffer, sess, gamma, 0.0)

                if episode_count % 5 == 0 and episode_count != 0:
                    print(episode_count)
                    # if self.name == 'worker_0' and episode_count % 25 == 0:
                    #     time_per_step = 0.05
                    #     images = np.array(episode_frames)
                    #     make_gif(images, './frames/image' + str(episode_count) + '.gif')

                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print(episode_count)

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(value_loss))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(entropy))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(grad_norms))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

max_episode_length = 300
gamma = .99
state_size = 84 * 84 * 1
act_size = 3    #LEFT, RIGHT, FIRE
load_model = False
model_path = './doom'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = ACNetwork(state_size, act_size, 'global', None)
    num_workers = multiprocessing.cpu_count()
    workers = []

    for i in range(num_workers):
        workers.append(Worker(DoomGame(), i, state_size, act_size, trainer, model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    if load_model == True:
        print('Loading model')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    workers_threads = []

    for worker in workers:
        worker_work = lambda : worker.work(max_episode_length, gamma, sess, coord, saver)
        thread = threading.Thread(target=(worker_work))
        thread.start()
        sleep(0.5)
        workers_threads.append(thread)

    coord.join(workers_threads)