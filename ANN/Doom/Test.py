from vizdoom import *
import random
import time
import numpy as np
import tensorflow as tf

a = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.2, 0.1, 0.4, 0.3]])
ind = tf.range(3) * tf.shape(a)[1] + [1, 2, 3]
act = np.array([1, 2, 3])

for i in range(a):
    print(i)

sess = tf.Session()

print(sess.run(a[:, act]))
# print(sess.run(ind))
# print(sess.run(tf.shape(a)[0]))
# print(sess.run(tf.shape(a)[1]))
# a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print(a.shape)
# print(a[:1].shape)
# print(a[0].shape)
# print(a[:2])
# print(a)

# game = DoomGame()
# game.load_config("../../vizdoom/scenarios/basic.cfg")
# game.init()
#
# shoot = [0, 0, 1]
# left = [1, 0, 0]
# right = [0, 1, 0]
# actions = [shoot, left, right]
#
# episodes = 10
# for i in range(episodes):
#     game.new_episode()
#     while not game.is_episode_finished():
#         state = game.get_state()
#         img = state.screen_buffer
#         misc = state.game_variables
#         reward = game.make_action(random.choice(actions))
#         print("\treward:", reward)
#         time.sleep(0.02)
#     print("Result:", game.get_total_reward())
#     time.sleep(2)