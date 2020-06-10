import tensorflow as tf
# import gym
import numpy as np
import random
from collections import deque
# from gym.envs.registration import register

x = np.array([1,2, 3, 4])
x = x.view(2,)
print(x)

# class Buffer():
#     pass
#
# buffer = Buffer()
#
# buffer.state = np.empty(0).reshape(0, 2)
# buffer.target = np.empty(0).reshape(0, 2)
# buffer.act = np.empty(0).reshape(0, 2)
#
# state = np.array([1, 2])
# state = state.reshape(1, -1)
#
# target = np.array([1.1, 2.2])
# target = target.reshape(1, -1)
#
# act = np.array([3, 4])
# act2 = np.array([3.3, 4.4])
#
# print(np.argmax(act))
#
# length = state.shape[-1]
# print(length)
#
# buffer.state = np.vstack([buffer.state, state])
# buffer.target = np.vstack([buffer.target, target])
# buffer.act = np.vstack([buffer.act, act])
# buffer.act = np.vstack([buffer.act, act2])
# act = buffer.act[0]
# for i, act in enumerate(buffer.act):
#     buffer.act[i][np.argmax(act)] = 1
# print(buffer.act)


#
# # a = np.array([[]])
# buffer = np.array([[[]]])
# stack = np.empty(0).reshape(0, 1, 6)
# state = np.array([1.1, 2.1, 3.1, 2.2])
# act = np.array([3.3, 4.4])
# # d = np.array([1.1, 2.1])
#
# #state와 act 합치는 부분
# env = np.concatenate([state, act])
# env = env.reshape([1, 1, 6])
#
# #time step 쌓기
# try:
#     buffer = np.append(buffer, env, axis=1)
# except:
#     buffer = np.append(buffer, env, axis=2)
# print(buffer.shape, stack.shape)
# #batch 쌓기
# stack = np.vstack([stack, buffer])
# print(stack.shape)
# # a = np.append(a, d)


#
# def show():
#     table = np.identity(16)
#     table = np.insert(table, 16, 1, axis=1)
#     print(table)
# show()

# def re(a = 1, count = 0):
#     best = -100
#     for i in range(1, 3):
#         b = a
#         # print(i)
#         c = random.choice([1, 2])
#
#         if count < 9:
#             b = re(b + c, count + 1)
#
#         if best < b:
#             best = b
#     return best
# a = re(1)
# print(a)

# a = np.array([[1, 2, 3], [4, 5, 6]])
# a = np.vstack(a)
# b = np.array([[7, 8, 9], [10, 11, 12]])
# b = np.vstack(b)
# c = np.concatenate([a, b], 1)
# print(c)
# # b = 3
# c = [4, 5, 6, 7]
# d = [0.1]
# print(a + 1)
#
# map(1, buffer, buffer)
# print(buffer)
# a = np.array([1, 2, 3, 4])
#
# print(a[a[0:-1].argmax()])

# env = gym.make('FrozenLake-v0')
# init_state = env.reset()
# print(env.observation_space)
# a = float('inf')
# if a == float('inf'):
#     print(a)
#
# a= 1
# b = np.array([2, 3.3])
#
# c = round(b[1])
# # c = np.append(b, a)
# print(c)


# acts = np.array([[0, 1]])
# print(acts[0, 0])
# act = np.zeros_like(acts)
# act[random.choice(act)] = 1
# print(act)
# def onehot(x1):
#     onehot = np.zeros((16))
#     onehot[x1] = 1
#     return onehot
#
#
# env = gym.make('FrozenLake-v0')
# init_state = env.reset()
# print(env.action_space.n)
# print(init_state, onehot(init_state))

# a = [1, 2]
# b = [3, 4]
# c = np.concatenate([a, b])
# c = np.reshape(c, [1, 4])

# env = gym.make('CartPole-v0')
#
# b= 1
# c = 5
# if not (a > b and c > a):
#     print(b)

#
# def re(r = 0):
#     r1 = [0, 0]
#
#     for i in range(2):
#         if r < 5 and i != 1:
#             r1[i] = re(r + 1)
#         else:
#             r1[i] = r
#
#     print(r1[1])
#     return r1[0]
#
# print(re())
# if np.square(a[0][1] - a[0][2]) > np.square(a[0][1] - a[0][0]):
#     print(a)
# if np.mean(a) > np.mean(b):
#     print(a)

# acts = np.array([4, 3, 2, 1])
#
# while acts.max() != 0:
#     index_max = acts.argmax()
#
#     act = np.zeros_like(acts)
#     act[index_max] = 1
#
#     acts[acts.argmax()] = 0
#     print(acts)

# stack = np.empty(0).reshape(0, 1)
# for data, _ in a:  # batch를 만들어냄
#     stack = np.vstack([stack, data])  # state를 쌓음
# stack = np.vstack(a)
# print(stack)
# register(
#     id='CartPole-v2',
#     entry_point='gym.envs.classic_control:CartPoleEnv',
#     tags={'wrapper_config.TimeLimit.max_episode_steps':10000},
#     reward_threshold=10000.0,
# )
# env = gym.make('CartPole-v2')
#
# env.reset()
# state_new, reward, done, _ = env.step(0)
# print(np.concatenate([state_new, [0,0,0,0]]))

