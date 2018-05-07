import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

import random as pr

#Q learning = 어떤 상태(DistributedNet(Canceled))에서는 어떤 행동(a)가 제일 좋다.(r) 이 표를 만들 때 목표 부분부터 거꾸로 업데이트됨(재귀적)
#방법자체는 어떤 상태에서 어떤 행동을 한 후에 가능한 모든 경우의 수의 보상 합을 구하는 식(연산량이 많다) => monte carlo search 사용
#이것은 dummy Q-learning으로 Expolit & Explore 가 없고, dsicounted future reward가 적용되어 있지 않다.

# def rargmax(vector):    #r = reward, 가장 높은 보상을 주는 행동을 찾는다. 동일한게 있으면 그 중 무작위로 선택
#     m = np.amax(vector)
#     indices = np.nonzero(vector == m)[0]
#
#     return pr.choice(indices)
#
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)
env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = 0.85
discount = 0.99
num_episodes = 3000

rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    if i % 100 == 0:
        print(i)

    e = 1.0 / ((i / 100) + 1)

    while not done:

        #action을 선택
        action = np.argmax(Q[state, :] + \
                           np.random.randn(1, env.action_space.n) / (i + 1))

        #action을 실행
        new_state, reward, done, _ = env.step(action)

        # if done and reward == 0:  # 실패 시
        #     reward = -1
            # print(reward)


        # Get negative reward every step
        # if reward == 0 :
        #     reward = -0.001

        #받은 reward를 바탕으로 Q-table update
        # Q[state, action] += learning_rate * (reward + discount * np.max(Q[new_state, :] - Q[state, action]))
        Q[state, action] = reward + discount * np.max(Q[new_state, :])

        # print(Q)
        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()