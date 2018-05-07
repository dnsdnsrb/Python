import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

import random as pr

#Q learning = 어떤 상태(DistributedNet(Canceled))에서는 어떤 행동(a)가 제일 좋다.(r) 이 표를 만들 때 목표 부분부터 거꾸로 업데이트됨(재귀적)
#방법자체는 어떤 상태에서 어떤 행동을 한 후에 가능한 모든 경우의 수의 보상 합을 구하는 식(연산량이 많다) => monte carlo search 사용

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': True}
)
env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

discount = 0.99
num_episodes = 2000
lr = 0.85

rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    e = 1.0 / ((i / 100) + 1)

    while not done:
        #action을 선택
        action = np.argmax(Q[state, :] \
                           + np.random.randn(1, env.action_space.n) / (i + 1))

        #action을 실행
        new_state, reward, done, _ = env.step(action)

        if done and reward == 0:  # 실패 시
            reward = -0.1

        #받은 reward를 바탕으로 Q-table update
        Q[state, action] = (1 - lr) * Q[state, action] \
                           + lr * (reward + discount * np.max(Q[new_state, :]))

        if reward == -0.1:
            pass
        else:
            rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()