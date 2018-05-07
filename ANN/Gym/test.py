import pickle
import numpy as np
import NNutils


def normalize(value, min, max):
    value = (value - min) / abs(max - min)

    return value

def rewardCalculate(state, target, mask):
    #표준화

    state[0] = normalize(state[0], 0, 100)
    target[0] = normalize(target[0], 0, 100)
    state[1] = normalize(state[1], 0, 100)
    target[1] = normalize(target[1], 0, 100)
    #계산
    reward = 0
    maskNum = 0
    for i in range(len(target)):
        if mask[i] == True:
            maskNum += 1
            reward += (state[i] - target[i]) ** 2

    rewardAvg = reward / maskNum
    return rewardAvg

state = np.array([100., .0, 0, 0])
target = np.array([100., 50., -101010110, 2130213012])
mask = np.array([True, True, False, False])

print(state)
r = rewardCalculate(state.copy(), target.copy(), mask)
print(state)
print("r", r)

# class Data:
#     def __init__(self, a):
#         self.a = a
# list = []
# data = Data(1)
# list.append(data)
# data.a = 2
# list.append(data)
# print(list)
# #
# def join(a, b):
#     c = np.concatenate((a, b))
#     return c
#
# b = [4, 5, 6]
# c = [3, 5, 6]
# a = np.empty(0).reshape(0, len(b))
#
#
# np.append(a,b)
# print(a)
# print(join(a, b))
#
#
# a = 1
# b = [1, 2, 3]
# c = np.array([1, 2, 3])
# print(type(a))
# print(type(b))
# print(type(c))
# if type(a) == int:
#     print("e")
# # len(a)
# a = NNutils.onehot(a, 4, list=False)
# print(a)
# a = np.zeros((16))
# print(a.shape)
# a = np.vstack(a)
# a = np.transpose(a, 1)
# print(a.T)
# buffer = []
# buffer.append([1, 2, 3])
# buffer.append([4, 5, 6])
# buffer.append([7, 8, 9])
#
# for i, j, k in buffer:
#     print(i)
#
# for i in range(len(buffer)):
#     buffer[i][2] = 1
#
# print(buffer)

# data = {'a': [1,2], 'b': [2,3]}

# print(dict.get('a'))
# dict.update({'a':2})
# print(dict.get('a'))
# print(data.items())
#
# def save():
#     with open('test.pickle', 'wb') as file:
#         pickle.dump(dict, file, protocol=pickle.HIGHEST_PROTOCOL)
#
# def load():
#     with open('test.pickle', 'rb') as file:
#         data = pickle.load(file)
#
#         print(data.items())
#
# load()
# def write():
#     with open('test.dat', 'w') as file:
#         for key, items in dict.items():
#             file.write(str(key) + ':' + str(items))
#
# def read():
#     dict = {}
#
#     with open('test.dat', 'r') as file:
#         for item in file:
#             if ':' in item:
#                 key, value = item.split(':', 1)
#                 dict.update({key: value})
#             else:
#                 pass

#     print(dict.items())
# read()