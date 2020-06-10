import numpy as np
import scipy.integrate as integra
from matplotlib import pyplot as plt
# test = np.load('/home/futuremain-ywk/PycharmProjects/Python/Singal2Image/dataOutputs/4차 Correct/Misalignment01_Acc.npy')
# print(np.shape(test))
# print(test)
from nptdms import TdmsFile
from datetime import datetime
import datetime
import glob
import os
import scipy
import pandas
from math import sin, cos
a = [1, 2, 3, 4]

print(a[2:4], np.trapz(a[2:4]))

# x = np.arange(0, 100)
# y = np.array([sin(i) + 10 for i in x])
# print(y)
#
# plt.plot(x, y)
# plt.savefig('test.png')
# plt.close()
# fft = abs(np.fft.fft(y))
#
# plt.plot(x, fft)
# plt.savefig('test.png')

'''
base_dir = os.path.dirname(os.path.realpath(__file__))
file_paths = glob.glob(os.path.join(base_dir, 'RawDatas', '*.tdms'))
list.sort(file_paths)
print(file_paths)

time2 = None
time_diff = datetime.timedelta(0, 2400)
# vel_init = np.zeros(12)
sample_rate = 0
for file_path in file_paths:
    with TdmsFile.read(file_path) as tdms_file:
        df = tdms_file.as_dataframe(scaled_data=False)
        df = np.transpose(df.to_numpy())

        # 이름에서 시간 구하기
        time1 = file_path.rpartition('/')[-1]
        time1 = datetime.datetime.strptime(time1.rpartition('.')[0], '%Y%m%d_%H%M%S')
        if time2 != None: # 제일 첫번째는 없음 => time_diff 초기값 사용
            time_diff = time1 - time2

        # sample rate 구하기.
        rate = time_diff.seconds / np.shape(df)[1]

        if rate < 1: # 시간 간격이 매우 크다 = 꺼진걸로 판단
            sample_rate = rate
        # else: # 꺼졌으면 초기 속도를 없앤다.
            # vel_init = np.ones(12)

        vel_list = []
        # print(df[0][0:2])
        # for point_n in range(0, np.shape(df)[0]):
        #     test = scipy.integrate.simps(df[0][11:13]) * sample_rate
        # print(np.shape(test))
        # print(test)
        # for point_n in range(0, np.shape(df)[0]):
        #     scipy.integrate.simps(0, )
        # exit()


        for point_n in range(0, np.shape(df)[0]):
            vel = np.zeros(np.shape(df)[1]) # 장비는 고정되어있으므로, 속도는 쌓이지 않고 진동만 한다고 가정하여 0으로 둔다.
            # vel[-1] = vel_init[point_n] # for문에서 vel[-1]을 사용하므로 -1을 초기값으로 활용
            acc_prior = 0
            for data_n, acc in enumerate(df[point_n]):
                # if data_n >= 2:
                #     vel[data_n] = scipy.integrate.romb(df[point_n][data_n - 2:data_n]) * sample_rate  # 속도 계산. vel[-1]은 끝을 가리킨다 = 초기값
                # else:
                #     pass

                if data_n == 0:
                    vel[data_n] = 0
                elif acc - acc_prior > 0:
                    vel[data_n] = acc * sample_rate - (acc - acc_prior) * sample_rate / 2
                else:
                    vel[data_n] = acc * sample_rate + (acc - acc_prior) * sample_rate / 2
                acc_prior = acc

                # vel[data_n] = acc * sample_rate  # 속도 계산. vel[-1]은 끝을 가리킨다 = 초기값
                # vel[data_n] = vel[data_n - 1] + acc * sample_rate  # 속도 계산. vel[-1]은 끝을 가리킨다 = 초기값
                # vel = np.append(vel, vel[-1] + acc * sample_rate) # 속도 계산.
            vel_list.append(vel)
            # vel_init[point_n] = vel[-1]
        vel_list = np.stack(vel_list) # 포인트 별 속도

        # print(vel_init)
        print(sample_rate, np.shape(vel_list), np.max(vel_list), np.min(vel_list))
        # for i in range(0, np.shape(df)[0]):
        #     print(np.max(df[i]), np.max(vel_list[i]), np.min(vel_list[i]))
        # print(vel_list[1])

        time2 = time1

        # print(name)'''


# for file_path in file_paths:
#     with TdmsFile.read(file_path) as tdms_file:
#         df = tdms_file.as_dataframe(scaled_data=False)
#         # print(df)
#         df = np.transpose(df.to_numpy())
#         # print(np.shape(df))
#         # print(df[])
#
#         vel_list = []
#         for i in range(0, np.shape(df)[0]):
#             vel = np.array([0])
#             for acc in df[i]:
#                 vel = np.append(vel, vel[-1] + acc * 0.375) #  * 22.5
#             vel_list.append(vel[1:])
#         vel_list = np.stack(vel_list)
#         print(np.shape(vel_list))
#         for i in range(0, np.shape(df)[0]):
#             print(np.max(df[i]), np.max(vel_list[i]), np.min(vel_list[i]))
#     # print(vel_list[1])

print((40 * 60 / 6400))
print(22.5 * 6400)
print(85 * 60 / 25600)

