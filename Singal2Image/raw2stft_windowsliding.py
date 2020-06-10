import numpy as np
import os
from nptdms import TdmsFile
import glob
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
import sys




# window sliding이 들어간 시간순 stft
rawdatas = np.load('Rawdatas.npy', mmap_mode='r')
print(np.shape(rawdatas))
type_n = np.shape(rawdatas)[1]

print(30208000 / 2156)
print(64 * 64)
list = []
for c in range(0, type_n):
    print("count", c)
    datas = np.transpose(rawdatas)[0]

    _, _, Zxx = signal.stft(datas, nperseg=64*64, nfft=64*64)
    print(np.shape(Zxx))
    # print(np.shape(Zxx), Zxx[0])
    # Zxx = Zxx.reshape(8193, -1, 15)
    Zxx = np.transpose(Zxx)
    list.append(Zxx)
list = np.stack(list)
print(np.shape(list))

# 뭔가 잘못함
list = np.transpose(list, (1, 2, 0))
    # for i in range(6, 30):
    #     if np.shape(Zxx)[0] % i == 0:
    #         divisor = i
    #         print(i)
    #         break
    # print(np.shape(Zxx), Zxx[0])
    # Zxx = Zxx.reshape(-1, divisor, 8193)
    # list.append(Zxx)
    # try:
    #     Zxx = np.expand_dims(Zxx, axis=0)
    #     Zxx1 = np.vstack((Zxx1, Zxx))
    # except:
    #     Zxx1 = Zxx
    #     # print("error")
    # finally:
    #     print(np.shape(Zxx1))


    # list = np.vstack((list, Zxx))

    # print(sys.getsizeof(Zxx))



plt.imshow(np.abs(list[0]), aspect='auto', interpolation='none')
plt.colorbar()
plt.savefig('test.png')

#