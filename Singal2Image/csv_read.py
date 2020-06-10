import tensorflow as tf
import numpy as np
import os
import glob
import pandas as pd
import csv
import scipy
from scipy.signal import stft
from matplotlib import pyplot as plt
class CSV:
    def __init__(self):
        pass

base_dir = os.path.dirname(os.path.realpath(__file__))
file_paths = glob.glob(os.path.join(base_dir, '1st_test', '*'))

list.sort(file_paths)
file_paths = file_paths
print(file_paths)
data_list = []

# for c, file_path in enumerate(file_paths):
#     data = pd.read_csv(file_path, delim_whitespace=True, header=None)
#     data_list.append(data)
#     if c % 10 == 0:
#         print(c)
# dataset = pd.concat(data_list, ignore_index=True)
# print(dataset[0])
# stft = tf.signal.stft(dataset[0], 100, 100)

# dataset = pd.read_csv(file_paths[0], delim_whitespace=True, header=None)
# print(np.shape(data))

fft_list = []
label_list = []
for c, file_path in enumerate(file_paths):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    fft = abs(np.fft.fft(data, ))
    fft_list.append(fft)
    # label_list.append()
    if c % 10 == 0:
        print(c)



# fft = np.fft.fft(dataset)
# dB = abs(fft)
print(np.shape(fft_list))
# with open('test.npy', 'wb') as f:
np.save(f, fft_list)

# with open('test.npy', 'rb') as f:
#     np.
# file_csv = open('test.csv', 'w', encoding='utf-8')
# csv_writer = csv.writer(file_csv)
# csv_writer.writerows(fft_list)
# print(dB[0])
# plt.imshow(fft_list[0], aspect='auto', cmap='gray', interpolation='none')
# plt.colorbar()
# plt.savefig('test.png')

# print(stft)
# f, t, Zxx = stft(dataset[0])
# plt.pcolormesh(t, f, np.abs(Zxx))
# plt.savefig('test.png')
# print(stft)

# pd.read_csv()
# print(len(test[0])/ 300)
# print(test[0].dtype)
# fft = scipy.fft(test)
# plt.imshow(fft)
# print(fft.dtype)


# tf.data.experimental.make_csv_dataset()