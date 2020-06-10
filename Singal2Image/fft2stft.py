from scipy import signal
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import figure
import numpy as np
import librosa.display
import pandas as pd
import glob
import os
from PIL import Image
import pathlib
base_dir = os.path.dirname(os.path.realpath(__file__))
file_paths = glob.glob(os.path.join(base_dir, 'Datasets', '*Acc.CSV'))
from mpl_toolkits.mplot3d import Axes3D
print(base_dir)

MAX_CH = 9
list.sort(file_paths)

file_paths = np.array(file_paths).reshape(-1, MAX_CH)

# Time
# file_paths = np.transpose(file_paths)[0]
# max_time_len = len(file_paths)

# Type
# file_paths = file_paths[0]
class FFTprocess:
    def __init__(self, point_n=9):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_paths = glob.glob(os.path.join(base_dir, 'data', '**', '*.CSV'), recursive=True)
        list.sort(self.file_paths)
        self.output_paths = [path.replace('data', 'dataOutputs') for path in self.file_paths]
        self.point_n = point_n
        self.fft_amp_max = 0.5
        self.resolution = 0.25
        self.lim_n = 4000
        self.color_min = -7
        self.color_max = 2

    def create_datas(self):
        # stft = np.array([])
        stft = []
        stft_acc_list = []
        stft_vel_list = []
        prior_folder_path = None
        for i, file_path in enumerate(self.file_paths):
            create_img = True # 이미지를 생성할 지, STFT 데이터세트를 만들기위한 작업만 할지 결정하는 플래그
            folder_path = self.output_paths[i].rpartition('/')[0]
            spectrogram_path = os.path.join(folder_path, 'Spectrogram')

            if os.path.exists(self.output_paths[i] + '.png'): # 이미지 파일이 있으면 건너뛴다. 스펙트로그램은 스펙트럼 다음에 수행되므로, 스펙트럼이 만들어졌다면 해당 스펙트로그램도 만들어져서 검사할 필요가 없다.
                print(i, self.output_paths[i],'already exists, skipping img creation')
                create_img = False
            else:
                print(i, self.output_paths[i])

            if not os.path.exists(spectrogram_path): # 스펙트로그램 폴더가 있는지 확인하고 만든다.
                pathlib.Path(spectrogram_path).mkdir(parents=True, exist_ok=True)
                print("created = ", spectrogram_path)

            if prior_folder_path != folder_path and prior_folder_path != None:
                stft_acc_list = np.stack(stft_acc_list)
                stft_vel_list = np.stack(stft_vel_list)

                np.save(prior_folder_path + '_Acc.npy', stft_acc_list)
                np.save(prior_folder_path + '_Vel.npy', stft_vel_list)

                stft_acc_list = []
                stft_vel_list = []
                print('TrainSet Saved')
            prior_folder_path = folder_path

            time, data = np.transpose(np.genfromtxt(file_path, delimiter=','))
            stft.append(data)

            if create_img == True:
                if i % 2 == 0:
                    ylabel = 'Amp (m/s^2) (rms)'
                else:
                    ylabel = 'Amp (m/s) (rms)'
                self.create_fft_img(data, self.output_paths[i], ylabel)

            if (i + 1) % (self.point_n * 2) == 0:
                # print('yeah', len(stft))
                stft_acc = np.stack([stft[acc] for acc in range(0, len(stft), 2)])
                stft_acc_list.append(stft_acc)

                stft_vel = np.stack([stft[acc] for acc in range(1, len(stft), 2)])
                stft_vel_list.append(stft_vel)

                if create_img == True:
                    self.create_stft_img(stft_acc, self.output_paths[i], '_Acc')
                    self.create_stft_img(stft_vel, self.output_paths[i], '_Vel')

                stft = []




            # with open(file_path, 'r', encoding='utf-8') as datas:
            #     datas = csv.reader(datas, quoting=csv.QUOTE_NONNUMERIC)
            #     for data in datas:
            #         print(data)
                # values = [float(data[1]) for data in datas]
                # print(values)
                # for data in datas:
                #     print(data[1])
                    # value = float(data[1])
                    # values.append(value)

    def create_stft_img(self, data, path, name):
        # LOG_MIN = -8
        # figure(figsize=[6.4, 4.8], dpi=100)

        # stft = convert2rms(stft, 16000, 'log10')
        # print(np.shape(stft))

        data = np.log10(data)
        # stft = np.nan_to_num(stft, neginf=LOG_MIN)
        plt.imshow(data, origin='lower', aspect='auto', interpolation='none')
        clb = plt.colorbar()

        plt.xlabel('Freq (Hz)')
        plt.ylabel('Point')
        clb.ax.set_title('log10 scale')

        plt.xlim(0, self.lim_n)
        plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x * self.resolution), ','), ))
        plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x + 1), ','), ))
        plt.clim(self.color_min, self.color_max)

        plt.tight_layout()
        spectrogram_path = path.rpartition('/')[0] + '/Spectrogram/'
        file_name = path.rpartition('/')[2]
        # print(file_name)
        file_name = file_name.rsplit('_', 2)[0]
        # print(path)
        # path.replace('CH', '')

        # try:
        plt.savefig(spectrogram_path + file_name + name + '.png')
        # except:
        #     os.mkdir(spectrogram_path)
        #     plt.savefig(spectrogram_path + file_name + name + '.png')
        plt.close()
        # plt.savefig(path + '/' + name + '.png')

    def create_fft_img(self, data, path, ylabel = 'Amp (m/s^2) (rms)'):

        plt.plot(np.arange(0, len(data)), data, linewidth=0.5)
        plt.grid(b=True)

        plt.xlabel('Freq (Hz)')
        plt.ylabel(ylabel)
        # plt.xlabel()
        # plt.ylabel()
        max_value = max(self.fft_amp_max, np.max(data))
        plt.ylim(0, max_value)
        plt.xlim(0, self.lim_n)
        plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(x * self.resolution, ','), ))
        # try:
        plt.savefig(path + '.png')
        # except:
        #     path_recursive = path.rpartition('/')[0]
        #     pathlib.Path(path_recursive).mkdir(parents=True, exist_ok=True)
        #     plt.savefig(path + '.png')
        plt.close()

                    # plt.savefig('fft.png')
            # stft = np.append(stft, values)
        # return stft.reshape([MAX_CH, -1])

        MAX_CH = 9
        # list.sort(file_paths)

        # file_paths = np.array(file_paths).reshape(-1, MAX_CH)

test = FFTprocess()
print(test.output_paths[0:10])
print(np.shape(test.output_paths))
test.create_datas()

def create_stft(file_paths):
    stft = np.array([])
    for file_path in file_paths:
        values = []
        with open(file_path, 'r', encoding='utf-8') as datas:
            datas = csv.reader(datas)
            for data in datas:
                # print(data[1])
                value = float(data[1])
                values.append(value)

        stft = np.append(stft, values)
    return stft.reshape([MAX_CH, -1])
# stft = create_stft(file_paths)
# stft = stft.reshape([max_time_len, -1])

def by_mean(stft, savefig='by_mean.png', cmap='jet'):
    MEAN_VALUE = 0.5

    convert2rms(stft, 1600)

    # figure()
    # max_values = 0
    # total_mean_value = 0
    # for ch in range(len(stft)):
    #     max_values += np.max(stft[ch])
    #     total_mean_value = max_values / len(stft)

    # print(total_mean_value)
    # plt.figure(figsize=[6.4, 4.8], dpi=100)
    plt.imshow(stft, origin='lower', aspect='auto', cmap=cmap, interpolation='none')
    plt.colorbar(format = '%2.0f m/s^2')
    plt.clim(0, MEAN_VALUE)
    plt.tight_layout()
    plt.xlim(0, 2500)
    plt.savefig(savefig)
    plt.close()
# by_mean(stft)

def by_log(stft, savefig='by_log.png', cmap='magma'):
    LOG_MIN = -18
    # figure(figsize=[6.4, 4.8], dpi=100)

    stft = convert2rms(stft, 16000, 'log2')
    # print(np.shape(stft))

    # stft = np.log10(stft)
    stft = np.nan_to_num(stft, neginf=LOG_MIN)
    plt.imshow(stft, origin='lower', aspect='auto', cmap=cmap, interpolation='none')
    plt.colorbar(format = 'log2 %2.0f m/s^2')
    plt.tight_layout()
    plt.xlim(0, 2500)
    plt.savefig(savefig)
    plt.close()
# by_log(stft)

def by_dB(stft, savefig='by_dB.png', cmap='magma'):
    # figure(figsize=[6.4, 4.8], dpi=100)

    stft = librosa.amplitude_to_db(stft)
    # librosa.display.specshow(stft)
    plt.imshow(stft, origin='lower', aspect='auto', cmap=cmap, interpolation='none')
    plt.colorbar(format = '%2.0f dB')
    plt.tight_layout()
    plt.xlim(0, 2500)
    plt.savefig(savefig)
    plt.close()
# by_dB(stft)

def convert2rms(datas, size, process='none'):
    datas = datas.reshape(9, size, -1)

    if process == 'sqrt':
        datas = np.sqrt(np.mean(datas, 2))
    elif process == 'log10':
        datas = np.log10(np.mean(datas, 2))
    elif process == 'log2':
        datas = np.log2(np.mean(datas, 2))

    return datas

def create_sem(stft, savefig = 'by_SEM.png'):
    # stft = stft.reshape(9, 1000, -1)
    # stft = np.mean(stft, 2)

    h = 40
    w = 40

    stft = convert2rms(stft, h*w, 'sqrt')

    img = stft.reshape(9, h, w)
    img = 255 * img
    img = img.astype(np.uint8)
    img_mult = Image.new('L', (w * 3, h * 3))

    for count, i in enumerate(range(0, w * 3, w)):
        for j in range(0, h * 3, h):
            img_mult.paste(Image.fromarray(img[count], 'L'), (i, j))
    img_mult.save(savefig)
# stft = create_stft(file_paths[0])
# by_dB(stft)
def create_fft_img():
    for file_n, file_path in enumerate(file_paths):
        stft = create_stft(file_path)
        for type_n in range(0, len(file_path)):
            plt.plot(range(0, 16000), stft[type_n], linewidth=0.5)
            plt.grid(b=True)
            # plt.xlabel()
            # plt.ylabel()
            plt.ylim(0, 2)
            plt.xlim(0, 2500)
            plt.savefig('FFTOutputs/fft' + str(file_n + 1) + str(type_n + 1) + '.png')
            plt.close()

        # for n in file_path
        # stft = create_stft(file_paths[0])
        # plt.plot(range(0, 16000), stft[0])
        # plt.grid(b=True)
        # plt.xlim(0, 2500)

        # plt.savefig('fft.png')
# create_fft_img()
    # print(np.shape(stft[0]))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # x = np.arange(0, 16000)
    # x = np.stack([x, x, x, x, x, x, x, x, x])
    # ax.plot(xs = x, ys=range(0,9), zs = stft)
    #     plt.imshow(stft[0])


def create_png():
    for i, file_path in enumerate(file_paths):
        stft = create_stft(file_path)

        if os.path.exists('./Outputs/') == False:
            os.mkdir('./Outputs')

        i += 1
        by_log(stft, './Outputs/' + 'by_log' + str(i) + '.png')
        # by_dB(stft, './Outputs/' + 'by_dB' + str(i) + '.png')
        # by_mean(stft, './Outputs/' + 'by_mean' + str(i) + '.png')
        # create_sem(stft, './Outputs/' + 'by_SEM' + str(i) + '.png')
# create_png()
# plt.subplots(3, 3, )
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(img[i], cmap='gray', interpolation='none')
#     plt.axis('off')
# plt.tight_layout()
# frame = plt.gca()

# plt.savefig('test.png')


# img = 255 * img
# img = img.astype(np.uint8)
#
# img = Image.fromarray(img, 'L')
# img.show()
# img.save('test.png')

# librosa.display.specshow(librosa.amplitude_to_db(stft, top_db=50), x_axis='linear', sr=32000)
# stft *= 10.0
# max_value *= 10.0
# ax = fig.add_subplot(111)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)

# ax.set_xlim([0, 100])




# plt.colorbar(im, cax)

# librosa.display.specshow(librosa.amplitude_to_db(stft), x_axis='coolwarm', sr=32000)
#
# plt.title('Power spectrogram')
# plt.xlabel('Frequency')
# plt.colorbar(format = '%2.0f dB')

# print(max_value)
# color맵만 변경하면 문제가 해결될듯
# frame = plt.gca()
# frame.axes.get_yaxis().set_visible(False)
# frame.axes.get_xaxis().set_visible(False)
# m/s^2 mm/s
# plt.gray()


# file_csv1 = open('20190916_144542_ch1_Vel.CSV', 'r', encoding='utf-8')
# file_csv2 = open('20190916_144542_ch2_Vel.CSV', 'r', encoding='utf-8')
# reader_csv1 = csv.reader(file_csv1)
# reader_csv2 = csv.reader(file_csv2)

# stft = np.array([])
# class Data:
#     freq_n = 0
#     freq = []
#
# data1 = Data()
# x = []
# y = []
# xmax = 0
# ymax = 0
# for i in reader_csv1:
#     x_float = float(i[0])
#     y_float = float(i[1])
#     x.append(x_float)
#     y.append(y_float)
#
#     if xmax < x_float:
#         xmax = x_float
#     if ymax < y_float:
#         ymax = y_float
#
# stft = np.append(stft, y)
# x = []
# y = []
# for i in reader_csv2:
#     x_float = float(i[0])
#     y_float = float(i[1])
#     x.append(x_float)
#     y.append(y_float)
#
#     if xmax < x_float:
#         xmax = x_float
#     if ymax < y_float:
#         ymax = y_float
# stft = np.append(stft, y)
# print(np.max)
# stft = stft.reshape([2, -1])
# stft = np.transpose(stft)
# print(np.shape(stft))
# print(stft[0])
# print(stft.dtype)


# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max))
# plt.title('Power spectrogram')
# plt.colorbar(format = '%2.0f dB')
# plt.tight_layout()
# plt.savefig('librosa.png')


# plt.plot(x, y)
# plt.xticks(np.arange(0, xmax, step=xmax * 0.1))
# plt.yticks(np.arange(0, ymax, ymax * 0.1))
# plt.grid()
# plt.savefig('fft.png')