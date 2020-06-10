import numpy as np
import os
from nptdms import TdmsFile
import glob
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import math
import scipy
from scipy import integrate
from scipy.ndimage.interpolation import shift
import shutil

class RawPreprocess:
    def __init__(self, dataset_dir, output_dir):
        # 이곳에는 각종 경로 변수들을 선언한다.
        self.path_base = os.path.dirname(os.path.realpath(__file__))
        self.path_output = os.path.join(self.path_base, output_dir)
        self.path_spectrogram = os.path.join(self.path_output, 'STFT')
        self.path_spectrum = os.path.join(self.path_output, 'FFT')
        
        self.path_Acc = os.path.join(self.path_output, 'Acc')
        self.path_Vel = os.path.join(self.path_output, 'Vel')
        self.path_FFT2Vel =  os.path.join(self.path_output, 'FFT2Vel')
        self.path_Vel2FFT = os.path.join(self.path_output, 'Vel2FFT')
        self.path_Acc_FFT2Vel_FFT = os.path.join(self.path_output, 'Acc_FTT2Vel_FTT')
        self.path_Compare = os.path.join(self.path_output, 'Compare')




        self.file_paths = glob.glob(os.path.join(self.path_base, dataset_dir, '*.tdms'))
        list.sort(self.file_paths)  # 뒤죽박죽이라서 정렬해야함
        # 파일이름만
        self.filename_list = [f.rpartition('.')[0] for f in os.listdir(os.path.join(self.path_base, dataset_dir)) if f.endswith('.tdms')]
        list.sort(self.filename_list)  # 뒤죽박죽이라서 정렬해야함

    def create_paths(self, paths):
        # 경로가 없다면 경로를 만든다.
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)

    def fft(self, datas, norm=True, skip=0, fft_n=None):
        # fft를 실행한다.
        # fft = np.fft.fft(datas)  # np fft를 사용
        # fft = abs(np.array_split(fft, 2, axis=1)[0]) / np.shape(datas)[1]  # 전처리 : one-side 처리 -> 절대값 처리 -> 정규화(normalizing)
        if fft_n == None:
            fft = np.fft.rfft(datas)  # np fft를 사용
        else:
            fft = np.fft.rfft(datas, n=fft_n)
        fft = abs(fft)
        if norm == True:
            fft = fft / np.shape(datas)[1]  # 전처리 : one-side 처리 -> 절대값 처리 -> 정규화(normalizing)

        if skip != 0:
            fft = fft[skip:]

        return fft

    def create_fft_img(self, fft, name, ylabel='Amp (m/s^2)'):
        # 포인트별 스펙트럼 이미지를 만들어낸다.
        # 사용자가 확인하기 위해 만드는 이미지이다.
        # 실제 딥러닝에는 사용되지 않는다.
        for fft_n in range(0, np.shape(fft)[0]):  # fft 이미지 생성
            save_dir = os.path.join(self.path_spectrum, str(fft_n + 1))
            if not os.path.exists(save_dir):
                self.create_paths([save_dir])

            img_path = os.path.join(save_dir, name + '.png')
            # if os.path.exists(img_path):
            #     print(img_path, 'already exists')
            #     continue

            plt.plot(range(0, int(np.shape(fft)[1])), fft[fft_n], linewidth=0.5)
            plt.grid(b=True)
            plt.xlabel('Freq (Hz)')
            plt.ylabel(ylabel)

            max_value = max(0.3, np.max(fft[fft_n]))  # 최소 1 초과하면 더 크게
            plt.ylim(0, max_value)
            # plt.xlim(0,)

            plt.savefig(img_path)
            plt.close()

    def create_stft_img(self, fft, name):
        # 스펙트로그램 이미지를 만들어낸다.
        # 실제로 딥러닝에 사용될 이미지는 아니다.
        # 딥러닝에 사용될 이미지는 행렬 형태로 .npy 파일에 저장할 생각이다. 그쪽이 훨씬 저장과 활용이 용이하다.
        # 이 함수에서 생성하는 이미지는 사용자가 확인할 이미지이다.
        img_path = os.path.join(self.path_spectrogram, name + '.png')
        # if os.path.exists(img_path):
        #     print(img_path, "already exists")
        #     return None

        fft = np.log10(fft)
        plt.imshow(fft, aspect='auto', interpolation='none')
        # plt.xlim(0, 2500)
        clb = plt.colorbar()
        clb.ax.set_title('log10 scale')

        # plt.yticks(np.arange(0, 12, 1))
        # plt.clabel('log10 scale')
        plt.xlabel('Freq (Hz)')
        plt.ylabel('Point')

        plt.clim(-8, 1)
        # plt.xlim(0, )
        # plt.clim(-2, 6)
        plt.savefig(img_path)
        plt.close()

    def create_velocity(self, datas, sample_rate, integrate_range_rate=0.015):
        # 적분을 통해 속도를 계산한다.
        # 전체 기간을 적분하면, 오차값이 쌓여서 그런지 값이 발산하는 경향이 있다. 따라서, 적분 구간(범위)를 한정하였다.
        vel_list = []

        integrate_n = int(np.shape(datas)[1] * integrate_range_rate)
        # while(integrate_n * sample_rate < 1):
        #     integrate_n = integrate_n + 1

        for point_n in range(0, np.shape(datas)[0]):
            # print("point n ", point_n)
            vel = np.zeros(np.shape(datas)[1]) # 장비는 고정되어있으므로, 속도는 쌓이지 않고 진동만 한다고 가정하여 0으로 둔다.
            # vel[-1] = vel_init[point_n] # for문에서 vel[-1]을 사용하므로 -1을 초기값으로 활용
            # acc_prior = 0
            # acc_list = np.zeros(integrate_n)
            initial_value = 0.
            data_avg = np.mean(datas[point_n])
            datas_test = datas[point_n] - data_avg

            vel = scipy.integrate.cumtrapz(datas_test, dx=sample_rate, initial=0)

            # for data_n, acc in enumerate(datas[point_n]):
            #     data_avg = np.mean(datas[point_n])
            #     datas_test = datas[point_n] - data_avg
            #
            #     if data_n >= integrate_n:
            #         vel[data_n] = np.trapz(datas_test[data_n - integrate_n:data_n], dx=sample_rate) # 속도 계산.
            #     else:
            #         vel[data_n] = np.trapz(datas_test[0:data_n], dx=sample_rate)

                # integrate_n = int(64)
            #     # 심프슨 ( 점(샘플)들을 연결하는 고차 방정식을 만들어 적분)
            #     if data_n >= integrate_n:
            #         vel[data_n] = integrate.simps(datas[point_n][data_n - integrate_n:data_n], dx=sample_rate)# 속도 계산. vel[-1]은 끝을 가리킨다 = 초기값
            #     elif data_n >= 2:
            #         vel[data_n] = integrate.simps(datas[point_n][0:data_n], dx=sample_rate)
            #     else:
            #         vel[data_n] = integrate.trapz(datas[point_n][0:data_n], dx=sample_rate)

                # 롬버그 (사다리꼴 계산에 리차드슨 외삽법(수렴률 향상) 사용)
                    # print(data_n)
                # if data_n >= integrate_n + 1:
                #     vel[data_n] = integrate.romb(datas[point_n][data_n - integrate_n - 1:data_n], dx=1) # 속도 계산. vel[-1]은 끝을 가리킨다 = 초기값
                # else:
                #     vel[data_n] = integrate.trapz(datas[point_n][0:data_n], dx=1)

                # 사다리꼴

                # if data_n >= integrate_n:
                #     vel[data_n] = integrate.trapz(datas[point_n][data_n - integrate_n:data_n], dx=1) # 속도 계산. vel[-1]은 끝을 가리킨다 = 초기값
                # else:
                #     vel[data_n] = integrate.trapz(datas[point_n][0:data_n], dx=1)
                #
                # if data_n >= integrate_n:
                #     vel[data_n] = np.trapz(datas[point_n][data_n - integrate_n:data_n], dx=sample_rate) # 속도 계산.
                # else:
                #     vel[data_n] = np.trapz(datas[point_n][0:data_n], dx=sample_rate)

            vel = vel / np.shape(datas)[1]

            # center = (np.max(vel) + np.min(vel)) / 2.0
            # vel = vel - center

            # avg = np.average(vel)
            # vel = vel - avg





                #

                # 커스텀 사다리꼴
                # acc_list = np.roll(acc_list, -1) # 큐 구조
                # acc_list[-1] = acc
                # # acc_list[data_n % integrate_n] = acc
                #
                # if data_n >= integrate_n:
                #     for acc_n in range(1, len(acc_list)): # 적분은 값과 값(샘플) 사이에 나오므로 1부터 시작하는게 맞다.
                #         vel[data_n] = vel[data_n] + (acc_list[acc_n] + acc_list[acc_n - 1]) / 2.0
            vel_list.append(vel)
            # vel_init[point_n] = vel[-1]
        vel_list = np.stack(vel_list) # 포인트 별 속도

        return vel_list

    def calculate_sample_rate(self, file_paths, file_n):
        # 파일명에 적힌 날짜와 다음 파일명에 적힌 날짜를 비교하여 기간을 알아내고, 데이터 개수를 측정하여 샘플링 레이트를 알아낸다.
        with TdmsFile.read(file_paths[file_n]) as tdms_file:
            df = tdms_file.as_dataframe(scaled_data=False)  # pandas 데이터프레임
            datas = df.to_numpy()  # 이래야 사용 가능
            datas = np.transpose(datas)  # 안 뒤집으면 fft 방향이 잘못됨

        time1 = datetime.datetime.strptime(self.filename_list[file_n], '%Y%m%d_%H%M%S')
        time2 = datetime.datetime.strptime(self.filename_list[file_n + 1], '%Y%m%d_%H%M%S')

        time_diff = time2 - time1

        sample_rate = time_diff.seconds / np.shape(datas)[1]

        return sample_rate

    def fft_integration(self, fft, skipHz=0, sample_rate=1.):
        fft_integral = np.zeros_like(fft)

        # fft 적분 => fft / (1j * 2 * pi * f) f는 전체 주파수 영역(ex> 1~6400), 1j = sqrt(-1)

        for point in range(0, np.shape(fft_integral)[0]):
            for freq in range(0, np.shape(fft_integral)[1]):
                if freq <= skipHz:  # 5Hz 까지는 연산하지 않는다.(자른다)
                    continue
                fft_integral[point][freq] = fft[point][freq] / (2 * np.pi * ( (1 / sample_rate ) * freq + 1))
        return fft_integral

    def fft_compare(self, skipHz = 0):
        shutil.rmtree('./test')

        data_size = 0
        sample_rate = 1
        compare_avg_list = []

        file_paths = self.file_paths[0:1]

        self.create_paths([self.path_output, self.path_Acc, self.path_Vel, self.path_FFT2Vel, self.path_Vel2FFT, self.path_Acc_FFT2Vel_FFT, self.path_Compare])

        for file_n, file_path in enumerate(file_paths):
            with TdmsFile.read(file_path) as tdms_file:
                df = tdms_file.as_dataframe(scaled_data=False)  # pandas 데이터프레임
                datas = df.to_numpy()  # 이래야 사용 가능
                datas = np.transpose(datas)  # 안 뒤집으면 fft 방향이 잘못됨

            data_value_n = np.shape(datas)[1]
            data_point_n = np.shape(datas)[0]

            if data_value_n != data_size:
                sample_rate = self.calculate_sample_rate(file_paths, file_n)
                data_size = np.shape(datas)[1]
                print("sample_rate is modified to", sample_rate)

            datas_vel = self.create_velocity(datas, sample_rate)
            for point_n in range(0, data_point_n):
                file_name = str(file_n) + '_' + str(point_n) + '.png'
                ymax = 0

                plt.plot(range(0, data_value_n), datas[point_n], linewidth=0.5)
                plt.savefig(os.path.join(self.path_Acc, 'Acc_' + file_name))
                plt.close()

                plt.plot(range(0, data_value_n), datas_vel[point_n], linewidth=0.5)
                plt.savefig(os.path.join(self.path_Vel, 'Vel_' + file_name))
                plt.close()

                # fft 미분 => 속도 FFT 생성
                # fft_acc = np.fft.fft(datas)
                fft_acc = np.fft.rfft(datas)
                fft_vel = self.fft_integration(fft_acc, skipHz=skipHz, sample_rate=sample_rate)
                # fft_plot = abs(np.array_split(fft_vel[point_n], 2)[0])
                fft_plot = abs(fft_vel[point_n])
                ymax = max(ymax, np.max(fft_plot))
                plt.plot(range(0, np.shape(fft_plot)[0]), fft_plot, linewidth=0.5)
                plt.savefig(os.path.join(self.path_Acc_FFT2Vel_FFT, 'Acc_FTT2Vel_FTT_' + file_name))
                plt.close()

                # 역 푸리에 변환 => 속도 생성
                inv_fft = np.fft.irfft(fft_vel)
                # inv_fft = inv_fft.astype(np.float)
                plt.plot(range(0, np.shape(inv_fft)[1]), inv_fft[point_n], linewidth=0.5)
                plt.savefig(os.path.join(self.path_FFT2Vel, 'FTT2Vel' + file_name))
                plt.close()

                # 속도를 FFT 변환 => 속도 FFT 생성
                # vel_fft = np.fft.fft(datas_vel)
                vel_fft = np.fft.rfft(datas_vel)
                if skipHz != 0:
                    vel_fft[point_n][0:skipHz] = 0.0 # 저주파 제거
                # vel_fft_plot = abs(np.array_split(vel_fft[point_n], 2)[0])
                vel_fft_plot = abs(vel_fft[point_n])
                ymax = max(ymax, np.max(fft_plot))

                plt.plot(range(0, int(np.shape(vel_fft_plot)[0])), vel_fft_plot, linewidth=0.5)
                plt.savefig(os.path.join(self.path_Vel2FFT, 'Vel2FTT' + file_name))
                plt.close()

                # 속도 FFT 2개를 서로 비교
                fft_compare = abs(np.subtract(fft_plot, vel_fft_plot))

                plt.plot(range(0, np.shape(fft_compare)[0]), fft_compare, linewidth=0.5)
                plt.ylim(0.0, ymax)
                plt.savefig(os.path.join(self.path_Compare, 'Compare_' + file_name))
                plt.close()

                compare_avg = np.mean(fft_compare)
                print("error =", compare_avg)
                compare_avg_list.append(compare_avg)

        compare_avg_list = np.mean(np.stack(compare_avg_list))
        print("mean error =", compare_avg_list)

    def create_fft_stft(self, create_img=True):
        # 스텍트럼과 스펙트로그램을 생성한다.
        # 만일 파일이 존재한다면 통과한다.
        # 경로가 없다면 경로를 만들어서 저장한다.
        self.create_paths([self.path_output, self.path_spectrogram, self.path_spectrum])

        # sample_rate = self.calculate_sample_rate(self.file_paths, 0)
        stft_acc_list = []
        stft_vel_list = []
        data_size = 0
        for file_n, file_path in enumerate(self.file_paths):
            print(file_n, self.filename_list[file_n])

            # 파일있으면 통과
            # 루프의 가장 마지막에 생성되는 파일이 있는지 검사한다.
            img_path = os.path.join(self.path_spectrogram, self.filename_list[file_n] + '_Vel.png')
            if os.path.exists(img_path):
                print(img_path, "already exists")
                continue

            with TdmsFile.read(file_path) as tdms_file:
                df = tdms_file.as_dataframe(scaled_data=False)  # pandas 데이터프레임
                datas = df.to_numpy()  # 이래야 사용 가능
                datas = np.transpose(datas)  # 안 뒤집으면 fft 방향이 잘못됨

                if np.shape(datas)[1] != data_size:
                    sample_rate = self.calculate_sample_rate(self.file_paths, file_n)
                    data_size = np.shape(datas)[1]
                    print("sample_rate is modified to", sample_rate)

                datas_vel = self.create_velocity(datas, sample_rate)

                fft = self.fft(datas, fft_n=25600)
                fft_vel = self.fft(datas_vel, norm=False, skip=5, fft_n=25600)
                if create_img == True:
                    self.create_fft_img(fft, self.filename_list[file_n] + '_Acc')
                    self.create_fft_img(fft_vel, self.filename_list[file_n] + '_Vel', 'Amp (m/s)')

                    self.create_stft_img(fft, self.filename_list[file_n] + '_Acc')
                    self.create_stft_img(fft_vel, self.filename_list[file_n] + '_Vel')
                stft_acc_list.append(fft)
                stft_vel_list.append(fft_vel)

                # fft_no_process = fft_no_process.astype(np.float)
                # rows = np.shape(datas)[1]
                # cols = np.shape(datas)[0]
                # for point_n in range(0, 12):
                #     ymax = 0
                #
                #     plt.plot(range(0, rows), datas[point_n], linewidth=0.5)
                #     plt.savefig('acc/acc_' + str(point_n) + '.png')
                #     plt.close()
                #
                #     plt.plot(range(0, rows), datas_vel[point_n], linewidth=0.5)
                #     plt.savefig('vel/vel_' + str(point_n) + '.png')
                #     plt.close()
                #
                #
                #     # fft 미분 => 속도 FFT 생성
                #     # fft_acc = np.fft.fft(datas)
                #     fft_acc = np.fft.rfft(datas)
                #     fft_vel = self.fft_integration(fft_acc, skipHz=0, sample_rate=sample_rate)
                #     # fft_plot = abs(np.array_split(fft_vel[point_n], 2)[0])
                #     fft_plot = abs(fft_vel[point_n])
                #     ymax = max(ymax ,np.max(fft_plot))
                #     # fft_nope = fft_nope / (-1j * 2 * np.pi * np.arange(0, 6400)) # X(f) / (2 * pi * f)
                #     plt.plot(range(0, np.shape(fft_plot)[0]), fft_plot, linewidth=0.5)
                #     plt.savefig('fft_vel/fft_vel_' + str(point_n) + '.png')
                #     plt.close()
                #
                #
                #     # 역 푸리에 변환 => 속도 생성
                #     inv_fft = np.fft.irfft(fft_vel)
                #     # inv_fft = inv_fft.astype(np.float)
                #     plt.plot(range(0, np.shape(inv_fft)[1]), inv_fft[point_n], linewidth=0.5)
                #     plt.savefig('invfft2vel/invfft2vel_' + str(point_n) + '.png')
                #     plt.close()
                #
                #     # 속도를 FFT 변환 => 속도 FFT 생성
                #     # vel_fft = np.fft.fft(datas_vel)
                #
                #     vel_fft = np.fft.rfft(datas_vel)
                #     # vel_fft[point_n][0:5] = 0.0
                #     # vel_fft_plot = abs(np.array_split(vel_fft[point_n], 2)[0])
                #     vel_fft_plot = abs(vel_fft[point_n])
                #     ymax = max(ymax ,np.max(fft_plot))
                #
                #     plt.plot(range(0, int(np.shape(vel_fft_plot)[0])), vel_fft_plot, linewidth=0.5)
                #     plt.savefig('./vel_fft/vel_fft_' + str(point_n) + '.png')
                #     plt.close()
                #
                #     # 속도 FFT 2개를 서로 비교
                #     # inv_fft = inv_fft.astype(np.float)
                #     # diff = abs(datas_vel[0] - inv_fft[point_n])
                #     # plt.plot(range(0, np.shape(diff)[0]), diff, linewidth=0.5)
                #     # plt.savefig('./diff/diff_' + str(point_n) + '.png')
                #     # plt.close()
                #     # print(np.mean(abs(datas_vel[point_n] - inv_fft[point_n])))
                #
                #     # 속도 FFT 2개를 서로 비교
                #     fft_comp = abs(np.subtract(fft_plot, vel_fft_plot))
                #
                #     # print(np.mean(diff))
                #
                #     # datas_vel_shift = np.roll(datas_vel[point_n], -2)[point_n]
                #     # diff = abs(np.subtract(datas_vel_shift, inv_fft[point_n]))
                #     plt.plot(range(0, np.shape(fft_comp)[0]), fft_comp, linewidth=0.5)
                #     plt.ylim(0.0, ymax)
                #     plt.savefig('./diff/diff_' + str(point_n) + '.png')
                #     plt.close()
                #     print(np.mean(fft_comp))



        stft_acc_list = np.stack(stft_acc_list)
        stft_vel_list = np.stack(stft_vel_list)

        np.save(os.path.join(self.path_base, 'stft_acc.npy'), stft_acc_list)
        np.save(os.path.join(self.path_base, 'stft_vel.npy'), stft_vel_list)

p = RawPreprocess('RawDatas', 'test')

p.create_fft_stft(create_img=True)
# p.fft_compare(skipHz=0)

'''
base_dir = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists('RawOutputs'):
    os.mkdir('RawOutputs')



# 2개 다 쓰는건 비효율적일수있지만, 오래 걸리지 않으니 상관없을
# 경로 포함 이름
file_paths = glob.glob(os.path.join(base_dir, 'RawDatas', '*.tdms'))
list.sort(file_paths) # 파일 이름이 뒤죽박죽이라서 정렬해야함
# 파일이름만
filename_list = [f for f in os.listdir(os.path.join(base_dir, 'RawDatas')) if f.endswith('.tdms')]
list.sort(filename_list) # 파일 이름이 뒤죽박죽이라서 정렬해야함
print(file_paths)
# file_paths = file_paths[:5]


# 포인트별 stft
fft_list = []
for file_n, file_path in enumerate(file_paths):
    with TdmsFile.read(file_path) as tdms_file:
        df = tdms_file.as_dataframe(scaled_data=False) # pandas 데이터프레임
        datas = df.to_numpy() # 이래야 사용 가능
        datas = np.transpose(datas) # 안 뒤집으면 fft 방향이 잘못됨
        print(np.shape(datas), np.max(datas))

        # fft_list = []
        # for i in range(0, np.shape(datas)[0]):
            # fft = abs(np.fft.fft(np.transpose(datas[i])))
        # fft = np.transpose(abs(np.fft.fft(datas[i])))
        fft = abs(np.fft.fft(datas, n=25600)) / 25600 # 정규화(normalize)와 절대값을 취한다.

        # print(np.shape(fft))
        fft = np.array_split(fft, 2, axis=1)[0] # oneside를 만듦. 안 하면 two side(대칭)이 됨.

        # print(np.shape(fft))

        # fft = fft.astype(float)
            # print('fft', np.shape(fft))
        # fft_list.append(fft)
        # fft_list = np.stack(fft_list)
        # print(np.shape(fft_list))

        # fft_list = np.transpose(fft_list)
        # plt.imshow
        for fft_n in range(0, np.shape(fft)[0]): # fft 이미지 생성
            plt.plot(range(0, int(25600 / 2)), fft[fft_n])
            plt.savefig('Raw2FFT/fft'+ str(file_n + 1) + '_' + str(fft_n + 1) + '.png')
            plt.close()

        fft = np.log10(fft)
        plt.imshow(fft, aspect='auto', interpolation='none')
        # plt.xlim(0, 2500)
        plt.colorbar()
        plt.clim(-8, 1)
        # plt.clim(-2, 6)
        plt.savefig('RawOutputs/stft' + str(file_n + 1) + '.png')
        plt.close()

        # print(np.shape(fft))
'''