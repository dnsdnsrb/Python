from scipy import signal
import csv
import matplotlib.pyplot as plt
import numpy as np


# aaa = np.linspace(1.0, 10000.0, 10000)
# bbb = aaa
# bbb = aaa + 1j*aaa
# print(np.shape(bbb), bbb[3])
# test = signal.stft(bbb)
#



fs = 10e3
N = 1e5
pi = np.pi
#
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500 * np.cos(2 * pi * 0.25 * time)
carrier = amp * np.sin(2 * pi * 3e3 * time + mod)
# carrier = np.linspace(1.0, 10000.0, 10000)
noise = np.random.normal(scale=np.sqrt(noise_power))
x = carrier + noise

f, t, Zxx = signal.stft(x, fs, nperseg=1000)
print(np.shape(f), np.shape(t), np.shape(Zxx))
print(np.shape(x))
print(np.shape(Zxx[0]))

print(Zxx[0])
# f, t, Zxx = signal.stft(y, fs = 1, nperseg=1000)

plt.pcolormesh(t, f, np.abs(Zxx), vmin = 0, vmax = amp)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [HZ]')
plt.xlabel('Time [sec]')
plt.savefig('stft.png')

