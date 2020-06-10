import numpy as np
import matplotlib.pyplot as plt

import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt

from scipy import signal
import matplotlib.pyplot as plt

def cwt_obspy():
    st = obspy.read()
    tr = st[0]
    npts = tr.stats.npts
    dt = tr.stats.delta
    t = np.linspace(0, dt * npts, npts)
    f_min = 1
    f_max = 50

    scalogram = cwt(tr.data, dt, 8, f_min, f_max)
    plt.imshow(abs(scalogram), aspect='auto')

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # x, y = np.meshgrid(t, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
    #
    # ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
    # ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
    # ax.set_ylabel("Frequency [Hz]")
    # ax.set_yscale('log')
    # ax.set_ylim(f_min, f_max)
    # plt.tight_layout()
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)
    plt.savefig('cwt_obspy.png')
    plt.close()
cwt_obspy()

def cwt_scipy():
    st = obspy.read()[0].data

    t = np.linspace(-1, 1, 200, endpoint=False)
    sig = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)


    print(np.shape(st), st.dtype)
    print(np.shape(sig), sig.dtype)
    widths = np.arange(1, 100)
    cwtmatr = abs(signal.cwt(st, signal.morlet2, widths))

    # t = np.linspace(0, dt * npts, npts)
    # x, y = np.meshgrid(t, np.logspace(np.log10(1), np.log10(50), cwtmatr.shape[0]))

    # print(np.shape(y))

    # plt.pcolormesh(x, y, np.abs(cwtmatr), cmap=obspy_sequential)


    # plt.yscale('log')
    plt.imshow(cwtmatr, aspect='auto')

    # plt.imshow(cwtmatr, extent=[-1, 1, 1, 30], cmap='PRGn', aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.savefig('cwt_scipy.png')
    plt.close()
cwt_scipy()