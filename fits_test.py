import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from matplotlib import rc
from tcspcfit import *

rc('text', usetex=True)

fitlist = np.array([[0, 0, 0, 0]])

parameters = np.loadtxt('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/sukses.csv', skiprows=1)

for i in range(1):
    data = np.loadtxt('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/Decaytrace' + str(i))
    irfdata = np.loadtxt('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/savedata.txt')

    t = irfdata[:, 0]
    measured = data[:, 1]
    irf = irfdata[:, 2]
    irflength = np.size(irf)

    window = np.size(irf)
    channelwidth = max(t) / window
    print(channelwidth)

    param = parameters[i, :]

    model = param[0] * np.exp(-t / param[2]) + param[1] * np.exp(-t / param[3])

    irf = irf - param[6]
    irf = irf.clip(0)
    measured = measured - param[4]
    measured = measured.clip(0)
    # measured = measured / np.sum(measured)
    irf = irf * (np.sum(measured) / np.sum(irf))

    irs = colorshift(irf, param[5], np.size(irf), t).flatten()
    convd = convolve(irs, model)
    convd = convd / np.sum(convd) * np.sum(measured)

    t = t[0:np.size(measured)]
    irf = irf[0:np.size(measured)]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_yscale('log')
    ax1.set_ylim(0.5, np.max(irf))
    ax1.plot(t, irf)
    ax1.plot(t, measured)
    ax1.plot(t, convd[0:np.size(measured)])
    # plt.xlim((400, 1000))
    # plt.plot(t)
    ax2.plot(t[0:3000], convd[0:3000] - measured[0:3000], '.')
    # residuals = np.sum((convd[0:3000] - measured[0:3000]) ** 2)
    # plt.text(2000, -10, str(residuals))
    # plt.show()
    plt.savefig('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/plots/decay%d' % i)
    plt.clf()
