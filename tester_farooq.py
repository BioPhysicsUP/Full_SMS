import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from matplotlib import rc
from tcspcfit import *

rc('text', usetex=True)

fitlist = np.array([[0, 0, 0, 0]])
for i in range(1):
    data = np.loadtxt('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/Decaytrace' + str(i))
    irfdata = np.loadtxt('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/IRF_data.txt')
    # # simdata = np.loadtxt('/home/bertus/temp/convd.txt')

    # a1 = 0.624
    # a2 = 0.423

    t = irfdata[:, 0]
    measured = data[:, 1]
    irf = irfdata[:, 1]
    irflength = np.size(irf)

    # measured = measured * (np.max(irf) / np.max(measured))
    # plt.plot(measured)
    # plt.plot(irf)
    # plt.show()
    # measured, blabla = np.histogram(meas_data, bins=1000)
    # scale = np.max(measured)
    # savedata = np.column_stack((t[:-1], measured, irf))
    # np.savetxt('savedata.txt', savedata)

    window = np.size(irf)
    channelwidth = max(t) / window
    print(channelwidth)

    # window = 200

    # irf = irf - 31
    # irf = irf * (np.max(measured) / np.max(irf))
    # irf = colorshift(irf, -50, 0, 0).flatten()

    model = 11.19 * np.exp(-t / 2.52) + 39.5 * np.exp(-t / 0.336)

    irf = irf - 31
    irf = irf.clip(0)
    measured = measured - 0.535
    measured = measured.clip(0)
    # measured = measured / np.sum(measured)
    irf = irf * (np.sum(measured) / np.sum(irf))

    irf = colorshift(irf, -15, np.size(irf), t).flatten()
    convd = convolve(irf, model)
    convd = convd / np.sum(convd) * np.sum(measured)
    # plt.plot(irf)
    # plt.plot(model)
    # plt.plot(measured)
    # plt.xlim((400, 1000))
    # plt.plot(t)
    plt.plot(convd[0:3000] - measured[0:3000], '.')
    residuals = np.sum((convd[0:3000] - measured[0:3000]) ** 2)
    plt.text(2000, -10, str(residuals))
    plt.show()

    # fit = fluofit(irf, measured, t, window, channelwidth, tau=[2.52, 0.336], startpoint=300, endpoint=3000, ploton=False)
    # print(np.array([[fit[3][0], fit[3][1], fit[5][0], fit[5][1]]]))
    # fitlist = np.append(fitlist, np.array([[fit[3][0], fit[3][1], fit[5][0], fit[5][1]]]), axis=0)

# np.savetxt('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/fitlist.txt', fitlist, fmt='%5.3f')

