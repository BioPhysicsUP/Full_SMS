import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from matplotlib import rc
from tcspcfit import *

rc('text', usetex=True)

fitlist = np.array([[0, 0, 0, 0]])
for i in range(21):
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

    # window = 200
    numPoints = window / channelwidth
    # channelwidth = window/numPoints
    tauIRF_ns = 1
    a1 = 0.05
    tau1 = 9
    a2 = 0.5
    tau2 = 1.7
    a3 = 0.2
    tau3 = 0.5
    a4 = 1.4
    tau4 = 0.04
    delay_ns = 20

    startpoint = 0
    endpoint = 4000
    model_in = np.append(t, irf)
    model_in = np.append(model_in, startpoint)
    model_in = np.append(model_in, endpoint)
    model_in = np.append(model_in, irflength)
    # measured = three_exp(model_in, tau1, tau2, tau3, np.max(irf), a1, a2, a3, 0)
    # measured = four_exp(model_in, tau1, tau2, tau3, tau4, np.max(irf), a1, a2, a3, a4, 0)
    # measured = one_exp(model_in, tau1, np.max(irf), a1, 0)
    measured = np.abs(measured)
    # measured = np.random.poisson(measured)

    irf = irf * (np.max(measured) / np.max(irf))
    # plt.plot(irf)
    # irf = colorshift(irf, -100, np.size(irf), t).flatten()
    # plt.plot(measured)
    # plt.plot(irf)
    # plt.xlim((400, 1000))
    # plt.plot(t)
    # plt.show()

    fit = fluofit(irf, measured, t, window, channelwidth, tau=[3.6, 0.3], startpoint=300, endpoint=3000, ploton=False)
    print(np.array([[fit[3][0], fit[3][1], fit[5][0], fit[5][1]]]))
    fitlist = np.append(fitlist, np.array([[fit[3][0], fit[3][1], fit[5][0], fit[5][1]]]), axis=0)

np.savetxt('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/fitlist.txt', fitlist, fmt='%5.3f')

