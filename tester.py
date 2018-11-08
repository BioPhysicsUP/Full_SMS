import numpy as np
from scipy.signal import convolve
from tcspcfit import *
from matplotlib import pyplot as plt
import os


def gendata(window_ns, numpoints, tauirf_ns, delay_ns, ampl1, tau1, ampl2, tau2, addnoise, perfconv=True, remBack = 0):

    times = np.linspace(0, window_ns, numpoints)
    fdata = ampl1 * np.exp(-times / tau1) + ampl2 * np.exp(-times / tau2)
    # fdata = np.array(fdata)
    airf = np.max(fdata)

    delay_pts = delay_ns / (window_ns / numpoints)

    irf = np.exp(-times / tauirf_ns) / (1 + np.exp(-delay_ns * (times - delay_ns)))

    irf = irf - min(irf.flatten())
    irf = airf * (irf / np.max(irf.flatten()))
    irf = np.array(irf)

    # fData = [np.zeros(1, delay_pts) fData(1, 1:end - delay_pts)]
    # fData1 = [np.zeros(1, delay_pts) fData1(1, 1:end - delay_pts)]
    # fData2 = [np.zeros(1, delay_pts) fData2(1, 1:end - delay_pts)]

    tcspcdata = fdata
    if perfconv:
        print('lala')
        tcspcdata = convolve(irf, fdata)[:numpoints]
        tcspcdata = tcspcdata - min(tcspcdata.flatten())
        tcspcdata = airf * (tcspcdata / max(tcspcdata.flatten()))

    if addnoise:
        # backG = Airf * 0.05 * np.ones(1, np.size(TCSPCdata, 1))
        backg = airf * 0.05 * np.ones(np.size(tcspcdata))

        tcspcdata = np.random.poisson(tcspcdata + backg)
        irf = np.random.poisson(irf + backg)
    if remBack:
        backlevel = np.mean(tcspcdata[1: int(delay_pts) - 1])
        irf = irf - backlevel
        tcspcdata = tcspcdata - backlevel

    return tcspcdata, irf


window = 200
numPoints = 10000
chnlWidth_ns = window / numPoints
tauIRF_ns = 1
A1 = 10000
tau1_ns = 9
A2 = 6000
tau2_ns = 30
delay_ns = 20
addNoise = False

TCSPCdata, IRF = gendata(window, numPoints, tauIRF_ns, delay_ns, A1, tau1_ns, A2, tau2_ns, addNoise)
t = np.linspace(0, window, numPoints)

# plt.plot(I/RF)
# plt.show()

fit = TwoExp(IRF, TCSPCdata, t, chnlWidth_ns, [tau1_ns, tau2_ns], ploton=True)


