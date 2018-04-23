# import numpy as np
# from tcspcfit import *
# from matplotlib import pyplot as plt
#
# window_ns = 200
# numPoints = 200
# chnlWidth_ns = window_ns/numPoints
# tauIRF_ns = 1
# A1 = 10000
# tau1_ns = 8
# A2 = 6000
# tau2_ns = 30
# delay_ns = 20
# addNoise = True
#
# perfConv = 1
# remBack = 1
#
# t = np.linspace(0, window_ns, numPoints)
# fData = A1 * np.exp(-t / tau1_ns) + A2 * np.exp(-t / tau2_ns)
# fData = np.array([fData])
# fData1 = np.array([A1 * np.exp(-t / tau1_ns)])
# fData2 = np.array([A2 * np.exp(-t / tau2_ns)])
# Airf = np.max(fData)
#
# delay_pts = delay_ns / (window_ns / numPoints)
#
# IRF = np.exp(-t / tauIRF_ns) / (1 + np.exp(-delay_ns * (t - delay_ns)))
#
# IRF = IRF - min(IRF.flatten())
# IRF = Airf * (IRF / np.max(IRF.flatten()))
# IRF = np.array([IRF])
# # IRF = colorshift(IRF, cshift, np.size(IRF.flatten()), t)
#
# # fData = [np.zeros(1, delay_pts) fData(1, 1:end - delay_pts)]
# # fData1 = [np.zeros(1, delay_pts) fData1(1, 1:end - delay_pts)]
# # fData2 = [np.zeros(1, delay_pts) fData2(1, 1:end - delay_pts)]
#
# TCSPCdata = fData
# if perfConv:
#     TCSPCdata = convol(IRF, fData)
#     TCSPCdata = TCSPCdata - min(TCSPCdata.flatten())
#     TCSPCdata = Airf * (TCSPCdata / max(TCSPCdata.flatten()))
#
# if addNoise:
#     # backG = Airf * 0.05 * np.ones(1, np.size(TCSPCdata, 1))
#     backG = Airf * 0.05 * np.ones([1, np.size(TCSPCdata, 1)])
#
#     TCSPCdata = np.random.poisson(TCSPCdata + backG)
#     fData1 = np.random.poisson(fData1 + backG)
#     fData2 = np.random.poisson(fData2 + backG)
#     IRF = np.random.poisson(IRF + backG)
#     if remBack:
#         backLevel = np.mean(TCSPCdata[0, 1: int(delay_pts) - 1])
#         IRF = IRF - backLevel
#         TCSPCdata = TCSPCdata - backLevel
#
# # err, A, z = lsfit(np.array([[1.5, 0.0004, 8, 30]]), IRF, TCSPCdata, 200)
# # print(err, A)
# # plt.plot(z.transpose()[0])
# # plt.plot(TCSPCdata[0])
# # plt.show()
#
# c, offset, A, tau, dc, dtau, irs, zz, t, chi = fluofit(IRF, TCSPCdata, window_ns, chnlWidth_ns, np.array([[tau1_ns, tau2_ns]]))
#
# # print("Amplitudes:", A)
# print("Decay times:", tau)
# print("Decay times error:", dtau)
# # print("Offset, cshift:", offset, c)
# # fitted = A[0] * np.exp(-t / tau[0]) + A[1] * np.exp(-t / tau[1])
# # plt.plot(4000000*fitted)
# # plt.plot(fData[0])
# # plt.plot(TCSPCdata[0])
# # # plt.ylim([-10, 300])
# # plt.show()

import numpy as np
from tcspcfit import *
from matplotlib import pyplot as plt


def gendata(window_ns, numPoints, tauIRF_ns, A1, tau1, A2, tau2, delay, addNoise):
    perfConv = 1
    remBack = 1

    t = np.linspace(0, window_ns, numPoints)
    fData = A1 * np.exp(-t / tau1_ns) + A2 * np.exp(-t / tau2_ns)
    fData = np.array([fData])
    fData1 = np.array([A1 * np.exp(-t / tau1_ns)])
    fData2 = np.array([A2 * np.exp(-t / tau2_ns)])
    Airf = np.max(fData)

    delay_pts = delay_ns / (window_ns / numPoints)

    IRF = np.exp(-t / tauIRF_ns) / (1 + np.exp(-delay_ns * (t - delay_ns)))

    IRF = IRF - min(IRF.flatten())
    IRF = Airf * (IRF / np.max(IRF.flatten()))
    IRF = np.array([IRF])

    # fData = [np.zeros(1, delay_pts) fData(1, 1:end - delay_pts)]
    # fData1 = [np.zeros(1, delay_pts) fData1(1, 1:end - delay_pts)]
    # fData2 = [np.zeros(1, delay_pts) fData2(1, 1:end - delay_pts)]

    TCSPCdata = fData
    if perfConv:
        TCSPCdata = convol(IRF, fData)
        TCSPCdata = TCSPCdata - min(TCSPCdata.flatten())
        TCSPCdata = Airf * (TCSPCdata / max(TCSPCdata.flatten()))

    if addNoise:
        # backG = Airf * 0.05 * np.ones(1, np.size(TCSPCdata, 1))
        backG = Airf * 0.05 * np.ones([1, np.size(TCSPCdata, 1)])

        TCSPCdata = np.random.poisson(TCSPCdata + backG)
        fData1 = np.random.poisson(fData1 + backG)
        fData2 = np.random.poisson(fData2 + backG)
        IRF = np.random.poisson(IRF + backG)
        if remBack:
            backLevel = np.mean(TCSPCdata[0, 1: int(delay_pts) - 1])
            IRF = IRF - backLevel
            TCSPCdata = TCSPCdata - backLevel

    return TCSPCdata, IRF



# window_ns = 200
# numPoints = 10000
# chnlWidth_ns = window_ns/numPoints
# tauIRF_ns = 1
# A1 = 10000
# tau1_ns = 8
# A2 = 6000
# tau2_ns = 30
# delay_ns = 20
# addNoise = True
#
# TCSPCdata, IRF = gendata(window_ns, numPoints, tauIRF_ns, A1, tau1_ns, A2, tau2_ns, delay_ns, addNoise)
# c, offset, A, tau, dc, dtau, irs, zz, t, chi = fluofit(IRF, TCSPCdata, window_ns, chnlWidth_ns, np.array([[8, 30]]))
#
# tau1list = np.array([])
# tau2list = np.array([])
# for count in range(1):
#     TCSPCdata, IRF = gendata(window_ns, numPoints, tauIRF_ns, A1, tau1_ns, A2, tau2_ns, delay_ns, addNoise)
#     c, offset, A, tau, dc, dtau, irs, zz, t, chi = fluofit(IRF, TCSPCdata, window_ns, chnlWidth_ns, np.array([[9, 30]]))
#     tau1list = np.append(tau1list, tau[0])
#     tau2list = np.append(tau2list, tau[1])
# print(np.mean(tau1list - 8))
# print(np.mean(tau2list - 30))
# # print("Amplitudes:", A)
# print("Decay times:", tau)
# print("Decay times error:", dtau)
# # fitted = A[0] * np.exp(-t / tau[0]) + A[1] * np.exp(-t / tau[1])
# # plt.plot(4000000*fitted)
# # plt.plot(fData[0])
# # plt.plot(TCSPCdata[0])
# # plt.ylim([-10, 300])
# # plt.show()

window_ns = 200
numPoints = 10000
chnlWidth_ns = window_ns/numPoints
tauIRF_ns = 1
A1 = 10000
tau1_ns = 8
A2 = 6000
tau2_ns = 30
delay_ns = 20
addNoise = True

tau1list = np.array([])
tau2list = np.array([])
for count in range(100):
    print(count)
    TCSPCdata, IRF = gendata(window_ns, numPoints, tauIRF_ns, A1, tau1_ns, A2, tau2_ns, delay_ns, addNoise)
    c, offset, A, tau, dc, dtau, irs, zz, t, chi = fluofit(IRF, TCSPCdata, window_ns, chnlWidth_ns, np.array([[8, 30]]), ploton=False)
    tau1list = np.append(tau1list, tau[0])
    tau2list = np.append(tau2list, tau[1])
print(np.mean(tau1list - 8))
print(np.mean(tau2list - 30))
# print("Amplitudes:", A)
print("Decay times:", tau)
print("Decay times error:", dtau)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(tau1list)
ax1.axhline(np.mean(tau1list))
ax2.plot(tau2list, color='C1')
ax2.axhline(np.mean(tau2list), color='C1')
fig.savefig('/home/bertus/Documents/Honneurs/Projek/python2.png', dpi=600)
plt.show()