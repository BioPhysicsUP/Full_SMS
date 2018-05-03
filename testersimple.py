import numpy as np
from tcspcfit import *
from matplotlib import pyplot as plt
# import tracemalloc

# tracemalloc.start()


# def gendata(window_ns, numPoints, tauIRF_ns, A1, tau1, A2, tau2, delay, addNoise):
#     perfConv = 1
#     remBack = 1
#
#     t = np.linspace(0, window_ns, numPoints)
#     fData = A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)
#     fData = np.array([fData])
#     fData1 = np.array([A1 * np.exp(-t / tau1)])
#     fData2 = np.array([A2 * np.exp(-t / tau2)])
#     Airf = np.max(fData)
#
#     delay_pts = delay_ns / (window_ns / numPoints)
#
#     IRF = np.exp(-t / tauIRF_ns) / (1 + np.exp(-delay_ns * (t - delay_ns)))
#
#     IRF = IRF - min(IRF.flatten())
#     IRF = Airf * (IRF / np.max(IRF.flatten()))
#     IRF = np.array([IRF])
#
#     # fData = [np.zeros(1, delay_pts) fData(1, 1:end - delay_pts)]
#     # fData1 = [np.zeros(1, delay_pts) fData1(1, 1:end - delay_pts)]
#     # fData2 = [np.zeros(1, delay_pts) fData2(1, 1:end - delay_pts)]
#
#     TCSPCdata = fData
#     if perfConv:
#         TCSPCdata = convol(IRF, fData)
#         TCSPCdata = TCSPCdata - min(TCSPCdata.flatten())
#         TCSPCdata = Airf * (TCSPCdata / max(TCSPCdata.flatten()))
#
#     if addNoise:
#         # backG = Airf * 0.05 * np.ones(1, np.size(TCSPCdata, 1))
#         backG = Airf * 0.05 * np.ones([1, np.size(TCSPCdata, 1)])
#
#         TCSPCdata = np.random.poisson(TCSPCdata + backG)
#         fData1 = np.random.poisson(fData1 + backG)
#         fData2 = np.random.poisson(fData2 + backG)
#         IRF = np.random.poisson(IRF + backG)
#         if remBack:
#             backLevel = np.mean(TCSPCdata[0, 1: int(delay_pts) - 1])
#             IRF = IRF - backLevel
#             TCSPCdata = TCSPCdata - backLevel
#
#     return TCSPCdata, IRF

def gendata(window_ns, numpoints, tauirf_ns, ampl1, tau1, ampl2, tau2, delay, addnoise, perfConv = 1, remBack = 1):

    times = np.linspace(0, window_ns, numpoints)
    fdata = ampl1 * np.exp(-times / tau1) + ampl2 * np.exp(-times / tau2)
    fdata = np.array([fdata])
    fdata1 = np.array([ampl1 * np.exp(-times / tau1)])
    fdata2 = np.array([ampl2 * np.exp(-times / tau2)])
    airf = np.max(fdata)

    delay_pts = delay_ns / (window_ns / numpoints)

    irf = np.exp(-times / tauirf_ns) / (1 + np.exp(-delay_ns * (times - delay_ns)))

    irf = irf - min(irf.flatten())
    irf = airf * (irf / np.max(irf.flatten()))
    irf = np.array([irf])

    # fData = [np.zeros(1, delay_pts) fData(1, 1:end - delay_pts)]
    # fData1 = [np.zeros(1, delay_pts) fData1(1, 1:end - delay_pts)]
    # fData2 = [np.zeros(1, delay_pts) fData2(1, 1:end - delay_pts)]

    tcspcdata = fdata
    if perfConv:
        tcspcdata = convol(irf, fdata)
        tcspcdata = tcspcdata - min(tcspcdata.flatten())
        tcspcdata = airf * (tcspcdata / max(tcspcdata.flatten()))

    if addnoise:
        # backG = Airf * 0.05 * np.ones(1, np.size(TCSPCdata, 1))
        backg = airf * 0.05 * np.ones([1, np.size(tcspcdata, 1)])

        tcspcdata = np.random.poisson(tcspcdata + backg)
        fdata1 = np.random.poisson(fdata1 + backg)
        fdata2 = np.random.poisson(fdata2 + backg)
        irf = np.random.poisson(irf + backg)
        if remBack:
            backlevel = np.mean(tcspcdata[0, 1: int(delay_pts) - 1])
            irf = irf - backlevel
            tcspcdata = tcspcdata - backlevel

    return tcspcdata, irf


window_ns = 200
numPoints = 10000
chnlWidth_ns = window_ns/numPoints
tauIRF_ns = 1
A1 = 10000
tau1 = 8
A2 = 6000
tau2 = 30
delay_ns = 20
addNoise = True

tau1list = np.array([])
tau2list = np.array([])
counter = 0
for count in range(100):
    print(count)
    TCSPCdata, IRF = gendata(window_ns, numPoints, tauIRF_ns, A1, tau1, A2, tau2, delay_ns, addNoise)
    # TCSPCdata = datfile[count]
    # IRF = irffile[count]
    c, offset, A, tau, dc, dtau, irs, zz, t, chi = fluofit(IRF, TCSPCdata, window_ns, chnlWidth_ns,
                                                           tau=np.array([tau1, tau2]), ploton=False)
    # distfluofit(IRF, TCSPCdata, window_ns, chnlWidth_ns)
    # print(tau)
    if np.size(tau, 0) == 2:
        tau1list = np.append(tau1list, tau[0])
        tau2list = np.append(tau2list, tau[1])
    else:
        counter += 1

# print("Amplitudes:", A)

print(counter)
tau1mean = np.mean(tau1list)
tau2mean = np.mean(tau2list)
tau1std = np.std(tau1list)
tau2std = np.std(tau2list)
tau1max = np.std(tau1list)
tau1min = np.std(tau1list)
tau2max = np.std(tau2list)
tau2min = np.std(tau2list)

print("Decay times:", tau1mean, tau2mean)
print("Decay times error:", dtau)
plt.hist(tau1list, bins='auto')
plt.hist(tau2list, bins='auto')
plt.show()
# print("Decay times max/min:", tau1std, tau2std)

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(tau1list)
# ax1.axhline(tau1mean)
# ax1.axhline(tau1mean+tau1std)
# ax1.axhline(tau1mean-tau1std)
# ax2.plot(tau2list, color='C1')
# ax2.axhline(tau2mean, color='C1')
# ax2.axhline(tau2mean+tau2std, color='C1')
# ax2.axhline(tau2mean-tau2std, color='C1')
# # fig.savefig('/home/bertus/Documents/Honneurs/Projek/python.png', dpi=600)
# plt.show()

# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')
# for stat in top_stats[:10]:
#     print(stat)



    






