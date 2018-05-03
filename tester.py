import numpy as np
from tcspcfit import fluofit, makerow, convol
from matplotlib import pyplot as plt
import os


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


window = 200
numPoints = 10000
chnlWidth_ns = window / numPoints
tauIRF_ns = 1
A1 = 10000
tau1_ns = 8
A2 = 6000
tau2_ns = 30
delay_ns = 20
addNoise = True

TCSPCdata, IRF = gendata(window, numPoints, tauIRF_ns, A1, tau1_ns, A2, tau2_ns, delay_ns, addNoise)
    # TCSPCdata = datfile[count]
    # IRF = irffile[count]
c, offset, A, tau, dc, dtau, irs, zz, t, chi = fluofit(IRF, TCSPCdata, window, chnlWidth_ns,
                                                           np.array([tau1_ns, tau2_ns]), ploton=False)
print(tau)


tau1array = np.zeros((1, 20))
tau2array = np.zeros((1, 20))
tau1err_arr = np.zeros((1, 20))
tau2err_arr = np.zeros((1, 20))
for tau1 in range(1, 21):
    tau1mean = np.array([[]])
    tau2mean = np.array([[]])
    tau1err = np.array([[]])
    tau2err = np.array([[]])
    for tau2 in range(1, 21):
        tau1list = np.array([])
        tau2list = np.array([])
        if tau1 == tau2:
            tau1mean = np.append(tau1mean, tau1)
            tau2mean = np.append(tau2mean, tau2)
            tau1err = np.append(tau1err, 0)
            tau2err = np.append(tau2err, 0)
        else:
            for count in range(10):
                # print(tau1, tau2, count)
                perc = (tau1 - 1) * 5 + (tau2 - 1) * 0.25 + (count - 1) * 0.025
                print(str(perc) + ' %')
                TCSPCdata, IRF = gendata(window, numPoints, tauIRF_ns, A1, tau1, A2, tau2, delay_ns, addNoise)
                # TCSPCdata = datfile[count]
                # IRF = irffile[count]
                c, offset, A, tau, dc, dtau, irs, zz, t, chi = fluofit(IRF, TCSPCdata, window, chnlWidth_ns,
                                                                       np.array([[tau1, tau2]]), ploton=False)
                tau1list = np.append(tau1list, tau[0])
                tau2list = np.append(tau2list, tau[1])
            tau1mean = np.append(tau1mean, np.mean(tau1list))
            tau2mean = np.append(tau2mean, np.mean(tau2list))
            tau1err = np.append(tau1err, np.std(tau1list))
            tau2err = np.append(tau2err, np.std(tau2list))
    tau1mean = makerow(tau1mean)
    tau2mean = makerow(tau2mean)
    tau1err = makerow(tau1err)
    tau2err = makerow(tau2err)
    tau1array = np.append(tau1array, tau1mean, axis=0)
    tau2array = np.append(tau2array, tau2mean, axis=0)
    tau1err_arr = np.append(tau1err_arr, tau1err, axis=0)
    tau2err_arr = np.append(tau2err_arr, tau2err, axis=0)

# Remove zero rows
tau1array = tau1array[1:, :]
tau1err_arr = tau1err_arr[1:, :]
tau2array = tau2array[1:, :]
tau2err_arr = tau2err_arr[1:, :]

#  Create a new file each time we run
tau1filename = 'tau1_#'
outputVersion = 1
while os.path.isfile(tau1filename.replace("#", str(outputVersion))):
    outputVersion += 1
tau1filename = tau1filename.replace("#", str(outputVersion))

tau2filename = 'tau2_#'
outputVersion = 1
while os.path.isfile(tau2filename.replace("#", str(outputVersion))):
    outputVersion += 1
tau2filename = tau2filename.replace("#", str(outputVersion))

err1filename = 'mean1_#'
outputVersion = 1
while os.path.isfile(err1filename.replace("#", str(outputVersion))):
    outputVersion += 1
err1filename = err1filename.replace("#", str(outputVersion))

err2filename = 'mean2_#'
outputVersion = 1
while os.path.isfile(err2filename.replace("#", str(outputVersion))):
    outputVersion += 1
err2filename = err2filename.replace("#", str(outputVersion))

np.savetxt(tau1filename, tau1array, fmt='%6.2f', delimiter=',')
np.savetxt(tau2filename, tau2array, fmt='%6.2f', delimiter=',')
np.savetxt(err1filename, tau1err_arr, fmt='%6.2f', delimiter=',')
np.savetxt(err2filename, tau2err_arr, fmt='%6.2f', delimiter=',')

# print("Amplitudes:", A)
print("Decay times:", tau)
# print("Decay times error:", dtau)

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(tau1list)
# ax1.axhline(np.mean(tau1list))
# ax2.plot(tau2list, color='C1')
# ax2.axhline(np.mean(tau2list), color='C1')
# # fig.savefig('/home/bertus/Documents/Honneurs/Projek/python.png', dpi=600)
# plt.show()
