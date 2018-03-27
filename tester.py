import numpy as np
from tcspcfit import *
from matplotlib import pyplot as plt

window_ns = 200
numPoints = 200
chnlWidth_ns = window_ns/numPoints
tauIRF_ns = 1
A1 = 10000
tau1_ns = 8
A2 = 20000
tau2_ns = 30
delay_ns = 20
addNoise = True

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

# plt.plot(TCSPCdata)
# plt.show()
print(np.shape(IRF))

err, A, z = lsfit(np.array([[1000, 0.0004, 8, 30]]), IRF, TCSPCdata, 200)
print(A)
