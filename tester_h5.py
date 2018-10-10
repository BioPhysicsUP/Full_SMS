import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from matplotlib import rc
from tcspcfit import *
import h5py
rc('text', usetex=True)

irf_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/metings/Farooq_intensity/IRF (SLOW APD@680nm).h5', 'r')
meas_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/metings/Farooq_intensity/LHCII-PLL(slow APD)-608nW.h5', 'r')
irf_data = irf_file['Particle 1/Micro Times (s)'][:]
meas_data = meas_file['Particle 1/Micro Times (s)'][:]

datahist, edges = np.histogram(meas_data, bins=2522)
print((edges[-1] - edges[0]) / 0.01220703)
datahist, edges = datahist[:-20], edges[:-20]
# plt.bar(edges[:-1], datahist, width=edges[1]-edges[0])
# plt.plot(edges[:-1], datahist, color='C1')
# plt.xlim(28, 31)
# plt.plot(np.sort(irf_data))
# plt.show()

differences = np.diff(np.sort(irf_data))
channelwidth = np.unique(differences)[1]
print(channelwidth)
assert channelwidth == np.unique(np.diff(np.sort(meas_data)))[1]

tmin = min(irf_data.min(), meas_data.min())
tmax = max(irf_data.max(), meas_data.max())
window = tmax - tmin
numpoints = int(window // channelwidth)
print(numpoints)

t = np.linspace(0, window, numpoints)
irf_data -= irf_data.min()
meas_data -= meas_data.min()

irf, t = np.histogram(irf_data, bins=t)
measured, t = np.histogram(meas_data, bins=t)

irf = irf[:-20]
measured = measured[:-20]

# plt.bar(t[:-1], irf*np.sum(measured)/np.sum(irf), width=channelwidth)
# plt.bar(t[:-1], measured, width=channelwidth)
# plt.plot(irf * measured.max() / irf.max())
# plt.plot(measured)
# plt.show()
irflength = np.size(irf)

# model1 = 1 * np.exp(-t / 0.3)
# model2 = 9 * np.exp(-t / 3)
# plt.plot(model1)
# plt.plot(model2)
# plt.plot(t)
# print(t)
model = 1 * np.exp(-t / 3) + 2 * np.exp(-t / 0.3)

irs = colorshift(irf, -25)

# convd1 = convolve(irs, model1)
# convd2 = convolve(irs, model2)
convd = convolve(irs, model)
# plt.plot(convd1)
# plt.plot(convd2)
# convd = convd[:1800]
# measured = measured[:1800]
convd = convd / np.sum(convd) * np.sum(measured)

# irf = irf * (np.max(measured) / np.max(irf))
# plt.plot(measured)
# plt.plot(convd)
# plt.yscale('log')
# plt.plot(irf)
# plt.plot(t)
# plt.show()
# plt.yscale('log')
# plt.plot(t, measured)
# plt.plot(t, irf)
# plt.show()

fit = TwoExp(irf, measured, t, channelwidth, tau=[2, 0.3], startpoint=650, endpoint=1800, ploton=True)
# print(fit)




