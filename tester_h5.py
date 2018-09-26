import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from matplotlib import rc
from tcspcfit import *
import h5py
rc('text', usetex=True)

f = h5py.File('IRF 680nm.h5', 'r')
meas = h5py.File('LHCII.h5', 'r')
data = f['Particle 1/Micro Times (s)'][:]
meas_data = meas['Particle 1/Micro Times (s)'][:]

# data = np.loadtxt('gendata5_20.txt')
# # simdata = np.loadtxt('/home/bertus/temp/convd.txt')

# a1 = 0.624
# a2 = 0.423

# t = data[:, 0]
# measured = data[:, 1]
irf, t = np.histogram(data, bins=1000)
irflength = np.size(irf)

t = t[:-1]

measured, blabla = np.histogram(meas_data, bins=1000)
# scale = np.max(measured)
savedata = np.column_stack((t, measured, irf))
np.savetxt('savedata.txt', savedata)

window = np.size(irf)
channelwidth = max(t) / window

# root = 200
numPoints = window / channelwidth
# channelwidth = root/numPoints
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
# plt.plot(measured)
# plt.plot(irf)
# plt.plot(t)
# plt.show()
plt.yscale('log')
plt.plot(t, measured)
plt.plot(t, irf)
plt.show()

fit = fluofit(irf, measured, t, window, channelwidth, tau=[2.8, 0.3], startpoint=300, endpoint=1000, ploton=True)
# print(fit)




