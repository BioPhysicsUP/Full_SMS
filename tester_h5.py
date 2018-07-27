import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from matplotlib import rc
from tcspcfit import *
import h5py
rc('text', usetex=True)

f = h5py.File('IRF 680nm.h5', 'r')
data = f['Particle 1/Micro Times (s)'][:]

# data = np.loadtxt('gendata5_20.txt')
# # simdata = np.loadtxt('/home/bertus/temp/convd.txt')

# a1 = 0.624
# a2 = 0.423

# t = data[:, 0]
# measured = data[:, 1]
irf = data
irflength = np.size(irf)
# scale = np.max(measured)

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
measured = four_exp(model_in, tau1, tau2, tau3, tau4, np.max(irf), a1, a2, a3, a4, 0)
# measured = one_exp(model_in, tau1, np.max(irf), a1, 0)
measured = np.abs(measured)
measured = np.random.poisson(measured)
# plt.plot(measured)
# plt.plot(irf)
# plt.show()

fit = fluofit(irf, measured, t, window, channelwidth, tau=[9, 0.5, 1.7, 0.04], startpoint=310, endpoint=3500, ploton=True)
print(fit)




