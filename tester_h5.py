import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from matplotlib import rc
from tcspcfit import *
import h5py
from smsh5 import *
rc('text', usetex=True)

irf_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/metings/Farooq_intensity/IRF (SLOW APD@680nm).h5', 'r')
meas_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/metings/Farooq_intensity/LHCII-PLL(slow APD)-410nW.h5', 'r')
irf_data = irf_file['Particle 1/Micro Times (s)'][:]
meas_data = meas_file['Particle 2/Micro Times (s)'][:]

differences = np.diff(np.sort(irf_data))
channelwidth = np.unique(differences)[1]
assert channelwidth == np.unique(np.diff(np.sort(meas_data)))[1]

tmin = min(irf_data.min(), meas_data.min())
tmax = max(irf_data.max(), meas_data.max())
window = tmax - tmin
numpoints = int(window // channelwidth)

t = np.linspace(0, window, numpoints)
irf_data -= irf_data.min()
meas_data -= meas_data.min()

irf, t = np.histogram(irf_data, bins=t)
measured, t = np.histogram(meas_data, bins=t)

irf = irf[:-20]  # This is due to some bug in the setup software putting a bunch of very long times in at the end
measured = measured[:-20]
t = t[:-20]

# fit = TwoExp(irf, measured, t, channelwidth, tau=[2, 0.3], ploton=True)
# print(fit)

particle1 = Particle(meas_file, 1)




