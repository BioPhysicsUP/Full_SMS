import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from matplotlib import rc
from tcspcfit import fluofit, convol
rc('text', usetex=True)


data = np.loadtxt('gendata5_20.txt')
# simdata = np.loadtxt('/home/bertus/temp/convd.txt')

a1 = 0.624
a2 = 0.423

t = data[:, 0]
measured = data[:, 1]
irf = data[:, 2]
irflength = np.size(irf)
scale = np.max(measured)

window = np.size(irf)
channelwidth = max(t) / window
fit = fluofit(irf, measured, t, window, channelwidth, tau=[5, 20], startpoint=300, ploton=True)
print(fit)





