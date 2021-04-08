from matplotlib import rc
from src.tcspcfit import *

rc('text', usetex=True)

data = np.loadtxt('Decaytrace1')
irfdata = np.loadtxt('IRF_data.txt')

t = irfdata[:, 0]
measured = data[:, 1]
irf = irfdata[:, 1]
irflength = np.size(irf)

window = np.size(irf)
channelwidth = max(t) / window

# How to specify initial values
# [init, min, max, fix]
tau = [[3, 0.01, 10, 0],
       [0.2, 0.01, 10, 0]]

shift = [0, -100, 2000, 0]  # Units are number of channels

# Only need to specify one amplitude as they sum to 100%
# [init (amp1 %), fix]
amp = [30, 0]

# Object orientated interface: Each fit is an object
fit = TwoExp(irf, measured, t, channelwidth, tau=tau, amp=amp, shift=shift, startpoint=300, ploton=True)
fit1 = OneExp(irf, measured, t, channelwidth, tau=3, shift=shift, startpoint=300, ploton=True)
# Initial guess not necessarily needed:
fit2 = OneExp(irf, measured, t, channelwidth, ploton=True)

tau = fit.tau
amp = fit.amp
shift = fit.shift

print('Lifetimes: {:5.3f} ns, {:5.3f} ns'.format(tau[0], tau[1]))
print('Amplitudes: {:4.2f} %, {:4.2f} %'.format(amp[0], amp[1]))
print('IRF Shift: {:4.2f} ns'.format(shift * channelwidth))
