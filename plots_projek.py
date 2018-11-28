import matplotlib as mpl
from matplotlib import rc
from tcspcfit import *
import h5py

rc('text', usetex=True)
mpl.rcParams.update({'font.size': 16})

irf_file = h5py.File('IRF_680nm.h5', 'r')
meas_file = h5py.File('LHCII_630nW.h5', 'r')
# meas_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/Johanette/40.h5', 'r')
irf_data = irf_file['Particle 1/Micro Times (s)'][:]
meas_data = meas_file['Particle 10/Micro Times (s)'][:]
# meas_data += 10
irf_data = irf_data - np.sort(meas_data)[1]
meas_data = meas_data - np.sort(meas_data)[1]

# Determine the channelwidth. I think this should rather be saved as a data attribute.
differences = np.diff(np.sort(irf_data))
channelwidth = np.unique(differences)[1]
assert channelwidth == np.unique(np.diff(np.sort(meas_data)))[1]

# Make histograms out of the microtimes
tmin = min(irf_data.min(), meas_data.min())
tmax = max(irf_data.max(), meas_data.max())
window = tmax - tmin
numpoints = int(window // channelwidth)
t = np.linspace(0, window, numpoints)
print(t)
irf, t = np.histogram(irf_data, bins=t)
measured, t = np.histogram(meas_data, bins=t)
irf = irf[:-20]
measured = measured[:-20]
t = t[:-20]
# plt.plot(measured)
# plt.plot(irf)
# plt.show()

# How to specify initial values
# [init, min, max, fix]
tau = [[3, 0.01, 10, 0],
       [0.2, 0.01, 10, 0]]

shift = [0, -100, 2000, 0]  # Units are number of channels

# Only need to specify one amplitude as they sum to 100%
# [init (amp1 %), fix]
amp = [30, 0]

# Object orientated interface: Each fit is an object
fit = TwoExp(irf, measured, t, channelwidth, tau=tau, amp=amp, shift=shift, startpoint=0, ploton=False)
# fit1 = OneExp(irf, measured, t, channelwidth, tau=3, shift=shift, startpoint=800, ploton=False)
# Initial guess not necessarily needed:
# fit2 = OneExp(irf, measured, t, channelwidth, ploton=False)

tau = fit.tau
amp = fit.amp
shift = fit.shift
print(shift)
start = int(10 / channelwidth)
end = fit.endpoint
convd = fit.convd[start:end]
irf = colorshift(fit.irf, shift / channelwidth)[start:end]
irf = irf * np.max(measured) / np.max(irf)
t = fit.t[start:end]
measured = fit.measured[start:end]

plt.plot(t, measured, color='xkcd:pale red', linewidth=1)
plt.plot(t, irf, color='xkcd:gray')
plt.plot(t, convd, color='xkcd:dark red', linewidth=2.5)
plt.yscale('log')
plt.xlabel('Time (ns)')
plt.ylabel('Intensity (counts)')
plt.ylim([0.5, 700])
plt.text(17, 200, r'$\tau_1 = {:3.2f}\ ns, \ \tau_2 = {:3.2f}\ ns$'.format(tau[0], tau[1]))
plt.text(17, 100, r'$A_1 = {:3.2f}\ \%, \ A_2 = {:3.2f}\ \%$'.format(amp[0], amp[1]))
plt.tight_layout()
plt.show()


