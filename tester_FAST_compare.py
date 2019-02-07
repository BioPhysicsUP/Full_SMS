from matplotlib import rc
from tcspcfit import *
import h5py

rc('text', usetex=True)

irf_file = h5py.File('IRF_680nm.h5', 'r')
meas_file = h5py.File('LHCII_630nW.h5', 'r')
# meas_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/Johanette/40.h5', 'r')
irf_data = irf_file['Particle 1/Micro Times (s)'][:]
meas_data = meas_file['Particle 7/Micro Times (s)'][:]
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
# irf = irf[:-20]
# measured = measured[:-20]
# t = t[:-20]

tau = [[3.333, 0.01, 10, 1],
       [0.015, 0.01, 10, 1]]
shift = [3.1018 / channelwidth, -100, 2000, 1]  # Units are number of channels

amp = [0.0064, 0.198, 1]

irf = irf - 0.6
measured = measured - 2.6

irf = colorshift(irf, shift[0])

decay = 0.0064 * np.exp(-t / 3.333) + 0.0949 * np.exp(-t / 0.015)

convd = convolve(irf, decay)

plt.plot(measured)
plt.plot(convd)
plt.yscale('log')
plt.show()



