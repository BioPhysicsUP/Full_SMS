from matplotlib import rc
from tcspcfit import *

rc('text', usetex=True)

fitlist = np.array([[0, 0, 0, 0]])
data = np.loadtxt('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/Decaytrace0')
irfdata = np.loadtxt('/home/bertus/Documents/Honneurs/Projek/Farooq_Data/IRF_data.txt')

t = irfdata[:, 0]
measured = data[:, 1]
irf = irfdata[:, 1]
irflength = np.size(irf)

window = np.size(irf)
print(window)
channelwidth = max(t) / window
print(channelwidth)

root = 200

# [init, min, max, fix]
tau = [[5, 0.01, 10, 1],
      [0.4, 0.01, 10, 1]]

shift = [0, -100, 100, 0]  # num of channels

# [init (amp1 %), fix]
amp = [30, 0]

fit = TwoExp(irf, measured, t, channelwidth, tau=tau, amp=[30, 0], shift=shift, startpoint=400, ploton=True)



