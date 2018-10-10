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

fit = TwoExp(irf, measured, t, channelwidth, tau=[5, 0.4], startpoint=300, endpoint=3000, ploton=True)
#
# print(t)
# model = 15.39 * np.exp(-t / 3.02) + 56.04 * np.exp(-t / 0.387)
#
# irs = colorshift(irf, -53.5)
#
# measured = measured[:3280]
# model = convolve(irs, model)[:3280]
# model = model * measured.sum() / model.sum()
#
# data_nonzero = measured[measured > 0]
# model_nonzero = model[measured > 0]
# chisum_next = ((measured - model) ** 2).sum() / 2
# print(chisum_next)
#
# model = fit.fitfunc(t, 3.02, 0.387, 15.39, 56.04, -53.5)
#
# print(measured)
#
# data_nonzero = measured[measured > 0]
# model_nonzero = model[measured > 0]
# chisum_next = ((data_nonzero - model_nonzero) ** 2).sum() / 2
# print(chisum_next)



