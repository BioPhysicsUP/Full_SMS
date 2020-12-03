from src.tcspcfit import *
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 16})


def gendata(window_ns, numpoints, tauirf_ns, delay_ns, cshift, ampl1, tau1, ampl2, tau2, addnoise, remback=True):

    times = np.linspace(0, window_ns, numpoints)
    fdata = ampl1 * np.exp(-times / tau1) + ampl2 * np.exp(-times / tau2)
    # fdata = np.array(fdata)
    airf = np.max(fdata)

    delay_pts = delay_ns / (window_ns / numpoints)

    irf = np.exp(-times / tauirf_ns) / (1 + np.exp(-delay_ns * (times - delay_ns)))

    irf = irf - min(irf.flatten())
    irf = airf * (irf / np.max(irf.flatten()))
    irf = np.array(irf)
    irs = colorshift(irf, cshift)

    # fData = [np.zeros(1, delay_pts) fData(1, 1:end - delay_pts)]
    # fData1 = [np.zeros(1, delay_pts) fData1(1, 1:end - delay_pts)]
    # fData2 = [np.zeros(1, delay_pts) fData2(1, 1:end - delay_pts)]

    tcspcdata = fdata
    tcspcdata = convolve(irs, fdata)[:numpoints]
    tcspcdata = tcspcdata - min(tcspcdata.flatten())
    tcspcdata = airf * (tcspcdata / max(tcspcdata.flatten()))

    if addnoise:
        # backG = Airf * 0.05 * np.ones(1, np.size(TCSPCdata, 1))
        backg = airf * 0.05 * np.ones(np.size(tcspcdata))

        tcspcdata = np.random.poisson(tcspcdata + backg)
        irf = np.random.poisson(irf + backg)
    if remback:
        backlevel = np.mean(tcspcdata[1: int(delay_pts) - 1])
        irf = irf - backlevel
        tcspcdata = tcspcdata - backlevel

    return tcspcdata, irf


window = 20
numPoints = 1000
chnlWidth_ns = window / numPoints
print(chnlWidth_ns)
tauIRF_ns = 0.2
A1 = 5000
tau1_ns = 2
A2 = 5000
tau2_ns = 10
delay_ns = 7
addNoise = True

TCSPCdata, IRF = gendata(window, numPoints, tauIRF_ns, delay_ns, 50, A1, tau1_ns, A2, tau2_ns, addNoise)
t = np.linspace(0, window, numPoints)

# plt.plot(IRF)
# plt.plot(TCSPCdata)
# plt.show()

# fit = TwoExp(IRF, TCSPCdata, t, chnlWidth_ns, [tau1_ns, tau2_ns], ploton=True)
# print(fit.shift)

irfdata = np.loadtxt('/home/bertus/Documents/Honneurs/Projek/Random/Farooq_Data/IRF_data.txt')

t = irfdata[:, 0]
irf = irfdata[:, 1]
irflength = np.size(irf)

# measured = measured * (np.max(irf) / np.max(measured))
# plt.plot(measured)
# plt.plot(irf)
# plt.show()
# measured, blabla = np.histogram(meas_data, bins=1000)
# scale = np.max(measured)
# savedata = np.column_stack((t[:-1], measured, irf))
# np.savetxt('savedata.txt', savedata)

window = np.size(irf)
channelwidth = max(t) / window

# root = 200

# irf = irf - 31
# irf = irf * (np.max(measured) / np.max(irf))
# irf = colorshift(irf, -50, 0, 0).flatten()

model = 0.3 * np.exp(-t / 3) + 0.7 * np.exp(-t / 0.05)
modelfit = 0.431 * np.exp(-t / 2.96) + 0.569 * np.exp(-t / 0.023)

# plt.plot(irf)
# irf = irf - 20
# irf = irf.clip(0)
# plt.plot(irf)
# plt.plot(measured)
# measured = measured - 0.535
# measured = measured.clip(0)
# plt.plot(measured)
# plt.yscale('log')
# plt.show()
# # measured = measured / np.sum(measured)
# irf = irf * (np.sum(measured) / np.sum(irf))
#
# irf = colorshift(irf, -40, np.size(irf), t).flatten()

# irf = self.irf
# irf = irf * (np.sum(self.measured) / np.sum(irf))
# irf = colorshift(irf, shift)
# convd = convolve(irf, model)
# self.scalefactor = np.sum(self.measured) / np.sum(convd)
# convd = convd * self.scalefactor
# convd = convd[self.startpoint:self.endpoint]

irs = colorshift(irf, 0)
# irs = irf
convd = convolve(irs, model)
convd_fit = convolve(irs, modelfit)
convd = convd[:np.size(irs)]
convd_fit = convd_fit[:np.size(irs)]

convd = convd / np.sum(convd) * np.sum(irs)
convd_fit = convd_fit / np.sum(convd_fit) * np.sum(irs)
convd_noise = np.random.poisson(convd / 10)

convd_fit *= (np.sum(convd) / np.sum(convd_fit))

np.savetxt('fast/gen_data.fst', np.column_stack((t, irf, convd_noise)))

# plt.plot(irf)
# plt.plot(model)
# plt.plot(measured)
# plt.figure()
# plt.plot(convd)
# plt.plot(convd_fit)
# plt.yscale('log')
# plt.xlim((200, 3000))
# plt.yscale('log')
# plt.ylim((0.01, 100))
# plt.plot(t)
# plt.figure()
# plt.plot((convd_fit - convd)[300:], '.', markersize=2)
# plt.plot(convd_fit - convd_noise, '.', markersize=2)
# residuals = np.sum((convd[0:3000] - measured[0:3000]) ** 2)
# plt.text(2000, -10, str(residuals))
# plt.show()

# fit = fluofit(irf, measured, t, window, channelwidth, tau=[2.52, 0.336], startpoint=300, endpoint=3000, ploton=True)

fit = TwoExp(irf, convd, t, channelwidth, tau=[[3, 0.1, 6, 0], [0.036, 0.01, 1, 0]], startpoint=300, endpoint=3000, ploton=True)
fit2 = TwoExp(irf, convd_noise, t, channelwidth, tau=[[4, 1, 15, 0], [0.1, 0.01, 4, 0]], startpoint=400, endpoint=2400, ploton=True)
# fit = OneExp(irf, convd_noise, t, channelwidth, tau=[[4, 1, 15, 0]], startpoint=400, endpoint=2400, ploton=True)
# sumamp = sum(fit.amp * fit.tau)
# print(sum(fit.amp * fit.tau ** 2)/ sumamp)
print(fit.shift)
print(fit.bg)
print(fit.irfbg)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(fit2.residuals, '.', markersize=2)
ax1.text(250, 4, r'$\chi ^2 = $ {:3.4f}'.format(fit2.chisq))
ax2.plot(fit.residuals, '.', markersize=2)
ax2.text(250, 5, r'$\chi ^2 = $ {:3.4f}'.format(fit.chisq))

# modeled = fit.makeconvd(0, model)
# plt.figure()
# plt.plot(modeled - convd[300:3000], '.', markersize=2)
# # plt.plot(fit.irf - irf, '.', markersize=2)
# plt.show()

