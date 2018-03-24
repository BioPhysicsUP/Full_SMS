import numpy as np
from matplotlib import pyplot as plt
from tcspcfit import convol
from scipy.fftpack import fft, ifft

x1 = np.array([[0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
x2 = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
irf = np.array([[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]])

y1 = convol(irf, x1)
y2 = convol(irf, x2)
plt.plot(irf)
plt.plot(x2.transpose())
plt.plot(y2.transpose())
plt.plot(ifft(fft(irf.transpose())*fft(x2.transpose())))  # Testing to see whether the plots overlap
plt.plot(x1[2])
plt.plot(y1[2])
plt.plot(ifft(fft(irf)*fft(x1[2])))  # Works for multi-dimensional x
plt.show()


