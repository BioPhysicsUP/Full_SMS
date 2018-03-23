import numpy as np
from matplotlib import pyplot as plt
from tcspcfit import convol
from scipy.fftpack import fft, ifft

x1 = np.array([[0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
x2 = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])  # TODO: should be [[a, b, ..., c, d]]
irf = np.array([0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0])  # same here

y1 = convol(irf, x1)
y2 = convol(irf, x2)
plt.plot(irf)
plt.plot(x2)
plt.plot(y2)
plt.plot(ifft(fft(irf)*fft(x2)))  # Testing to see whether the plots overlap
plt.plot(x1[1])
plt.plot(y1[1])
plt.plot(ifft(fft(irf)*fft(x1[1])))  # Works for multi-dimensional x
plt.show()


