import numpy as np
from matplotlib import pyplot as plt
from tcspcfit import convol

x = np.array([0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0])
irf = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1])

y = convol(irf, x)

plt.plot(irf)
plt.plot(x)
plt.plot(y)
plt.show()