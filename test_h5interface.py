import h5py
from smsh5 import *

meas_file = H5dataset('LHCII_630nW.h5')
meas_file.binints(1000)
