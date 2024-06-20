from scipy.io import loadmat
import numpy as np
import h5py
from datetime import datetime
import os

filedir = r'schlau-cohen'

export_path = os.path.join(filedir, r'export\wt vio 1.h5')
read_path = os.path.join(filedir, r'wt vio 1')
partnum_write = 0

with h5py.File(export_path, 'w') as h5_f:
    h5_f.attrs.create(name='Version', data='100')

# for part in range(46):
# for part in range(89):
# for part in range(110):
for part in range(102):
    try:
        mat = loadmat(os.path.join(read_path, f'default_{part:03d}.mat'))
    except FileNotFoundError:  # Data is already filtered so some numbers don't exist
        continue
    # print(mat.keys())
    # print(mat['t_units'])

    date = datetime.now().strftime("%A, %B %#d, %Y %#I:%M %p")

    abs_times = mat['tt1'].flatten() * 1e6  # convert ms to ns
    micro_times = mat['kin1'].flatten() / 1e2  # convert 10 ps to ns
    # print(np.unique(np.diff(np.sort(micro_times))))

    with h5py.File(export_path, 'r+') as h5_f:
        # h5_f.attrs.create(name='# Particles', data=partnum_write)
        # h5_f.attrs.create(name='Version', data='100')

        partnum_write += 1
        part_group = h5_f.create_group(f'Particle {partnum_write}')
        part_group.attrs.create('Date', date)
        part_group.attrs.create('Description', 'LHCSR3 Data from Schlau-Cohen Lab')
        part_group.attrs.create('Intensity?', 1)
        part_group.attrs.create('RS Coord. (um)', [0, 0])
        part_group.attrs.create('Spectra?', 0)
        part_group.attrs.create('User', 'Bertus')

        data_abs = part_group.create_dataset('Absolute Times (ns)',
                                  dtype=np.uint64,
                                  data=abs_times)
        data_abs.attrs['bh Card'] = 'None'
        data_micro = part_group.create_dataset('Micro Times (ns)',
                                  dtype=np.float64,
                                  data=micro_times)
        data_micro.attrs['bh Card'] = 'None'
        data_abs = part_group.create_dataset('Absolute Times 2 (ns)',
                                             dtype=np.uint64,
                                             data=[])
        data_abs.attrs['bh Card'] = 'None'
        data_micro = part_group.create_dataset('Micro Times 2 (ns)',
                                               dtype=np.float64,
                                               data=[])
        data_micro.attrs['bh Card'] = 'None'
        data_int = part_group.create_dataset('Intensity Trace (cps)',
                                             dtype=np.uint64,
                                             data=[0])
print('particles written = ', partnum_write)
with h5py.File(export_path, 'r+') as h5_f:
    h5_f.attrs.create(name='# Particles', data=partnum_write)
