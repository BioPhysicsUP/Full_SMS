import numpy as np

import h5py
from h5py._hl import group as h5group

import fnmatch

import matplotlib.pyplot as plt

import os


# TODO: Impliment Bertus's HDF5 backend instad of this temp one.
class H5:

	def __init__(self, path=None):
		self.file = None
		self.num_particles = None
		self.particles = None
		if path is not None:
			self.path = path
			self.open_h5()

	def open_h5(self, path=None):
		if path is None:
			assert hasattr(self, 'path'), "No path provided to open *.h5"
		else:
			self.path = path
		self.file = h5py.File(self.path, 'r')
		# self.num_particles = len(fnmatch.filter(self.file.keys(), 'Particle *'))
		self.num_particles = self.file.attrs['# Particles']
		self.particles = {'Particle ' + str(num): Particle(self, num) for num in range(1, self.num_particles + 1)}

	def get_particle(self, num=None):
		assert num is not None and (1 <= num <= self.num_particles), "Particle number not given or out of range."
		return self.particles['Particle ' + str(num)]

	def __copy__(self):
		return self

	def particles(self):
		return [Particle(self, part_num) for part_num in range(1, self.num_particles+1)]


class Particle:

	def __init__(self, h5, num=None):
		assert type(h5) is H5 and num is not None, "Particle number not given to initialise Particle object."
		self.h5 = H5.__copy__(h5)
		self.name = 'Particle ' + str(num)
		self.group = self.h5.file[self.name]
		self.num_photons = self.h5.file[self.name + '/Absolute Times (ns)'].size
		self.meta = Meta(self.group)
		self.chg_pts = ChangePoints(self)

	def read_abs_times(self, start_ind=None, end_ind=None):
		if start_ind is None or end_ind is None:
			start_ind = 1
			end_ind = self.num_photons
		return self.h5.file[self.name + '/Absolute Times (ns)'][start_ind:end_ind]

	def read_micro_times(self, start_ind=None, end_ind=None):
		if start_ind is None or end_ind is None:
			start_ind = 1
			end_ind = self.num_photons
		return self.h5.file[self.name + '/Micro Times (s)'][start_ind:end_ind]

	def __copy__(self):
		return self

	def chg_pts(self, confidence=None):
		assert confidence is not None, "No confidence interval provided."
		self.chg_pts = ChangePoints(self)
		self.chg_pts.find(conf=99)
		pass


class TauData:
	def __init__(self):
		tau_data_path = os.getcwd() + os.path.sep + 'tau data'
		assert os.path.isdir(tau_data_path), "Tau data directory not found."
		tau_data_files = {'99_a': 'Ta-99.txt',
		                  '99_b': 'Tb-99.txt',
		                  '95_a': 'Ta-95.txt',
		                  '95_b': 'Tb-95.txt',
		                  '90_a': 'Ta-90.txt',
		                  '90_b': 'Tb-90.txt',
		                  '69_a': 'Ta-69.txt',
		                  '69_b': 'Tb-69.txt'}
		self.data = {}
		for tau_type, file_name in tau_data_files.items():
			assert os.path.isfile(tau_data_path+os.path.sep+file_name), 'Tau data file '+file_name+' does not exist.'
			self.data[tau_type] = np.loadtxt(tau_data_path+os.path.sep+file_name, usecols=1)


class Meta:
	def __init__(self, part_group=None):
		assert type(part_group) is h5group.Group, "No HDF5 Group object given."
		self.part_group = h5group.Group(part_group.id)
		# self.particle = Particle.__copy__(particle)
		self.date = str(self.part_group.attrs['Date'])[2:-1]
		self.descrip = str(self.part_group.attrs['Discription'])[2:-1]
		self.int_measured = bool(self.part_group.attrs['Intensity?'])
		self.rs_coord = self.part_group.attrs['RS Coord. (um)']
		self.spec_measured = bool(self.part_group.attrs['Spectra?'])
		self.user = str(self.part_group.attrs['User'])[2:-1]
		if self.user


class ChangePoints:

	def __init__(self, particle=None, confidence=None):
		# assert h5file is not None, "No HDF5 has been given"  # To ensure that a h5file is given
		assert type(particle) is Particle, "No Particle object given."
		self.particle = Particle.__copy__(particle)
		self.chg_pts = []
		if confidence is not None:
			self.find(conf=confidence)

	def weighted_likelihood_ratio(self, start_ind=None, end_ind=None):
		""" Calculates the Weighted & Standardised Likelihood ratio.

		Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
		from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)
		"""

		""" Testing code:
		with h5py.File('LHCII_630nW.h5', 'r') as f:
			time_data = f['Particle 1/Absolute Times (ns)'][0:1000]
			
		time_data = np.arange(2000, 3000)
		"""

		assert start_ind is not None and end_ind is not None and end_ind - start_ind <= 1000, \
			'Index\'s not given, or result in more than a segment of more than 1000 points.'

		time_data = self.particle.read_abs_time(start_ind, end_ind)
		n = len(time_data)

		assert n <= 1000, "Data segment for weighted likelihood ratio is more than 1000 points."

		ini_time = time_data[0]
		period = time_data[-1] - ini_time

		wlr = np.zeros(n, float)

		sig_e = np.pi ** 2 / 6 - sum(1 / j ** 2 for j in range(1, (n - 1) + 1))

		for k in range(2, (n - 2) + 1):  # Remember!!!! range(1, N) = [1, ... , N-1]
			# print(k)
			cap_v_k = (time_data[k] - ini_time) / period  # Just after eq. 4
			u_k = -sum(1 / j for j in range(k, (n - 1) + 1))  # Just after eq. 6
			u_n_k = -sum(1 / j for j in range(n - k, (n - 1) + 1))  # Just after eq. 6
			l0_minus_expec_l0 = -2 * k * np.log(cap_v_k) + 2 * k * u_k - 2 * (n - k) * np.log(1 - cap_v_k) + 2 * (
					n - k) * u_n_k  # Just after eq. 6
			v_k2 = sum(1 / j ** 2 for j in range(k, (n - 1) + 1))  # Just before eq. 7
			v_n_k2 = sum(1 / j ** 2 for j in range(n - k, (n - 1) + 1))  # Just before eq. 7
			sigma_k = np.sqrt(
				4 * (k ** 2) * v_k2 + 4 * ((n - k) ** 2) * v_n_k2 - 8 * k * (n - k) * sig_e)  # Just before eq. 7, and note errata
			w_k = (1 / 2) * np.log((4 * k * (n - k)) / n ** 2)  # Just after eq. 6

			wlr.itemset(k, l0_minus_expec_l0 / sigma_k + w_k)  # Eq. 6 and just after eq. 6

		# """ Testing code:
		# fig = plt.figure(dpi=300)
		# ax = fig.add_subplot(111)
		# ax.plot(wlr)
		# plt.show()
		# """

		max_ind = wlr.argmax()
		return max_ind, wlr.item(max_ind)

	def next_seg_ind(self, inds=None):
		# if inds is None:
		# if self.
		pass

	def find(self, conf=None):
		pass


# def find_change_points(self):


print('Start')
h5_file = H5('LHCII_630nW.h5')
particles = h5_file.particles
tau = TauData()
for part_name, part in particles.items():
	print(part_name+': '+part.meta.user)
pass
