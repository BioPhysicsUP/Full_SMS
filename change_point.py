import numpy as np

import h5py

import matplotlib.pyplot as plt

class ChangePoints:

	def __init__(self):
		# self.
		pass


def weighted_likelihood_ratio( time_data=np.array([]) ):
	""" Calculates the Weighted & Standardised Likelihood ratio.

	Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
	from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)
	"""

	""" Testing code:
	with h5py.File('LHCII_630nW.h5', 'r') as f:
		time_data = f['Particle 1/Absolute Times (ns)'][0:1000]
	"""
	n = len(time_data)

	assert n <= 1000, "Data segment for weighted likelihood ratio is more than 1000 points."

	ini_time = time_data[0]
	period = time_data[-1] - ini_time

	wlr = np.zeros(n, float)

	sig_e = np.pi**2/6 - sum(1/j**2 for j in range(1, (n - 1) + 1))

	for k in range(2, (n-1)+1):  # Remember!!!! range(1, N) = [1, ... , N-1]
		# print(k)
		cap_v_k = (time_data[k] - ini_time)/period  # Just after eq. 4
		u_k = -sum(1/j for j in range(k, (n-1)+1))  # Just after eq. 6
		u_n_k = -sum(1/j for j in range(n-k, (n-1)+1))  # Just after eq. 6
		l0_minus_expec_l0 = -2*k*np.log(cap_v_k) + 2*k*u_k - 2*(n-k)*np.log(1-cap_v_k) + 2*(n-k)*u_n_k  # Just after eq. 6
		v_k2 = sum(1/j**2 for j in range(k, (n-1)+1))  # Just before eq. 7
		v_n_k2 = sum(1/j**2 for j in range(n - k, (n-1)+1))  # Just before eq. 7
		sigma_k = np.sqrt(4*(k**2)*v_k2 + 4*((n-k)**2)*v_n_k2 - 8*k*(n-k)*sig_e)  # Just before eq. 7, and note errata
		w_k = (1/2)*np.log((4*k*(n-k))/n**2)   # Just after eq. 6

		wlr.itemset(k, l0_minus_expec_l0/sigma_k + w_k)  # Eq. 6 and just after eq. 6

	""" Testing code:
	fig = plt.figure(dpi=300)
	ax = fig.add_subplot(111)
	ax.plot(wlr)
	plt.show()
	"""
