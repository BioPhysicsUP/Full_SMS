import matplotlib.pyplot as plt

import smsh5

import os

import dbg


class TauData:
    def __init__(self, confidence=None):
        tau_data_path = os.getcwd() + os.path.sep + 'tau data'
        assert os.path.isdir(tau_data_path), "TauData:\tTau data directory not found."
        tau_data_files = {'99_a': 'Ta-99.txt',
                          '99_b': 'Tb-99.txt',
                          '95_a': 'Ta-95.txt',
                          '95_b': 'Tb-95.txt',
                          '90_a': 'Ta-90.txt',
                          '90_b': 'Tb-90.txt',
                          '69_a': 'Ta-69.txt',
                          '69_b': 'Tb-69.txt'}

        assert confidence in [0.99, 0.95, 0.90, 0.69], "TauData:\tInvalid confidence provided. Can not provide confidence key."
        if confidence == 0.99:
            conf_keys = ['99_a', '99_b']
        elif confidence == 0.95:
            conf_keys = ['95_a', '95_b']
        elif confidence == 0.90:
            conf_keys = ['90_a', '90_b']
        else:
            conf_keys = ['69_a', '69_b']

        for tau_type in conf_keys:
            file_name = tau_data_files[tau_type]
            full_path = tau_data_path + os.path.sep + file_name
            assert os.path.isfile(
                full_path), 'TauData:\tTau data file ' + file_name + ' does not exist. Look in' + full_path + '.'
            data = np.loadtxt(full_path, usecols=1)
            if tau_type[-1] is 'a':
                self._a = data
            else:
                self._b = data

    def get_tau_a(self, num_data_points=None):
        assert num_data_points is not None, "TauData:\tNumber of data points not given."
        return self._a[num_data_points]

    def get_tau_b(self, num_data_points=None):
        assert num_data_points is not None, "TauData:\tNumber of data points not given."
        return self._b[num_data_points]


class ChangePoints:

    def __init__(self, particle=None, confidence=None):
        # assert h5file is not None, "No HDF5 has been given"  # To ensure that a h5file is given
        assert type(particle) is smsh5.Particle, "ChangePoints:\tNo Particle object given."
        # assert confidence is not None, "ChangePoints:\tNo confidence parameter given."
        self.particle = smsh5.Particle.__copy__(particle)
        self.cpts = np.array([])
        self.conf_regions = np.array(tuple())
        self._finding = False
        self.found_cpts = False
        self.confidence = confidence
        if confidence is not None:
            self.tau = TauData(self.confidence)

    def __weighted_likelihood_ratio__(self, seg_inds=None):
        """ Calculates the Weighted & Standardised Likelihood ratio.

        Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
        from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)
        """

        """ Testing code:
        with h5py.File('LHCII_630nW.h5', 'r') as f:
            time_data = f['Particle 1/Absolute Times (ns)'][0:1000]

        time_data = np.arange(2000, 3000)
        """

        assert type(seg_inds) is tuple, 'ChangePoints:\tSegment index\'s not given.'
        start_ind, end_ind = seg_inds
        n = end_ind - start_ind
        assert n <= 1000, "ChangePoints:\tIndex\'s given result in more than a segment of more than 1000 points."
        time_data = self.particle.abstimes[start_ind:end_ind]

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

        if wlr[max_ind] >= self.tau.get_tau_a(n):
            self.cpts = np.append(self.cpts, max_ind)
            region_all = np.where(wlr >= self.tau.get_tau_b(n))[0]
            region = region_all[0], region_all[-1]
            self.conf_regions = np.append(self.conf_regions, region)
            cpt_found = True
            cpt_return = max_ind
        else:
            cpt_found = False
            cpt_return = None
        return cpt_found, cpt_return

    def __next_seg_ind__(self, prev_seg_inds=None):
        n_total = len(self.particle.abstimes)
        if prev_seg_inds is None:
            # Data sets need to be larger than 200 photons
            assert n_total >= 200, 'ChangePoints:\tData set needs to be at least 200 photons for change point detection.'
            if n_total > 1000:
                next_start_ind, next_end_ind = 1, 1001
            else:
                next_start_ind, next_end_ind = 1, n_total
        else:
            prev_start_ind, prev_end_ind = prev_seg_inds
            if len(self.cpts) == 0:
                if n_total >= prev_end_ind + 800:
                    next_start_ind, next_end_ind = prev_end_ind - 200, prev_end_ind + 800
                elif n_total - prev_end_ind < 50:  # Next segment needs to be at least 200 photons large.
                    next_start_ind, next_end_ind = self.cpts[-1] + 1, n_total - prev_end_ind
                else:
                    next_start_ind, next_end_ind = None, None
                    dbg.p("Warning, last photon segment smaller than 50 photons and was not tested", "Change Point")
            else:
                if prev_start_ind < self.cpts[-1] < prev_end_ind:
                    next_start_ind, next_end_ind = self.cpts[-1] + 1, self.cpts[-1] + 1001
                else:
                    next_start_ind, next_end_ind = prev_end_ind - 200, prev_end_ind + 800
        return next_start_ind, next_end_ind

    def __find_all_cps__(self, seg_inds=None):
        is_top_level = False

        if self._finding is False:
            is_top_level = True
            self._finding = True
            assert seg_inds is None, "ChangePoints:\tDo not provide seg_inds when calling, it's used for recursive calling only."
        if is_top_level:
            seg_inds = self.__next_seg_ind__()
        else:
            seg_inds = self.__next_seg_ind__(prev_seg_inds=seg_inds)

        local2global_offset = seg_inds[0] - 1
        cpt_found, found_cpt = self.__weighted_likelihood_ratio__(seg_inds)
        if cpt_found:
            self.__find_all_cps__((seg_inds[0], found_cpt - 1))  # Left
            self.__find_all_cps__((found_cpt, seg_inds[1]))  # Right
        elif is_top_level:
            self._finding = False
            self.found_cpts = True

    def run_cpa(self, confidence=None):
        if confidence is not None:
            assert confidence in [0.99, 0.95, 0.90, 0.69], "ChangePoints:\tConfidence value given not valid."
            self.confidence = confidence
            self.tau = TauData(confidence)
        else:
            assert self.confidence is not None, "ChangePoints:\tNo confidence value provided."
        self.__find_all_cps__()


class Level:
    def __init__(self, particle=None, level_inds=None):
        # assert h5file is not None, "No HDF5 has been given"  # To ensure that a h5file is given
        assert type(particle) is smsh5.Particle, "Level:\tNo Particle object given."
        self.particle = smsh5.Particle.__copy__(particle)
        assert type(level_inds) is tuple, "Level:\tLevel indexs argument is not a tuple (start, end)."
        self.level_inds = level_inds
        self.photon_first = self.level_inds[0]
        self.photon_last = self.level_inds[1]
        self.num_photons = self.photon_first - self.photon_last + 1
        self.time_start = self.particle.abstimes[self.photon_first]
        self.time_last = self.particle.abstimes[self.photon_last]
        self.dwell_time = self.time_start - self.time_last
        self.int = self.num_photons / self.dwell_time
        pass


# def find_change_points(self):


# print('Start')
# h5_file = H5()
# particles = h5_file.particles
# tau = TauData()
# for part_name, part in particles.items():
# 	print(part_name+': '+part.meta.user)

file = smsh5.H5dataset('LHCII_630nW.h5')
file.particles[0].cpa = ChangePoints(file.particles[0], )
# # cpts_analysis.__next_seg_ind__()
# inds = (3025, 4026)
# print(cpts_analysis.__next_seg_ind__(inds))
# cpts_analysis.cpts = [3333]
# inds = (3025, 4026)
# print(cpts_analysis.__next_seg_ind__(inds))
# inds = (4025, 5026)
# print(cpts_analysis.__next_seg_ind__(inds))
# tau_test = TauData(confidence=0.99)
file.particles[0].cpa
pass

# def atoi(text):
#     return int(text) if text.isdigit() else None
#
#
# def natural_keys(text):
#     return [int(c) for c in re.split('(\d+)', text) if text.isdigit()]
#
#
# my_list = ['Hello1', 'Hello12', 'Hello29', 'Hello2', 'Hello17', 'Hello25']
# my_list.sort(key=natural_keys)
# print(my_list)
#
# def atoi(text):
#     return int(text) if text.isdigit() else None
# def natural_keys(text):
#     return [ atoi(c) for c in re.split('(\d+)',text) ]
# my_list =['Hello1', 'Hello12', 'Hello29', 'Hello2', 'Hello17', 'Hello25']
# my_list.sort(key=natural_keys)
# print(my_list)
#
# natural_keys = []
# for name in my_list:
#     for seg in re.split('(\d+)', name):
#         if seg.isdigit():
#             natural_keys.append(int(seg))
