"""Module for handling analysis of change points and creation of consequent levels

Joshua Botha
University of Pretoria
2018
"""
import os
import numpy as np
import dbg
# from smsh5 import Particle


class ChangePoints:
    def __init__(self, particle=None, confidence=None, run_levels=None):
        self.__particle = particle
        self.cpa = ChangePointAnalysis(particle)

        if run_levels is not None:
            self._run_levels = run_levels
        else:
            self._run_levels = False

        if confidence is not None:
            self.run_cpa()
        else:
            self.confidence = None
            self.inds = None
            self.conf_regions = None
            self.dt_uncertainty = None
            self.cpa_has_run = False
            self.num_cpts = None

    def run_cpa(self, confidence=None, run_levels=None):
        if run_levels is not None:
            self._run_levels = run_levels
        self.confidence = confidence
        self.inds, self.conf_regions, self.dt_uncertainty = self.cpa.run_cpa(confidence)
        self.num_cpts = self.cpa.num_cpts
        self.cpa_has_run = True
        if self._run_levels:
            self.get_levels()

    def get_levels(self):
        self.cpa.get_levels()


class Level:
    def __init__(self, abstimes=None, level_inds=None):
        # assert h5file is not None, "No HDF5 has been given"  # To ensure that a h5file is given
        # assert type(particle) is smsh5.Particle, "Level:\tNo Particle object given."
        # self.particle = smsh5.Particle.__copy__(particle)
        assert abstimes is not None, "Levels:\tParameter 'abstimes' not given."
        assert level_inds is not None, "Levels:\tParameter 'level_inds' not given."
        assert type(level_inds) is tuple, "Level:\tLevel indexes argument is not a tuple (start, end)."
        self.level_inds = level_inds  # (first_ind, last_ind)
        self.num_photons = self.level_inds[1] - self.level_inds[0] + 1
        self.times = (abstimes[self.level_inds[0]], abstimes[self.level_inds[1]])
        self.dwell_time = self.times[1] - self.times[0]
        self.int = (self.num_photons / self.dwell_time) * 1e9


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


class ChangePointAnalysis:

    def __init__(self, particle=None, confidence=None):
        # assert h5file is not None, "No HDF5 has been given"  # To ensure that a h5file is given
        # assert type(particle) is smsh5.Particle, "ChangePoints:\tNo Particle object given."
        # assert confidence is not None, "ChangePoints:\tNo confidence parameter given."
        self._particle = particle
        self._abstimes = particle.abstimes
        self.num_photons = particle.num_photons
        self.cpts = np.array([], dtype=int)
        self.conf_regions = np.array(tuple())  # [(start, end)]
        self.dt_uncertainty = np.array([])  # dt
        self._finding = False
        self.found_cpts = False
        self.num_cpts = None
        self.confidence = confidence
        self._i = None
        if confidence is not None:
            self._tau = TauData(self.confidence)

    def __weighted_likelihood_ratio(self, seg_inds=None):
        """ Calculates the Weighted & Standardised Likelihood ratio.

        Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
        from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)
        """

        assert type(seg_inds) is tuple, 'ChangePointAnalysis:\tSegment index\'s not given.'
        start_ind, end_ind = seg_inds
        n = end_ind - start_ind
        assert n <= 1000, "ChangePointAnalysis:\tIndex's given result in more than a segment of more than 1000 points."
        print(start_ind, end_ind)
        time_data = self._abstimes[start_ind:end_ind]

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

        max_ind_local = int(wlr.argmax())

        if wlr[max_ind_local] >= self.__tau.get_tau_a(n):
            self.cpts = np.append(self.cpts, max_ind_local + start_ind)
            region_all_local = np.where(wlr >= self.__tau.get_tau_b(n))[0]
            region_local = [region_all_local[0], region_all_local[-1]]
            region = (region_local[0] + start_ind, region_local[1] + start_ind)
            dt = self._abstimes[region_local[1]] - self._abstimes[region_local[0]]
            self.conf_regions = np.append(self.conf_regions, region)
            self.dt_uncertainty = np.append(self.dt_uncertainty, dt)
            cpt_found = True
        else:
            cpt_found = False
        return cpt_found

    def __next_seg_ind(self, prev_seg_inds=None, side=None):
        last_photon_ind = self.num_photons - 1
        #  See code2flow.com for tree:
        #  https://code2flow.com/svLn85

        if prev_seg_inds is None:
            # Data sets need to be larger than 200 photons
            assert self.num_photons >= 200, 'ChangePointAnalysis:\tData set needs to be at least 200 photons for change point detection.'
            if self.num_photons > 1000:
                next_start_ind, next_end_ind = 0, 1000
            else:
                next_start_ind, next_end_ind = 0, last_photon_ind
        else:
            prev_start_ind, prev_end_ind = prev_seg_inds
            if len(self.cpts) == 0:
                if last_photon_ind >= prev_end_ind + 800:
                    next_start_ind, next_end_ind = prev_end_ind - 200, prev_end_ind + 800
                elif last_photon_ind - prev_end_ind >= 10:  # Next segment needs to be at least 10 photons large.
                    next_start_ind, next_end_ind = prev_end_ind - 200, last_photon_ind
                else:
                    next_start_ind, next_end_ind = None, None
                    dbg.p("Warning, last photon segment smaller than 50 photons and was not tested", "Change Point")
            elif side is not None:  # or prev_start_ind < self.cpts[-1] < prev_end_ind
                if side is not None:
                    assert side in ['left', 'right'], "ChangePointAnalysis:\tSide of change point invalid or not specified"
                if side == 'left':
                    next_start_ind, next_end_ind = prev_start_ind, int(self.cpts[-1] - 1)
                else:
                    next_start_ind, next_end_ind = int(self.cpts[-1]), prev_end_ind
            elif last_photon_ind >= prev_end_ind + 800:
                if prev_end_ind - 200 < self.cpts[-1] < prev_end_ind:
                    if last_photon_ind >= self.cpts[-1] + 1000:
                        next_start_ind, next_end_ind = int(self.cpts[-1]), int(self.cpts[-1]) + 1000
                    else:
                        next_start_ind, next_end_ind = int(self.cpts[-1]), last_photon_ind
                else:
                    next_start_ind, next_end_ind = prev_end_ind - 200, prev_end_ind + 800
            elif last_photon_ind - prev_end_ind >= 10:  # Next segment needs to be at least 10 photons large.
                if prev_end_ind - 200 < self.cpts[-1] < prev_end_ind:
                    next_start_ind, next_end_ind = int(self.cpts[-1]) + 1, last_photon_ind
                else:
                    next_start_ind, next_end_ind = prev_end_ind - 200, last_photon_ind
            else:
                next_start_ind, next_end_ind = None, None
                if prev_end_ind != last_photon_ind:
                    dbg.p("Warning, last photon segment smaller than 10 photons and was not tested", "Change Point")

        return next_start_ind, next_end_ind

    def __find_all_cpts(self, seg_inds=None, side=None):
        is_top_level = False
        # cpt_found = False

        if self._finding is False:
            is_top_level = True
            self._finding = True
            assert seg_inds is None, "ChangePointAnalysis:\tDo not provide seg_inds when calling, it's used for recursive calling only."

        if is_top_level:
            seg_inds = self.__next_seg_ind()
            self._i = 0
        else:
            seg_inds = self.__next_seg_ind(prev_seg_inds=seg_inds, side=side)
            print(seg_inds)
        self._i += 1
        print(self._i)

        if seg_inds != (None, None):
            cpt_found = self.__weighted_likelihood_ratio(seg_inds)
            # if seg_inds[1] != self.num_photons - 1:
                # assert side.lower() in ['left', 'right', 'l', 'r'], "ChangePointAnalysis:\tSide argument needs to be 'left' or 'right'."
            if cpt_found:
                self.__find_all_cpts(seg_inds, side='left')  # Left side of change point
                self.__find_all_cpts(seg_inds, side='right')
                pass# Right side of change point

            if seg_inds[1] <= self.num_photons + 9 and side is None:
                self.__find_all_cpts(seg_inds)

        if is_top_level:
            self._finding = False
            self.found_cpts = True
            self.cpts.sort()
            self.num_cpts = len(self.cpts)

    def get_levels(self):
        assert self.found_cpts, "ChangePointAnalysis:\tChange point analysis not done, or found no change points."
        num_levels = self.num_cpts + 1
        levels = [None] * num_levels

        for num, cpt in enumerate(self.cpts):
            print(num)
            if num == 0:  # First change point
                start_ind = 0
                end_ind = cpt-1
                levels[num] = Level(self._abstimes, level_inds=(start_ind, end_ind))

            start_ind = cpt
            if num != self.num_cpts-1:  # Not last change point
                end_ind = self.cpts[num + 1]
            else:  # Last photon
                end_ind = self.num_photons - 1
            levels[num] = Level(self._abstimes, level_inds=(start_ind, end_ind))

        self._particle.add_levels(levels, num_levels)
        # return levels, num_levels

    def run_cpa(self, confidence=None):
        if confidence is not None:
            assert confidence in [0.99, 0.95, 0.90, 0.69], "ChangePointAnalysis:\tConfidence value given not valid."
            self.confidence = confidence
            self.__tau = TauData(confidence)
        else:
            assert self.confidence is not None, "ChangePointAnalysis:\tNo confidence value provided."

        self.__find_all_cpts()

        return self.cpts, self.conf_regions, self.dt_uncertainty

###### Natural Key Sorting ########
# natural_keys = []
# for name in my_list:
#     for seg in re.split('(\d+)', name):
#         if seg.isdigit():
#             natural_keys.append(int(seg))
