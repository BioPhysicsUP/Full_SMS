"""Module for handling performing Agglomerative Hierarchical Clustering Algorithm.

Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)

Joshua Botha
University of Pretoria
2018
"""

from math import lgamma

import numpy as np
from matplotlib import pyplot as plt


class Calcs:
    """
    Class for storing calculation variables.
    """

    def __init__(self):
        self.cap_j = None
        self.p_mj = None
        self.em_p_mj = None
        self.cap_g = None
        self.n = None
        self.cap_t = None
        self.tot_t = None
        self.merged = None
        self.bic = None
        self.cap_g = None

    def setup(self, particle):
        """
        Set up initial values for all calcs.

        Parameters
        ----------
        particle: sms5h.Particle
            Parent Particle instance.

        Returns
        -------

        """
        assert hasattr(particle, 'levels'), 'No levels have been resolved to merge.'

        self.cap_j = particle.num_cpts
        self.p_mj = np.identity(particle.num_levels)
        self.em_p_mj = np.identity(particle.num_levels)
        self.cap_g = particle.num_levels
        self.n = np.array([l.num_photons for l in particle.levels])
        self.cap_t = np.array([l.dwell_time / 1E9 for l in particle.levels])
        self.tot_t = np.sum(self.cap_t)
        self.merged = list()
        self.bic = list()


def max_mj(array: np.ndarray):
    """ Finds the j and m index for the max value. """
    max_n = np.argmax(array)
    (xD, yD) = array.shape
    if max_n >= xD:
        max_m = max_n//xD
        max_j = max_n%xD
    else:
        max_j = max_n
        max_m = 0
    return max_m, max_j


def g(n: np.int, cap_i: np.float, cap_t: np.float) -> np.float:
    """
    Poisson probability of detection n number of photons in a level given I and T. Eq (2)

    Parameters
    ----------
    n: np.int
        Number of photons in level.
    cap_i: np.float
        Intensity of level.
    cap_t: np.float
        Dwell time of level.

    Returns
    -------
    np.float
        Poisson probability
    """

    return np.exp(n*np.log(cap_i*cap_t) - (cap_i*cap_t) - np.float(lgamma(n + 1)))


class AHCA:
    """
    Class for executing Agglomerative Hierarchical Clustering Algorithm and storing the results.
    """

    def __init__(self, particle):
        """

        Parameters
        ----------
        particle: smsh5.Particle
        """

        self._calcs = Calcs()
        self.has_grouped = False
        self.particle = particle

    def run_grouping(self):
        """
        Run grouping

        Returns
        -------

        """

        self._calcs.setup(self.particle)

        ############################
        plotting_on = True
        ############################
        if plotting_on:
            cap_j = self._calcs.cap_j
            rows = int(1 + np.sqrt(cap_j + 1)//1)
            columns = int(rows + np.ceil(np.sqrt(cap_j + 1) - rows))
            fig_p_mj, ax_p_mj = plt.subplots(rows, columns, sharex='col', sharey='row')
            fig_p_mj.subplots_adjust(hspace=0.3, wspace=0.1)
            fig_p_mj.canvas.set_window_title('p_mj')
            fig_p_mj.suptitle('Plots of p_mj')
            fig_em_p_mj, ax_em_p_mj = plt.subplots(rows, columns, sharex='col', sharey='row')
            fig_em_p_mj.subplots_adjust(hspace=0.3, wspace=0.1)
            fig_em_p_mj.canvas.set_window_title('em_p_mj')
            fig_em_p_mj.suptitle('Plots of em_p_mj')

        states = []
        all_em_p_mj = []
        for i in range(self._calcs.cap_j):
            self.ahg()
            self.em()
            # print(f"Number of States: {self._calcs.cap_g}, BIC: {self._calcs.bic[-1]}")
            all_em_p_mj.append(self._calcs.em_p_mj)
            if plotting_on:
                row = i//columns
                column = i - columns*(i//columns)
                title = f"i={i+1}, #S={self._calcs.cap_g}"
                ax_p_mj[row, column].pcolor(self._calcs.p_mj, edgecolors='k', linewidths=0.1, cmap='GnBu')
                ax_p_mj[row, column].set_title(title)
                ax_em_p_mj[row, column].pcolor(self._calcs.em_p_mj, edgecolors='k', linewidths=0.1, cmap='GnBu')
                ax_em_p_mj[row, column].set_title(title)
                states.append(self._calcs.cap_g)

        if plotting_on:
            states_x = [str(s) for s in states]
            fig_bics, ax_bics = plt.subplots()
            fig_bics.canvas.set_window_title('BIC')
            ax_bics.set_title('BIC of grouping')
            ax_bics.set_xlabel('Number of states')
            ax_bics.set_ylabel('BIC')
            ax_bics.plot(states, self._calcs.bic, marker='o')
        pass

    # Step 1
    #######################################################
    def ahg(self):
        """ Run initial Agglomerative Hierarchical Grouping """

        cap_j = self._calcs.cap_j
        n = self._calcs.n
        cap_t = self._calcs.cap_t

        merge_mj = np.full((cap_j + 1, cap_j + 1), -np.inf)
        p_mj = self._calcs.p_mj
        for j in range(cap_j + 1):  # column
            if j not in self._calcs.merged:
                n_j = np.sum(n*p_mj[:, j])
                cap_t_j = np.sum(cap_t*p_mj[:, j])
                for m in range(j + 1, cap_j + 1):  # row
                    if m not in self._calcs.merged:
                        n_m = np.sum(n*p_mj[m, :])
                        cap_t_m = np.sum(cap_t*p_mj[:, m])
                        merge_mj[m, j] = (n_m + n_j)*np.log((n_m + n_j)/(cap_t_m + cap_t_j)) \
                                         - n_m*np.log(n_m/cap_t_m) - n_j*np.log(n_j/cap_t_j)

        max_m, max_j = max_mj(merge_mj)
        m_merging = np.flatnonzero(self._calcs.p_mj[max_m, :])
        # self._calcs.p_mj[m_merging, max_j] = 1
        self._calcs.p_mj[max_j, m_merging] = 1
        self._calcs.p_mj[max_m, m_merging] = 0
        # self._calcs.p_mj[max_m, m_merging] = 0
        self._calcs.merged.append(max_m)
        self._calcs.cap_g -= 1
        # print(f"Merged {m_merging} into {max_j}")

    # Step 2
    #######################################################
    def em(self):
        """ Expectation Maximisation clustering """

        p_mj = self._calcs.p_mj

        i = 0
        max_diff = 1
        while max_diff > 1E-10 and i < 500:

            i += 1

            cap_j = self._calcs.cap_j
            cap_t = self._calcs.cap_t
            n = self._calcs.n

            # M-Step
            cap_t_m = np.zeros(cap_j + 1)
            n_m = np.zeros(cap_j + 1)
            p_m = np.zeros(cap_j + 1)
            cap_i_m = np.zeros(cap_j + 1)

            for m in range(cap_j + 1):
                if m not in self._calcs.merged:
                    cap_t_m[m] = np.sum(cap_t*p_mj[m, :])
                    n_m[m] = np.sum(n*p_mj[m, :])
                    p_m[m] = cap_t_m[m]/self._calcs.tot_t
                    cap_i_m[m] = np.sum((n/cap_t_m[m])*p_mj[m, :])

            # E-Step
            denom_j = np.zeros(cap_j + 1)
            for j in range(cap_j+1):
                #     p_m_g = np.array([p_m[m]*g(n[j], cap_i_m[m], cap_t[j])
                #                       for m in range(cap_j + 1) if m not in self._calcs.merged])
                # if j not in self._calcs.merged:
                n_j = n[j]
                cap_t_j = cap_t[j]
                denom_j[j] = np.sum([p_m[m]*g(n_j, cap_i_m[m], cap_t_j) for m in range(cap_j + 1) if m not in self._calcs.merged])
                # denom_j[j] = np.sum([p_m[m]*g(n_j, cap_i_m[m], cap_t_j) for m in range(cap_j + 1)
                #                   if m not in self._calcs.merged])

            new_p_mj = np.zeros_like(p_mj)
            p_m_g = np.zeros_like(p_mj)
            for j in range(cap_j + 1):  # column
            # if j not in self._calcs.merged:
                for m in range(cap_j + 1):  # row
                    if m not in self._calcs.merged:
                        p_m_g[m, j] = p_m[m]*g(n[j], cap_i_m[m], cap_t[j])
                        # p_m_g[j, m] = p_m_g[m, j]
                        # if denom_j[j] != 0:
                        new_p_mj[m, j] = p_m_g[m, j]/denom_j[j]
                        # new_p_mj[j, m] = new_p_mj[m, j]

            max_diff = np.max(abs(new_p_mj - p_mj))
            p_mj = new_p_mj

        self._calcs.em_p_mj = p_mj
        # round_p_mj = np.round(p_mj, 3)
        # sum_p_mj = np.sum(p_mj, 0)
        # final_p_mj = np.zeros_like(p_mj)
        # for m, j in enumerate(np.argmax(p_mj, 0)):
        #     final_p_mj[m, j] = 1
        #     final_p_mj[j, m] = 1

        # Step 3
        #######################################################
        log_l_em_mj = np.zeros((cap_j + 1, cap_j + 1))
        g_value = np.float(0)
        for j in range(cap_j + 1):  # column
            # if j not in self._calcs.merged:
            for m in range(cap_j + 1):
                if m not in self._calcs.merged:
                    g_value = g(n[j], cap_i_m[m], cap_t[j])
                    if g_value != 0:
                        np.seterr(under='raise')
                        try:
                            log_l_em_mj[m, j] = p_mj[m, j]*np.log(p_m[m]*g_value)
                        except FloatingPointError:
                            log_l_em_mj[m, j] = 0
                        np.seterr(under='warn')
            # for m in range(cap_j + 1):  # row
            #     if p_m_g[m, j] != 0:
            #         log_cap_l_em += p_mj[m, j]*np.log(p_m_g[m, j])

        log_l_em = np.sum(np.sum(log_l_em_mj))
        n_g = self._calcs.cap_g
        cap_n_cp = np.float(self.particle.num_cpts)
        cap_n = self.particle.num_photons

        # TODO Calculate n_g using sum of row -> less 1E-10 = 0 -> count non-zero elements

        bic = 2*log_l_em - (2*n_g - 1)*np.log(cap_n_cp) - cap_n_cp*np.log(cap_n)

        self._calcs.bic.append(bic)


def main():
    """
    Tests the AHCA class init.
    """

    test = AHCA()


if __name__ == '__main__':
    main()
