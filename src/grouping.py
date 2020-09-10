"""Module for handling performing Agglomerative Hierarchical Clustering Algorithm.

Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)

Joshua Botha
University of Pretoria
2018
"""

from __future__ import annotations
from math import lgamma

from typing import List, TYPE_CHECKING
import numpy as np
from scipy.stats import poisson
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from smsh5 import Particle

import dbg


def max_jm(array: np.ndarray):
    """ Finds the j and m index for the max value. """
    max_n = np.argmax(array)
    (xD, yD) = array.shape
    if max_n >= xD:
        max_j = max_n // xD
        max_m = max_n % xD
    else:
        max_m = max_n
        max_j = 0
    return max_j, max_m


class Group:

    def __init__(self, lvls_inds: List[int] = None, particle: Particle = None):
        self.lvls_inds = lvls_inds
        self.lvls = None

        if self.lvls_inds is not None and particle is not None:
            self.lvls = [particle.levels[i] for i in self.lvls_inds]

    @property
    def num_photons(self) -> int:
        return int(np.sum([level.num_photons for level in self.lvls]))

    @property
    def dwell_time(self) -> float:
        return float(np.sum([level.dwell_time_s for level in self.lvls]))


class Solution:

    def __init__(self, ahca_steps: List[ClusteringStep]):
        self._best_step = ahca_steps[np.argmax([step.bic for step in ahca_steps])]

    @property
    def groups(self) -> List[Group]:
        return self._best_step.groups

    @property
    def num_groups(self) -> int:
        return self._best_step.num_groups

    @property
    def num_levels(self) -> int:
        return self._best_step.num_levels


class ClusteringStep:

    def __init__(self,
                 particle: Particle,
                 first: bool = False,
                 groups: List[Group] = None,
                 merged_p_mj: np.ndarray = None):

        self.particle = particle
        if first:
            assert groups is None and merged_p_mj is None, "ClusteringStep: parameters not provided"
        self.num_levels = particle.num_levels
        self.log_l_em = None
        self.em_p_mj = None
        self.em_log_l = None
        self.bic = None

        if first:
            self.groups = [Group([i], particle) for i in range(self.num_levels)]
            self.ini_p_mj = np.identity(n=self.num_levels)
        else:
            self.groups = groups
            self.ini_p_mj = merged_p_mj

        self.num_groups = len(self.groups)

    def ahc(self) -> ClusteringStep:

        merge = np.full(shape=(self.num_groups, self.num_groups), fill_value=-np.inf)
        for j, group_j in enumerate(self.groups):  # Row
            for m, group_m in enumerate(self.groups):  # Column
                if j < m:
                    n_m = group_m.num_photons
                    n_j = group_j.num_photons
                    t_m = group_m.dwell_time
                    t_j = group_j.dwell_time

                    merge[j, m] = (n_m + n_j) * np.log((n_m + n_j) / (t_m + t_j)) - n_m * np.log(n_m / t_m) - n_j * np.log(
                        n_j / t_j)

        max_j, max_m = max_jm(merge)

        new_groups = []
        new_p_mj = np.zeros(shape=(self.num_groups-1, self.num_levels))
        group_num = -1
        for i, group_i in enumerate(self.groups):
            if i != max_m:
                group_num += 1
                if i == max_j:
                    merge_lvls = group_i.lvls_inds.copy()
                    merge_lvls.extend(self.groups[max_m].lvls_inds.copy())
                    merge_lvls.sort()
                    new_groups.append(Group(lvls_inds=merge_lvls, particle=self.particle))
                else:
                    new_groups.append(group_i)

                for ind in new_groups[-1].lvls_inds:
                    new_p_mj[group_num, ind] = 1

        return ClusteringStep(particle=self.particle, first=False, groups=new_groups, merged_p_mj=new_p_mj)

    def emc(self):
        """ Expectation Maximisation clustering """

        p_mj = self.ini_p_mj.copy()
        prev_p_mj = p_mj.copy()
        levels = self.particle.levels

        i = 0
        diff_p_mj = 1
        while diff_p_mj > 1E-10 and i < 500:

            i += 1
            # cap_j_plus_1 = self.num_levels
            cap_t = self.particle.dwell_time

            t_hat = np.zeros(shape=(self.num_groups,))
            n_hat = np.zeros_like(t_hat)
            p_hat = np.zeros_like(t_hat)
            i_hat = np.zeros_like(t_hat)

            p_hat_g = np.zeros(shape=(self.num_groups, self.num_levels))
            denom = np.zeros(shape=(self.num_levels,))

            for m, group in enumerate(self.groups):

                t_hat[m] = np.sum([p_mj[m, j]*l.dwell_time_s for j, l in enumerate(levels)])
                n_hat[m] = np.sum([p_mj[m, j]*l.num_photons for j, l in enumerate(levels)])
                p_hat[m] = t_hat[m] / cap_t
                i_hat[m] = np.sum([p_mj[m, j]*l.num_photons / t_hat[m] for j, l in enumerate(levels)])

                denom_sum = 0
                for j, l in enumerate(levels):
                    p_hat_g[m, j] = p_hat[m] * poisson.pmf(l.num_photons, i_hat[m] * l.dwell_time_s)

            for j in range(self.num_levels):
                denom[j] = np.sum(p_hat_g[:, j])

                for m in range(self.num_groups):
                    p_mj[m, j] = p_hat_g[m, j] / denom[j]

            diff_p_mj = np.sum(np.abs(prev_p_mj - p_mj))
            prev_p_mj = p_mj.copy()

        level_p_max = np.argmax(p_mj, 0)
        definitive_p_mj = np.zeros_like(p_mj)
        for j in range(self.num_levels):
            definitive_p_mj[level_p_max[j], j] = 1

        self.em_p_mj = definitive_p_mj

        log_l = 0
        for m in range(self.num_groups):
            for j in range(self.num_levels):
                if p_hat_g[m, j] > 1E-300 and definitive_p_mj[m, j] != 0:  # Close to smallest value that doesn't result in -inf
                    log_l += definitive_p_mj[m, j] * np.log(p_hat_g[m, j])
        self.em_log_l = log_l

    def calc_bic(self):

        num_cp = self.particle.cpts.num_cpts
        num_g = np.sum(np.count_nonzero(np.sum(self.em_p_mj, 1)))

        self.bic = 2*self.em_log_l - (2*num_g - 1)*np.log(num_cp) - num_cp*np.log(self.particle.num_photons)


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

        self.has_groups = False
        self.particle = particle
        self.solution = None
        self.steps = None

    def run_grouping(self):
        """
        Run grouping

        Returns
        -------

        """

        if self.particle.has_levels:

            steps = [ClusteringStep(self.particle, first=True)]
            for sol_num in range(self.particle.num_levels - 1):
                new_step = steps[sol_num].ahc()
                new_step.emc()
                new_step.calc_bic()
                steps.append(new_step)

            steps.pop(0)
            self.steps = steps
            self.solution = Solution(ahca_steps=steps)
            self.has_groups = True
        else:
            dbg.p(f"{self.particle.name} has no levels to group", "ACHA")
