"""Module to calculates, stores and provides access to the sums that are used for the
change point detection algorithm in the change_point module.

Written by Joshua Botha, University of Pretoria, Physics department.
"""

__docformat__ = 'NumPy'

import os
import numpy as np
import pickle
import dbg
from PyQt5.QtCore import pyqtSignal


class CPSums:
    """Calculates, stores and get's sum values."""
    
    def __init__(self, n_max: int = None, n_min: int = None,
                 only_pickle: bool = False, auto_prog_sig: pyqtSignal = None):
        self._version = 1.4
        self.auto_prog_sig = auto_prog_sig
        
        if n_max is None:
            self.n_max = 1000
        else:
            self.n_max = n_max

        if n_min is None:
            self.n_min = 10
        else:
            self.n_min = n_min

        n_range = self.n_max-self.n_min
        self._sums_u_k = np.empty(shape=(self.n_max, n_range), dtype=np.float64)
        self._sums_u_n_k = np.empty(shape=(self.n_max, n_range), dtype=np.float64)
        self._sums_v2_k = np.empty(shape=(self.n_max, n_range), dtype=np.float64)
        self._sums_v2_n_k = np.empty(shape=(self.n_max, n_range), dtype=np.float64)
        self._sums_sig_e = np.empty(shape=n_range, dtype=np.float64)
        
        if only_pickle:
            self._calc_and_store()
        else:
            if os.path.exists(os.getcwd()+'\\all_sums.pickle') and os.path.isfile(os.getcwd()+'\\all_sums.pickle'):
                all_sums_file = open('all_sums.pickle', 'rb')
                all_sums = dict(pickle.load(all_sums_file))
                all_sums_file.close()
                if ('version' not in all_sums.keys()) or all_sums['version'] != self._version:
                    self._calc_and_store()
                else:
                    self._sums_u_k = all_sums['sums_u_k']
                    self._sums_u_n_k = all_sums['sums_u_n_k']
                    self._sums_v2_k = all_sums['sums_v2_k']
                    self._sums_v2_n_k = all_sums['sums_v2_n_k']
                    self._sums_sig_e = all_sums['sums_sig_e']
            else:
                self._calc_and_store()
    
    def _calc_sums(self) -> None:
        """
        Calculates the all the possible sums that might be used in the change_point module to detect
        change points.
        """
        
        tot_inds = sum(n for n in range(self.n_min, self.n_max))
        # print(tot_inds)
        accum_inds = int()
        dbg.u('Calculating sums: [{0}] {1}%'.format(' '*20, 0.0), 'CPSums')
        for n in range(self.n_min+1, self.n_max+1):
            self._sums_sig_e[n-self.n_min-1] = (np.pi**2)/6 \
                                                    - sum(1/j**2 for j in range(1, (n-1)+1))
            for k in range(1, n):
                self._sums_u_k[k-1, n-self.n_min-1] = -sum(1/j for j in range(k, (n-1)+1))
                self._sums_u_n_k[k-1, n-self.n_min-1] = -sum(1/j for j in range(n-k, (n-1)+1))
                self._sums_v2_k[k-1, n-self.n_min-1] = sum(1/j**2 for j in range(k, (n-1)+1))
                self._sums_v2_n_k[k-1, n-self.n_min-1] = sum(1/j**2 for j in range(n-k, (n-1)+1))

            accum_inds += n-1
            prog = round(100*accum_inds/tot_inds, 1)
            if self.prog_sig is not None:
                self.prog_sig.emit(prog, 'Calculating change point sums...')
            prog20 = int(prog//5)
            dbg.u('Calculating sums: [{0}{1}] {2}%'.format('#'*prog20, ' '*(20-prog20), prog),
                  'CPSums', n == self.n_max)
        # dbg.p('Calculating sums: [{0}] {1}%'.format('#'*20, 100.0), 'CPSums')
    
    def _calc_and_store(self):
        self._calc_sums()
        all_sums = {
            'version': self._version,
            'sums_u_k': self._sums_u_k,
            'sums_u_n_k': self._sums_u_n_k,
            'sums_v2_k': self._sums_v2_k,
            'sums_v2_n_k': self._sums_v2_n_k,
            'sums_sig_e': self._sums_sig_e
        }
        all_sums_file = open('all_sums.pickle', 'wb')
        pickle.dump(all_sums, all_sums_file)
        all_sums_file.close()
    
    def get_u_k(self, n, k) -> np.float64:
        """
        Used to get the sum value for u_k.

        Parameters
        ----------
        n : int
            The number of points in segment.
        k : int
            The number of the point in the segment being considered.

        Returns
        -------
        sum_u_k : np.float64
            As defined just after eq. 6
        """
        row = k-1
        column = n-self.n_min-1
        return self._sums_u_k[row, column]
    
    def get_u_k_n(self, n, k) -> np.float64:
        """
        Used to get the sum value for u_k_n.

        Parameters
        ----------
        n : int
            The number of points in segment.
        k : int
            The number of the point in the segment being considered.

        Returns
        -------
        sum_u_k : np.float64
            As defined just after eq. 6
        """
        row = k-1
        column = n-self.n_min-1
        return self._sums_u_n_k[row, column]
    
    def get_v2_k(self, n, k) -> np.float64:
        """
        Used to get the sum value for v2_k.

        Parameters
        ----------
        n : int
            The number of points in segment.
        k : int
            The number of the point in the segment being considered.

        Returns
        -------
        sum_u_k : np.float64
            AAs defined just before eq. 7
        """
        row = k-1
        column = n-self.n_min-1
        return self._sums_v2_k[row, column]
    
    def get_v2_k_n(self, n, k) -> np.float64:
        """
        Used to get the sum value for v2_k_n.

        Parameters
        ----------
        n : int
            The number of points in segment.
        k : int
            The number of the point in the segment being considered.

        Returns
        -------
        sum_u_k : np.float64
            As defined just before eq. 7
        """
        row = k-1
        column = n-self.n_min-1
        return self._sums_v2_n_k[row, column]
    
    def get_sig_e(self, n) -> np.float64:
        """
        Used to get the sum value for sig_e.

        Parameters
        ----------
        n : int
            The number of points in segment.

        Returns
        -------
        sum_u_k : np.float64
            As defined just before eq. 7
        """
        return self._sums_sig_e[n-self.n_min-1]
    
    def get_set(self, n, k):
        """
        Returns a dict with all the sums, except for sig_e.
        
        Parameters
        ----------
        n : int
            The number of points in segment.
        k : int
            The number of the point in the segment being considered.

        Returns
        -------
        set_sums : dict
            Dict of sums
        """
        row = k-1
        column = n-self.n_min-1
        set_sums = {
            'u_k': self._sums_u_k[row, column],
            'u_n_k': self._sums_u_n_k[row, column],
            'v2_k': self._sums_v2_k[row, column],
            'v2_n_k': self._sums_v2_n_k[row, column]
        }
        return set_sums

def main():
    """
    Just a test.
    """
    test = CPSums()
    n = 853
    k = 522
    print(f'Test for n={n} and k={k}')
    print('u_k:\t' + str(test.get_u_k(n, k)))
    print('u_k_n:\t' + str(test.get_u_k_n(n, k)))
    print('v2_k:\t' + str(test.get_v2_k(n, k)))
    print('v2_k_n:\t' + str(test.get_v2_k_n(n, k)))
    print('sig_e:\t' + str(test.get_sig_e(n, k)))


if __name__ == '__main__':
    main()
