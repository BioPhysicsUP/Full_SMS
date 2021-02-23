"""Module to calculates, stores and provides access to the sums that are used for the
change point detection algorithm in the change_point module.

Written by Joshua Botha, University of Pretoria, Physics department.
"""

__docformat__ = 'NumPy'

import os
import numpy as np
import pickle
import dbg
import file_manager as fm
from processes import ProcessProgFeedback
from my_logger import setup_logger


logger = setup_logger(__name__)


class CPSums:
    """Calculates, stores and get's sum values."""
    
    def __init__(self, n_max: int = None, n_min: int = None,
                 only_pickle: bool = False, prog_fb: ProcessProgFeedback = None):
        self._version = 1.4
        self.prog_fb = prog_fb

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
            all_sums_path = fm.path('all_sums.pickle', fm.Type.Data)
            if os.path.exists(all_sums_path) and os.path.isfile(all_sums_path):
                with open(all_sums_path, 'rb') as all_sums_file:
                    all_sums:dict = pickle.load(all_sums_file)
                if ('version' not in all_sums.keys()) or all_sums['version'] != self._version:
                    self._calc_and_store()
                else:
                    self._sums_u_k = all_sums['sums_u_k']
                    self._sums_u_n_k = all_sums['sums_u_n_k']
                    self._sums_v2_k = all_sums['sums_v2_k']
                    self._sums_v2_n_k = all_sums['sums_v2_n_k']
                    self._sums_sig_e = all_sums['sums_sig_e']
                    logger.info('all_sums.pickle found')
            else:
                self._calc_and_store()

    def _calc_sums(self) -> None:
        """
        Calculates the all the possible sums that might be used in the change_point module to detect
        change points.
        """
        
        for n in range(self.n_min+1, self.n_max+1):
            self._sums_sig_e[n-self.n_min-1] = (np.pi**2)/6 \
                                                    - sum(1/j**2 for j in range(1, (n-1)+1))
            for k in range(1, n):
                self._sums_u_k[k-1, n-self.n_min-1] = -sum(1/j for j in range(k, (n-1)+1))
                self._sums_u_n_k[k-1, n-self.n_min-1] = -sum(1/j for j in range(n-k, (n-1)+1))
                self._sums_v2_k[k-1, n-self.n_min-1] = sum(1/j**2 for j in range(k, (n-1)+1))
                self._sums_v2_n_k[k-1, n-self.n_min-1] = sum(1/j**2 for j in range(n-k, (n-1)+1))

    def _calc_and_store(self):
        logger.info('Calculating all_sums.pickle')
        self.prog_fb.set_status(status="Calculating sums...")
        self._calc_sums()
        all_sums = {
            'version': self._version,
            'sums_u_k': self._sums_u_k,
            'sums_u_n_k': self._sums_u_n_k,
            'sums_v2_k': self._sums_v2_k,
            'sums_v2_n_k': self._sums_v2_n_k,
            'sums_sig_e': self._sums_sig_e
        }
        with open(fm.path('all_sums.pickle', fm.Type.Data), 'wb') as all_sums_file:
            pickle.dump(all_sums, all_sums_file)
            logger.info('all_sums.pickle calculated and saved')

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
