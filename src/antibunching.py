""" Module for performing antibunching-type calculations.

Currently only performs simple second-order correlation.
More advanced functionality might be added in the future.

Bertus van Heerden,
University of Pretoria,
2023
"""
from __future__ import annotations

__docformat__ = "NumPy"

import numpy as np
from my_logger import setup_logger

logger = setup_logger(__name__)


class AntibunchingAnalysis:
    """Performs second-order correlation and stores the result.

    Parameters:
    -----------
    particle : smsh5.particle
        particle object to perform analysis on."""
    def __init__(self, particle):
        self._particle = particle
        self.uuid = self._particle.uuid
        self.corr_hist = None
        self.corr_bins = None
        self.corr_events = None
        self.has_corr = False

    def correlate_particle(self, difftime: float, window: float, binsize: float):
        """Calculate second-order correlation for this particle.

        Mainly a wrapper around static method `correlate_times`.
        Stores result in class attributes `corr_bins`, `corr_hist` and `corr_events` and sets `has_corr` to `True`.

        Arguments:
        ----------
        difftime : float
            time difference between channels (ch. 1 - ch. 2) in ns
        window : float
            time window for correlation in ns
        binsize : float
            bin_old size for correlation histogram in ns
        """

        abstimes1 = self._particle.abstimes[:]
        abstimes2 = self._particle.sec_part.abstimes[:]
        microtimes1 = self._particle.microtimes[:]
        microtimes2 = self._particle.sec_part.microtimes[:]
        bins, hist, events = self.correlate_times(
            abstimes1, abstimes2, microtimes1, microtimes2, difftime, window, binsize
        )
        self.corr_bins = bins
        self.corr_hist = hist
        self.corr_events = events
        self.has_corr = True
        logger.info(msg=f"{self._particle.name} photons correlated")

    @staticmethod
    def correlate_times(
        abstimes1,
        abstimes2,
        microtimes1,
        microtimes2,
        difftime=0.0,
        window=500.0,
        binsize=0.5,
    ):
        """Calculate second-order correlation based on time-tagged time-resolved photon data.

        The function is a simple nested loop that runs through every photon within a certain window and
        checks for coincidences. Each coincidence gets a relative time, which are all put in a histogram
        to get the second-order correlation. Before the calculation, the arrival times are corrected
        based on the difftime parameter, which accounts for possible delay between two TCSPC cards.

        Arguments:
        ----------
        abstimes1 : 1D array
            absolute times for channel 1 in ns
        abstimes2 : 1D array
            absolute times for channel 2 in ns
        microtimes1 : 1D array
            micro times for channel 1 in ns
        microtimes2 : 1D array
            micro times for channel 2 in ns
        difftime : float
            time difference between channels (ch. 1 - ch. 2) in ns
        window : float
            time window for correlation in ns
        binsize : float
            bin_old size for correlation histogram in ns

        Returns:
        --------
        bins : 1D array
            correlation histogram bins
        corr : 1D array
            correlation histogram values
        events : 1D array
            difftimes used to construct histogram, returned in case rebinning is needed.
        """
        abstimes1 = abstimes1 + microtimes1
        abstimes2 = abstimes2 + microtimes2 + difftime
        size1 = np.size(abstimes1)
        size2 = np.size(abstimes2)
        channel = np.concatenate(
            (np.zeros(size1), np.ones(size2))
        )  # create list of channels for each photon (ch. 0 or ch. 1)
        all_times = np.concatenate((abstimes1, abstimes2))
        ind = all_times.argsort()
        all_times = all_times[ind]
        channel = channel[ind]  # sort channel array to match times

        events = []
        for i, time1 in enumerate(all_times):
            for j, time2 in enumerate(all_times[i:]):
                channel1 = channel[i]
                channel2 = channel[i + j]
                if channel1 == channel2:
                    continue  # ignore photons from same card
                difftime = time2 - time1
                if channel1 == 1:
                    difftime = - difftime  # channel 0 is start channel
                if abs(difftime) > window:
                    break
                events.append(difftime)
        numbins = int(window / binsize)
        corr, bins = np.histogram(events, numbins)
        events = np.array(events)
        return bins[:-1], corr, events

    def rebin_corr(self, window, binsize):
        numbins = int(window / binsize)
        corr, bins = np.histogram(self.corr_events, numbins)
        self.corr_bins = bins[:-1]
        self.corr_hist = corr

