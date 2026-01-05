import numpy as np
import h5py
import pyarrow.parquet as pq
from datetime import datetime
from matplotlib import pyplot as plt


def correlate_times(
        abstimes1,
        abstimes2,
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
    difftime : float
        time difference between channels (ch. 1 - ch. 2) in ns
    window : float
        time window for correlation in ns
    binsize : float
        bin size for correlation histogram in ns

    Returns:
    --------
    bins : 1D array
        correlation histogram bins
    corr : 1D array
        correlation histogram values
    events : 1D array
        difftimes used to construct histogram, returned in case rebinning is needed.
    """
    abstimes1 = abstimes1
    abstimes2 = abstimes2
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
    laser_times = []
    for i, time1 in enumerate(all_times):
        for j, time2 in enumerate(all_times[i:]):
            channel1 = channel[i]
            channel2 = channel[i + j]
            if channel1 == channel2:
                continue  # ignore photons from same card
            difftime = time2 - time1
            if difftime > window:  # 500 ns window
                break
            if channel2 == 0:
                events.append(difftime)
                laser_times.append(time2)
    numbins = int(window / binsize)
    corr, bins = np.histogram(events, numbins)
    events = np.array(events)
    laser_times = np.array(laser_times)
    return bins[:-1], corr, events, laser_times


chA = pq.read_table('koenderink/Dot_15/timestamps_chA_bin.parquet')
chB = pq.read_table('koenderink/Dot_15/timestamps_chB_bin.parquet')
chR = pq.read_table('koenderink/Dot_15/timestamps_chR_bin.parquet')

chA = np.int64(np.array(chA).flatten()) #* 0.16461
chB = np.int64(np.array(chB).flatten()) #* 0.16461
chR = np.int64(np.array(chR).flatten()) #* 0.16461

with h5py.File('koenderink/simdata.h5', "a") as h5_f:
    h5_f.attrs.modify(name="# Particles", value=15)
    h5_f.attrs.create("Version", '1.07')
    part_group = h5_f.create_group("Particle 15")
    part_group.attrs.create("Date", datetime.now().strftime("%A, %B %d, %Y, %-I:%M %p"))
    print(datetime.now().strftime("%A, %B %d, %Y, %-I:%M %p"))
    part_group.attrs.create("Description", 'Simulated data from Palstra code.')
    part_group.attrs.create("Intensity?", True)
    part_group.attrs.create("RS Coord. (um)", [0, 0])
    part_group.attrs.create("Spectra?", False)
    part_group.attrs.create("User", 'Bertus')

    # abs_times = chA
    bins, corr, events, abs_times = correlate_times(chR, chA, window=700, binsize=1)
    micro_times = 770 - events
    micro_times = micro_times * 0.16461  # multiply with time resolution (channel width)
    abs_times = abs_times * 0.16461
    # plt.plot(bins, corr)
    # plt.hist(micro_times, bins=100)
    # plt.show()
    # print(np.diff(np.sort(micro_times)[:200]))

    # abs_times2 = chB
    bins2, corr2, events2, abs_times2 = correlate_times(chR, chB, window=700, binsize=1)
    micro_times2 = 770 - events2
    micro_times2 = micro_times2 * 0.16461
    abs_times2 = abs_times2 * 0.16461
    # plt.hist(micro_times2, bins=100)
    # plt.show()

    abs1data = part_group.create_dataset("Absolute Times (ns)", dtype=np.uint64, data=abs_times)
    abs1data.attrs.create('bh Card', '1')
    part_group.create_dataset("Micro Times (ns)", dtype=np.float64, data=micro_times)
    abs2data = part_group.create_dataset("Absolute Times 2 (ns)", dtype=np.uint64, data=abs_times2)
    abs2data.attrs.create('bh Card', '2')
    part_group.create_dataset("Micro Times 2 (ns)", dtype=np.float64, data=micro_times2)

    part_group.create_dataset("Intensity Trace (cps)", dtype=np.float64, data=np.zeros(10))
