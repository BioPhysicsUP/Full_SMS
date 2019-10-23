""" Module that generates quasi-data of a trace to test change point and
clustering function.
"""

__docformat__ = 'NumPy'

import numpy as np
from typing import Tuple, List, Optional, Dict
from matplotlib import pyplot as plt

PLOT_ON = True


def quasi_trace(max_int: Optional[float] = 350,
                min_int: Optional[float] = 60,
                num_level_groups: Optional[int] = 5,
                trace_length: Optional[float] = 60.0,
                short_level: Optional[float] = 0.1,
                long_level: Optional[float] = 5.0,
                noise_type: Optional[str] = 'poisson') ->\
        dict:
    """ Generates a quasi-trace for testing of change point detection and
    clustering.
    
    Parameters
    ----------
    max_int: float, optional
        The maximum duration of level.
    min_int: float, optional
        The maximum duration of level.
    num_level_groups: int
        The number of levels in trace.
    trace_length: float
        The total duration of the level, in seconds.
    short_level: float
        The duration of the shortest level, in seconds.
    long_level: float
        The duration of the longest level, in seconds.
    noise_type: ['poisson', 'gaussian', 'none']
        Type of noise to add to data

    Returns
    -------
    trace: Returns a dictionary with the following structure indexes of the
    detected change points.
    """
    
    if min_int is None:
        min_int = 0.1 * max_int
    assert noise_type in {'none', 'gaussian', 'poisson'},\
        'Argument `noise_type` does not match one of the possible values.'
    
    # Defines the groups of intensities
    int_groups = np.linspace(min_int, max_int, num_level_groups)
    
    dwell_times = np.empty([0, 1])
    tot_time = 0  # time of trace built in loop
    while tot_time < trace_length:
        # Uniform random selection of time between short and long
        new_dt = np.random.uniform(short_level, long_level)
        if tot_time + new_dt > trace_length:  # if tot_time too large
            diff_time = trace_length - tot_time
            if diff_time >= short_level:  # if time left is NOT too short
                new_dt = diff_time
            else:  # if time left is too short
                # new_dt then defined as time between second last dwell time
                # to end of level time
                new_dt = trace_length - (tot_time - dwell_times[-1])
                dwell_times = dwell_times[:-1]  # remove last added dwell time
        tot_time += new_dt
        dwell_times = np.append(dwell_times, new_dt)
    
    tot_time = 0
    times = np.empty([0, 1])
    int_levels = np.empty([0, 1])
    num_ph_levels = np.empty([0, 1])
    for i, dwell in enumerate(dwell_times):
        new_times = np.array([0, 1])  # Clear
        # Add random int for level
        int_levels = np.append(int_levels, np.random.choice(int_groups))
        # Calc number of photons  for level with newest level_int and
        # corresponding dwell time `dwell`
        num_ph_theory = np.int(round(int_levels[i] * dwell))
        if noise_type is 'none':
            new_times = np.linspace(tot_time, dwell + tot_time,
                                    num_ph_theory)
        else:
            new_times = np.array([tot_time])
            while new_times[-1] <= tot_time + dwell:
                if noise_type is 'poisson':
                    # Poisson noise based on num_photons
                    noise_int = np.random.poisson(num_ph_theory) / dwell
                else:  # Then Gaussian
                    # Gaussian noise based on level intensity
                    noise_int = np.random.normal(int_levels[i])
                # dt =  1/noise
                new_times = np.append(new_times,
                                      new_times[-1] + (1 / noise_int))
            # End while
            if new_times[-1] >= tot_time:
                # Remove last photon to prevent overlap with next level
                new_times = new_times[:-1]
            # end if
        # end if
        num_ph_levels = np.append(num_ph_levels, len(new_times))
        # Add photon times (in ns) data to `times`
        times = np.append(times, new_times * 1E9)
        tot_time += dwell  # Add total time to shift data up
    # end for
    
    # Dict to contain data to define trace
    trace = {'times': times, 'int_levels': int_levels,
             'dwell_times': dwell_times, 'num_ph_levels': num_ph_levels,
             'params': {'max_int': max_int,
                        'min_int': min_int,
                        'num_level_groups': num_level_groups,
                        'trace_length': trace_length,
                        'short_level': short_level,
                        'long_level': long_level,
                        'noise_type': noise_type}
             }
    return trace


if __name__ == '__main__':
    trace = quasi_trace()
    
    if PLOT_ON:
        data = trace['times']
        
        binsize_ms = 50
        binsize_ns = binsize_ms * 1E6  # Convert ms to ns
        end_bin_time = (binsize_ms * 1E-3) * round((data[-1] * 1E-9) / (
                binsize_ms * 1E-3))
        endbin = np.int(end_bin_time / (binsize_ms * 1E-3))
        binned = np.zeros(endbin + 1, dtype=np.int)
        bin_times = np.linspace(0, end_bin_time, endbin+1)
        for step, step_time in enumerate(bin_times):
            if step != 0:
                binned[step] = np.size(
                    data[(step_time > data * 1E-9)
                         * (data * 1E-9 > step_time - binsize_ms * 1E-3)])
            if step == 1:
                binned[step - 1] = binned[step]
        binned_p_s = binned * 1 / (binsize_ms * 1E-3)

        plt.plot(bin_times, binned_p_s)
        plt.xlabel('Times (s)')
        plt.ylabel('Intensity (counts/s)')
        plt.show()
        pass
        
        # binsize_ms = 50
        # binsize_ns = binsize_ms * 1E6  # Convert ms to ns
        # end_bin_time = (binsize_ms*1E-3)*round((data[-1]*1E-9)/(
        #         binsize_ms*1E-3))
        # endbin = np.int(end_bin_time / (binsize_ms * 1E-3))
        # binned = np.zeros(endbin + 1, dtype=np.int)
        # bin_times = np.linspace(0, end_bin_time, endbin+1)
        # for step, step_time in enumerate(bin_times):
        #     if step != 0:
        #         binned[step] = np.size(
        #             data[(step_time > data*1E-9)
        #                  * (data*1E-9 > step_time - binsize_ms*1E-3)])
        #     if step == 1:
        #         binned[step - 1] = binned[step]
        # binned_ps = binned * 1 / (binsize_ms * 1E-3)
