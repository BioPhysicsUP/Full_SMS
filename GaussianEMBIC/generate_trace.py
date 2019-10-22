""" Module that generates quasi-data of a trace to test change point and
clustering function.
"""

__docformat__ = 'NumPy'

import numpy as np
from typing import Tuple, List, Optional, Dict


def quasi_trace(max_int: Optional[float] = 500.0,
                min_int: Optional[float] = None,
                num_level_groups: Optional[int] = 5,
                trace_length: Optional[float] = 60.0,
                short_level: Optional[float] = 0.1,
                long_level: Optional[float] = 20.0,
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
    trace: Returns a tuple with the following structure indexes of the
    detected change points.
    """
    
    if min_int is None:
        min_int = 0.1 * max_int
    assert noise_type in {'none', 'gaussian', 'poisson'},\
        'Argument `noise_type` does not match one of the possible values.'
    
    # Dict to contain data to define trace
    trace = {'times': [], 'cp_inds': [], 'ints': [], 'dwell_times': []}
    
    # Defines the groups of intensities
    int_groups = np.linspace(min_int, max_int, num_level_groups)
    
    dwell_times = []
    tot_time = 0
    while tot_time < trace_length:  # tot_time is time of trace built in loop
        # Uniform random selection of time between short and long
        new_dt = np.random.uniform(short_level, long_level)
        if tot_time + new_dt > trace_length:  # if tot_time too large
            diff_time = trace_length - tot_time
            if diff_time >= short_level:
                new_dt = diff_time
            else:
                new_dt = trace_length - (tot_time - dwell_times.pop())
        tot_time += new_dt
        dwell_times.append(new_dt)
    
    tot_time = 0
    data = []
    for i, int_i in enumerate(int_groups):
        dwell = dwell_times[i]
        num_photons = int(round(int_i * dwell))
        if noise_type is 'none':
            new_data = np.linspace(tot_time, dwell + tot_time, num_photons)
        else:
            new_data = [tot_time]
            while new_data[-1] <= dwell:
                if noise_type is 'poisson':
                    # Poisson noise based on num_photons
                    noise_int = np.random.poisson(num_photons) / dwell
                else:  # Then Gaussian
                    # Gaussian noise based on int_i
                    noise_int = np.random.normal(int_i)
                new_data.append(new_data[-1] + (1 / noise_int))  # dt = 1/noise
            _ = new_data.pop()  # Remove last photon to prevent overlap with
                                # next level
            tot_time += dwell
        
        data.append(new_data)  # Add total time to shift
        # data up
        tot_time = dwell
    
    trace['ints'] = int_groups
    trace['dwell_times'] = dwell_times
    return trace


if __name__ == '__main__':
    data = quasi_trace()
