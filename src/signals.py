import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

import smsh5


class WorkerSignals(QObject):
    """ A QObject with attributes  of pyqtSignal's that can be used
    to communicate between worker threads and the main thread. """

    resolve_finished = pyqtSignal(str)
    fitting_finished = pyqtSignal(str)
    grouping_finished = pyqtSignal(str)
    openfile_finished = pyqtSignal(bool)
    error = pyqtSignal(Exception)
    result = pyqtSignal(object)

    progress = pyqtSignal()
    auto_progress = pyqtSignal(int, str)
    start_progress = pyqtSignal(int)
    status_message = pyqtSignal(str)

    add_datasetindex = pyqtSignal(object)
    add_particlenode = pyqtSignal(object, object, int)

    reset_tree = pyqtSignal()
    data_loaded = pyqtSignal()
    bin_size = pyqtSignal(int)

    add_irf = pyqtSignal(np.ndarray, np.ndarray, smsh5.H5dataset)

    level_resolved = pyqtSignal()
    reset_gui = pyqtSignal()
    set_start = pyqtSignal(float)
    set_tmin = pyqtSignal(float)


class ProcessThreadSignals(QObject):
    """
    Defines the signals available from the running worker thread
    """

    finished = pyqtSignal(object)
    result = pyqtSignal(object)
    progress = pyqtSignal(int, int)
    status_update = pyqtSignal(int, str)
    error = pyqtSignal(tuple)
    test = pyqtSignal(str)