from __future__ import annotations
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from typing import TYPE_CHECKING
from my_logger import setup_logger

from enum import IntEnum, auto

if TYPE_CHECKING:
    from tree_model import DatasetTreeNode
    from multiprocessing import JoinableQueue
    from processes import ProgressTracker, ProcessProgressTask

logger = setup_logger(__name__)


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
    end_progress = pyqtSignal()
    status_message = pyqtSignal(str)

    add_datasetindex = pyqtSignal(object)
    add_particlenode = pyqtSignal(object, int)
    add_all_particlenodes = pyqtSignal(list)

    reset_tree = pyqtSignal()
    data_loaded = pyqtSignal()
    bin_size = pyqtSignal(int)

    add_irf = pyqtSignal(np.ndarray, np.ndarray, object)  # H5dataset)

    level_resolved = pyqtSignal()
    reset_gui = pyqtSignal()
    set_start = pyqtSignal(float)
    set_tmin = pyqtSignal(float)

    plot_trace = pyqtSignal(object, bool)
    plot_trace_lock = pyqtSignal(object, bool, object)
    plot_trace_export = pyqtSignal(object, bool, str)
    plot_trace_export_lock = pyqtSignal(object, bool, str, object)
    plot_levels = pyqtSignal(object, bool)
    plot_levels_lock = pyqtSignal(object, bool, object)
    plot_levels_export = pyqtSignal(object, bool, str, object)
    plot_levels_export_lock = pyqtSignal(object, bool, str, object)
    # plot_group_bounds = pyqtSignal(object, bool)
    plot_group_bounds_export = pyqtSignal(object, bool, str)
    plot_group_bounds_export_lock = pyqtSignal(object, bool, str, object)
    # plot_grouping_bic = pyqtSignal(object, bool)
    plot_grouping_bic_export = pyqtSignal(object, bool, str)
    plot_grouping_bic_export_lock = pyqtSignal(object, bool, str, object)
    plot_decay = pyqtSignal(int, object, bool, bool)
    plot_decay_lock = pyqtSignal(int, object, bool, bool, object)
    plot_decay_export = pyqtSignal(int, object, bool, bool, str)
    plot_decay_export_lock = pyqtSignal(int, object, bool, bool, str, object)
    # plot_convd= pyqtSignal(int, object, bool)
    plot_convd_export = pyqtSignal(int, object, bool, bool, str)
    plot_convd_export_lock = pyqtSignal(int, object, bool, bool, str, object)
    plot_decay_convd_export = pyqtSignal(object, str, bool)
    plot_decay_convd_export_lock = pyqtSignal(object, str, bool, object)
    show_residual_widget = pyqtSignal(bool)
    show_residual_widget_lock = pyqtSignal(bool, object)
    plot_residuals_export = pyqtSignal(int, object, bool, str)
    plot_residuals_export_lock = pyqtSignal(int, object, bool, str, object)
    plot_spectra_export = pyqtSignal(object, bool, str)
    plot_spectra_export_lock = pyqtSignal(object, bool, str, object)
    plot_raster_scan_export = pyqtSignal(object, object, bool, str)
    plot_raster_scan_export_lock = pyqtSignal(object, object, bool, str, object)


class WorkerSigPassType(IntEnum):
    resolve_finished = auto()
    fitting_finished = auto()
    grouping_finished = auto()
    openfile_finished = auto()
    error = auto()
    result = auto()

    progress = auto()
    auto_progress = auto()
    start_progress = auto()
    end_progress = auto()
    status_message = auto()

    add_datasetindex = auto()
    add_particlenode = auto()
    add_all_particlenodes = auto()

    reset_tree = auto()
    data_loaded = auto()
    bin_size = auto()

    add_irf = auto()

    level_resolved = auto()
    reset_gui = auto()
    set_start = auto()
    set_tmin = auto()


def worker_sig_pass(signals: WorkerSignals,
                    sig_type: WorkerSigPassType,
                    args=None):
    assert type(signals) is WorkerSignals, "signals is incorrect type."
    assert type(sig_type) is WorkerSigPassType, "sig_type is incorrect type."

    if type(args) is not tuple:
        args = (args,)

    if sig_type is WorkerSigPassType.resolve_finished:
        signals.resolve_finished.emit(*args)
    elif sig_type is WorkerSigPassType.fitting_finished:
        signals.fitting_finished.emit(*args)
    elif sig_type is WorkerSigPassType.grouping_finished:
        signals.grouping_finished.emit(*args)
    elif sig_type is WorkerSigPassType.openfile_finished:
        signals.openfile_finished.emit(*args)
    elif sig_type is WorkerSigPassType.error:
        signals.error.emit(*args)
    elif sig_type is WorkerSigPassType.result:
        signals.result.emit(*args)
    elif sig_type is WorkerSigPassType.progress:
        signals.progress.emit(*args)
    elif sig_type is WorkerSigPassType.auto_progress:
        signals.auto_progress.emit(*args)
    elif sig_type is WorkerSigPassType.start_progress:
        signals.start_progress.emit(*args)
    elif sig_type is WorkerSigPassType.status_message:
        signals.status_message.emit(*args)
    elif sig_type is WorkerSigPassType.add_datasetindex:
        signals.add_datasetindex.emit(*args)
    elif sig_type is WorkerSigPassType.add_particlenode:
        signals.add_particlenode.emit(*args)
    elif sig_type is WorkerSigPassType.add_all_particlenodes:
        signals.add_all_particlenodes.emit(*args)
    elif sig_type is WorkerSigPassType.reset_tree:
        signals.reset_tree.emit()
    elif sig_type is WorkerSigPassType.data_loaded:
        signals.data_loaded.emit()
    elif sig_type is WorkerSigPassType.bin_size:
        signals.bin_size.emit(*args)
    elif sig_type is WorkerSigPassType.add_irf:
        signals.add_irf.emit(*args)
    elif sig_type is WorkerSigPassType.level_resolved:
        signals.level_resolved.emit(*args)
    elif sig_type is WorkerSigPassType.reset_gui:
        signals.reset_gui.emit(*args)
    elif sig_type is WorkerSigPassType.set_start:
        signals.set_start.emit(*args)
    elif sig_type is WorkerSigPassType.set_tmin:
        signals.set_tmin.emit(*args)
    else:
        logger.error(f"Feedback return not configured for: {sig_type}")


class ProcessThreadSignals(QObject):
    """
    Defines the signals available from the running worker thread
    """

    finished = pyqtSignal(object)
    # single_result = pyqtSignal(object)
    results = pyqtSignal(object)
    start_progress = pyqtSignal(int)
    set_progress = pyqtSignal(int)
    step_progress = pyqtSignal(float)
    step_one_progress = pyqtSignal()
    add_progress = pyqtSignal(int)
    end_progress = pyqtSignal()
    status_update = pyqtSignal(str)
    error = pyqtSignal(Exception)
    passthrough = pyqtSignal(object)
