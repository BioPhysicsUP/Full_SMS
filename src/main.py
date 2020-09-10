"""Module for analysis of SMS data from HDF5 files

Bertus van Heerden and Joshua Botha
University of Pretoria
2020
"""

__docformat__ = 'NumPy'

import csv
import os
import sys
import traceback
from platform import system

import numpy as np
import scipy
from PyQt5.QtCore import QObject, pyqtSignal, QAbstractItemModel, QModelIndex, \
    Qt, QThreadPool, QRunnable, pyqtSlot
from PyQt5.QtGui import QIcon, QResizeEvent, QPen, QColor
from PyQt5.QtWidgets import QMainWindow, QProgressBar, QFileDialog, QMessageBox, QInputDialog, \
    QApplication, QLineEdit, QComboBox, QDialog, QCheckBox, QStyleFactory
from PyQt5 import uic
import pyqtgraph as pg
from typing import Union

try:
    import pkg_resources.py2_warn
except ImportError:
    pass

import tcspcfit
import dbg
import smsh5
from generate_sums import CPSums
from smsh5 import start_at_value
from custom_dialogs import TimedMessageBox
from smsh5 import H5dataset, Particle
import resource_manager as rm

#  TODO: Needs to rather be reworked not to use recursion, but rather a loop of some sort
sys.setrecursionlimit(1000 * 10)

main_window_file = rm.path("mainwindow.ui", rm.RMType.UI)
UI_Main_Window, _ = uic.loadUiType(main_window_file)

fitting_dialog_file = rm.path("fitting_dialog.ui", rm.RMType.UI)
UI_Fitting_Dialog, _ = uic.loadUiType(fitting_dialog_file)


class WorkerSignals(QObject):
    """ A QObject with attributes  of pyqtSignal's that can be used
    to communicate between worker threads and the main thread. """

    resolve_finished = pyqtSignal(str)
    fitting_finished = pyqtSignal(str)
    grouping_finished = pyqtSignal(str)
    openfile_finished = pyqtSignal(bool)
    error = pyqtSignal(tuple)
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


class WorkerOpenFile(QRunnable):
    """ A QRunnable class to create a worker thread for opening h5 file. """

    # def __init__(self, fn, *args, **kwargs):
    def __init__(self, fname, irf=False, tmin=None):
        """
        Initiate Open File Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to open a h5 file
        and populate the tree in the mainwindow g

        Parameters
        ----------
        fname : str
            The name of the file.
        irf : bool
            Whether the thread is loading an IRF or not.
        """

        super(WorkerOpenFile, self).__init__()
        if irf:
            self.openfile_func = self.open_irf
        else:
            self.openfile_func = self.open_h5
        self.signals = WorkerSignals()
        self.fname = fname
        self.irf = irf
        self.tmin = tmin

    @pyqtSlot()
    def run(self) -> None:
        """ The code that will be run when the thread is started. """

        try:
            self.openfile_func(self.fname, self.tmin)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.openfile_finished.emit(self.irf)

    def open_h5(self, fname, tmin=None) -> None:
        """
        Read the selected h5 file and populates the tree on the gui with the file and the particles.

        Accepts a function that will be used to indicate the current progress.

        As this function is designed to be called from a thread other than the main one, no GUI code
        should be called here.

        Parameters
        ----------
        fname : str
            Path name to h5 file.
        """

        start_progress_sig = self.signals.start_progress
        progress_sig = self.signals.progress
        status_sig = self.signals.status_message
        add_dataset_sig = self.signals.add_datasetindex
        add_node_sig = self.signals.add_particlenode
        reset_tree_sig = self.signals.reset_tree
        data_loaded_sig = self.signals.data_loaded
        set_start_sig = self.signals.set_start
        set_tmin_sig = self.signals.set_tmin

        try:
            dataset = self.load_data(fname)

            datasetnode = DatasetTreeNode(fname[0][fname[0].rfind('/') + 1:-3], dataset, 'dataset')
            add_dataset_sig.emit(datasetnode)

            start_progress_sig.emit(dataset.numpart)
            status_sig.emit("Opening file: Adding particles...")
            for i, particle in enumerate(dataset.particles):
                particlenode = DatasetTreeNode(particle.name, particle, 'particle')
                add_node_sig.emit(particlenode, progress_sig, i)
                progress_sig.emit()
            reset_tree_sig.emit()

            starttimes = []
            tmins = []
            for particle in dataset.particles:
                # Find max, then search backward for first zero to find the best startpoint
                decay = particle.histogram.decay
                histmax_ind = np.argmax(decay)
                reverse = decay[:histmax_ind][::-1]
                zeros_rev = np.where(reverse == 0)[0]
                if len(zeros_rev) != 0:
                    length = 0
                    start_ind_rev = zeros_rev[0]
                    for i, val in enumerate(zeros_rev[:-1]):
                        if zeros_rev[i + 1] - val > 1:
                            length = 0
                            continue
                        length += 1
                        if length >= 10:
                            start_ind_rev = val
                            break
                    start_ind = histmax_ind - start_ind_rev
                    # starttime = particle.histogram.t[start_ind]
                    starttime = start_ind
                else:
                    starttime = 0
                starttimes.append(starttime)

                tmin = np.min(particle.histogram.microtimes)
                tmins.append(tmin)

            av_start = np.average(starttimes)
            set_start_sig.emit(av_start)

            global_tmin = np.min(tmins)
            for particle in dataset.particles:
                particle.tmin = global_tmin

            set_tmin_sig.emit(global_tmin)

            status_sig.emit("Done")
            data_loaded_sig.emit()
        except Exception as exc:
            raise RuntimeError("h5 data file was not loaded successfully.") from exc

    def open_irf(self, fname, tmin) -> None:
        """
        Read the selected h5 file and populates the tree on the gui with the file and the particles.

        Accepts a function that will be used to indicate the current progress.

        As this function is designed to be called from a thread other than the main one, no GUI code
        should be called here.

        Parameters
        ----------
        fname : str
            Path name to h5 file.
        """

        start_progress_sig = self.signals.start_progress
        status_sig = self.signals.status_message
        add_irf_sig = self.signals.add_irf

        try:
            dataset = self.load_data(fname)

            for particle in dataset.particles:
                particle.tmin = tmin
                # particle.tmin = np.min(particle.histogram.microtimes)
            irfhist = dataset.particles[0].histogram
            # irfhist.t -= irfhist.t.min()
            add_irf_sig.emit(irfhist.decay, irfhist.t, dataset)

            start_progress_sig.emit(dataset.numpart)
            status_sig.emit("Done")
        except Exception as exc:
            raise RuntimeError("h5 data file was not loaded successfully.") from exc

    def load_data(self, fname):

        auto_prog_sig = self.signals.auto_progress
        bin_size_sig = self.signals.bin_size
        progress_sig = self.signals.progress
        start_progress_sig = self.signals.start_progress

        status_sig = self.signals.status_message

        status_sig.emit("Opening file...")
        dataset = smsh5.H5dataset(fname[0], progress_sig, auto_prog_sig)
        bin_all(dataset, 100, start_progress_sig, progress_sig, status_sig, bin_size_sig)
        start_progress_sig.emit(dataset.numpart)
        status_sig.emit("Opening file: Building decay histograms...")
        dataset.makehistograms()
        return dataset


class WorkerBinAll(QRunnable):
    """ A QRunnable class to create a worker thread for binning all the data. """

    def __init__(self, dataset, binall_func, bin_size):
        """
        Initiate Open File Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to open a h5 file
        and populate the tree in the mainwindow g

        Parameters
        ----------
        fname : str
            The name of the file.
        binall_func : function
            Function to be called that will read the h5 file and populate the tree on the g
        """

        super(WorkerBinAll, self).__init__()
        self.dataset = dataset
        self.binall_func = binall_func
        self.signals = WorkerSignals()
        self.bin_size = bin_size

    @pyqtSlot()
    def run(self) -> None:
        """ The code that will be run when the thread is started. """

        try:
            self.binall_func(self.dataset, self.bin_size, self.signals.start_progress,
                             self.signals.progress, self.signals.status_message,
                             self.signals.bin_size)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            # self.signals.resolve_finished.emit(False)  ?????
            pass


def bin_all(dataset, bin_size, start_progress_sig, progress_sig, status_sig, bin_size_sig) -> None:
    """

    Parameters
    ----------
    bin_size
    dataset
    start_progress_sig
    progress_sig
    status_sig
    """

    start_progress_sig.emit(dataset.numpart)
    # if not self.data_loaded:
    #     part = "Opening file: "
    # else:
    #     part = ""
    # status_sig.emit(part + "Binning traces...")
    status_sig.emit("Binning traces...")
    dataset.bin_all_ints(bin_size, progress_sig)
    bin_size_sig.emit(bin_size)
    status_sig.emit("Done")


class WorkerResolveLevels(QRunnable):
    """ A QRunnable class to create a worker thread for resolving levels. """

    def __init__(self, resolve_levels_func, conf: Union[int, float], data: H5dataset, currentparticle: Particle,
                 mode: str,
                 resolve_selected=None,
                 end_time_s=None) -> None:
        """
        Initiate Resolve Levels Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to resolve a single,
        the selected, or all the particles'.

        Parameters
        ----------
        resolve_levels_func : function
            The function that will be called to perform the resolving of the levels.
        mode : {'current', 'selected', 'all'}
            Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
        resolve_selected : list[smsh5.Particle], optional
            The provided instances of the class Particle in smsh5 will be resolved.
        """

        super(WorkerResolveLevels, self).__init__()
        self.mode = mode
        self.signals = WorkerSignals()
        self.resolve_levels_func = resolve_levels_func
        self.resolve_selected = resolve_selected
        self.conf = conf
        self.data = data
        self.currentparticle = currentparticle
        self.end_time_s = end_time_s
        # print(self.currentparticle)

    @pyqtSlot()
    def run(self) -> None:
        """ The code that will be run when the thread is started. """

        try:
            self.resolve_levels_func(self.signals.start_progress, self.signals.progress,
                                     self.signals.status_message, self.signals.reset_gui, self.signals.level_resolved,
                                     self.conf, self.data, self.currentparticle,
                                     self.mode, self.resolve_selected, self.end_time_s)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.resolve_finished.emit(self.mode)


def resolve_levels(start_progress_sig: pyqtSignal, progress_sig: pyqtSignal,
                   status_sig: pyqtSignal, reset_gui_sig: pyqtSignal, level_resolved_sig: pyqtSignal,
                   conf: Union[int, float], data: H5dataset, currentparticle: Particle, mode: str,
                   resolve_selected=None,
                   end_time_s=None) -> None:
    """
    TODO: edit the docstring
    Resolves the levels in particles by finding the change points in the
    abstimes data of a Particle instance.

    Parameters
    ----------
    end_time_s
    currentparticle : Particle
    conf
    level_resolved_sig
    reset_gui_sig
    data : H5dataset
    start_progress_sig : pyqtSignal
        Used to call method to set up progress bar on G
    progress_sig : pyqtSignal
        Used to call method to increment progress bar on G
    status_sig : pyqtSignal
        Used to call method to show status bar message on G
    mode : {'current', 'selected', 'all'}
        Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
    resolve_selected : list[smsh5.Partilce]
        A list of Particle instances in smsh5, that isn't the current one, to be resolved.
    """

    # print(mode)
    assert mode in ['current', 'selected', 'all'], \
        "'resolve_all' and 'resolve_selected' can not both be given as parameters."

    if mode == 'current':  # Then resolve current
        currentparticle.cpts.run_cpa(confidence=conf / 100, run_levels=True, end_time_s=end_time_s)

    else:
        if mode == 'all':  # Then resolve all
            status_text = 'Resolving All Particle Levels...'
            parts = data.particles

        elif mode == 'selected':  # Then resolve selected
            assert resolve_selected is not None, \
                'No selected particles provided.'
            status_text = 'Resolving Selected Particle Levels...'
            parts = resolve_selected

        try:
            status_sig.emit(status_text)
            start_progress_sig.emit(len(parts))
            for num, part in enumerate(parts):
                dbg.p(f'Busy Resolving Particle {num + 1}')
                part.cpts.run_cpa(confidence=conf, run_levels=True, end_time_s=end_time_s)
                progress_sig.emit()
            status_sig.emit('Done')
        except Exception as exc:
            raise RuntimeError("Couldn't resolve levels.") from exc

    level_resolved_sig.emit()
    data.makehistograms(progress=False)
    reset_gui_sig.emit()


class WorkerFitLifetimes(QRunnable):
    """ A QRunnable class to create a worker thread for fitting lifetimes. """

    def __init__(self, fit_lifetimes_func, data, currentparticle, fitparam, mode: str, resolve_selected=None) -> None:
        """
        Initiate Resolve Levels Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to resolve a single,
        the selected, or all the particles'.

        Parameters
        ----------
        resolve_levels_func : function
            The function that will be called to perform the resolving of the levels.
        mode : {'current', 'selected', 'all'}
            Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
        resolve_selected : list[smsh5.Particle], optional
            The provided instances of the class Particle in smsh5 will be resolved.
        """

        super(WorkerFitLifetimes, self).__init__()
        self.mode = mode
        self.signals = WorkerSignals()
        self.fit_lifetimes_func = fit_lifetimes_func
        self.resolve_selected = resolve_selected
        self.data = data
        self.currentparticle = currentparticle
        self.fitparam = fitparam

    @pyqtSlot()
    def run(self) -> None:
        """ The code that will be run when the thread is started. """

        try:
            self.fit_lifetimes_func(self.signals.start_progress, self.signals.progress,
                                    self.signals.status_message, self.signals.reset_gui,
                                    self.data, self.currentparticle, self.fitparam,
                                    self.mode, self.resolve_selected)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.fitting_finished.emit(self.mode)


def fit_lifetimes(start_progress_sig: pyqtSignal, progress_sig: pyqtSignal,
                  status_sig: pyqtSignal, reset_gui_sig: pyqtSignal,
                  data, currentparticle, fitparam, mode: str,
                  resolve_selected=None) -> None:  # parallel: bool = False
    """
    TODO: edit the docstring
    Resolves the levels in particles by finding the change points in the
    abstimes data of a Particle instance.

    Parameters
    ----------
    start_progress_sig : pyqtSignal
        Used to call method to set up progress bar on G
    progress_sig : pyqtSignal
        Used to call method to increment progress bar on G
    status_sig : pyqtSignal
        Used to call method to show status bar message on G
    mode : {'current', 'selected', 'all'}
        Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
    resolve_selected : list[smsh5.Partilce]
        A list of Particle instances in smsh5, that isn't the current one, to be resolved.
    """

    print(mode)
    assert mode in ['current', 'selected', 'all'], \
        "'resolve_all' and 'resolve_selected' can not both be given as parameters."

    channelwidth = currentparticle.channelwidth
    if fitparam.start is None:
        start = None
    else:
        start = int(fitparam.start / channelwidth)
    if fitparam.end is None:
        end = None
    else:
        end = int(fitparam.end / channelwidth)

    if mode == 'current':  # Fit all levels in current particle
        status_sig.emit('Fitting Particle Levels...')
        start_progress_sig.emit(len(currentparticle.levels))

        for level in currentparticle.levels:
            try:
                if not level.histogram.fit(fitparam.numexp, fitparam.tau, fitparam.amp,
                                           fitparam.shift / channelwidth, fitparam.decaybg,
                                           fitparam.irfbg,
                                           start, end, fitparam.addopt,
                                           fitparam.irf, fitparam.shiftfix):
                    pass  # fit unsuccessful
                progress_sig.emit()
            except AttributeError:
                print("No decay")
        currentparticle.numexp = fitparam.numexp
        status_sig.emit("Ready...")

    elif mode == 'all':  # Fit all levels in all particles
        status_sig.emit('Fitting All Particle Levels...')
        start_progress_sig.emit(data.numpart)

        for particle in data.particles:
            fit_part_and_levels(channelwidth, end, fitparam, particle, progress_sig, start)
        status_sig.emit("Ready...")

    elif mode == 'selected':  # Fit all levels in selected particles
        assert resolve_selected is not None, \
            'No selected particles provided.'
        status_sig.emit('Resolving Selected Particle Levels...')
        start_progress_sig.emit(len(resolve_selected))
        for particle in resolve_selected:
            fit_part_and_levels(channelwidth, end, fitparam, particle, progress_sig, start)
        status_sig.emit('Ready...')

    reset_gui_sig.emit()


def fit_part_and_levels(channelwidth, end, fitparam, particle, progress_sig, start):
    if not particle.histogram.fit(fitparam.numexp, fitparam.tau, fitparam.amp,
                                  fitparam.shift / channelwidth, fitparam.decaybg,
                                  fitparam.irfbg,
                                  start, end, fitparam.addopt,
                                  fitparam.irf, fitparam.shiftfix):
        pass  # fit unsuccessful
    particle.numexp = fitparam.numexp
    progress_sig.emit()
    if not particle.has_levels:
        return
    for level in particle.levels:
        try:
            if not level.histogram.fit(fitparam.numexp, fitparam.tau, fitparam.amp,
                                       fitparam.shift / channelwidth, fitparam.decaybg,
                                       fitparam.irfbg,
                                       start, end, fitparam.addopt,
                                       fitparam.irf, fitparam.shiftfix):
                pass  # fit unsuccessful
        except AttributeError:
            print("No decay")


def group_levels(start_progress_sig: pyqtSignal,
                 progress_sig: pyqtSignal,
                 status_sig: pyqtSignal,
                 reset_gui_sig: pyqtSignal,
                 data: H5dataset,
                 mode: str,
                 currentparticle: Particle = None,
                 group_selected=None) -> None:
    """
    TODO: edit the docstring
    Resolves the levels in particles by finding the change points in the
    abstimes data of a Particle instance.

    Parameters
    ----------
    currentparticle : Particle
    conf
    level_resolved_sig
    reset_gui_sig
    data : H5dataset
    start_progress_sig : pyqtSignal
        Used to call method to set up progress bar on G
    progress_sig : pyqtSignal
        Used to call method to increment progress bar on G
    status_sig : pyqtSignal
        Used to call method to show status bar message on G
    mode : {'current', 'selected', 'all'}
        Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
    resolve_selected : list[smsh5.Partilce]
        A list of Particle instances in smsh5, that isn't the current one, to be resolved.
    """

    # print(mode)
    assert mode in ['current', 'selected', 'all'], \
        "'resolve_all' and 'resolve_selected' can not both be given as parameters."

    if mode == 'current':
        status_text = 'Grouping Current Particle Levels...'
        parts = [currentparticle]
    elif mode == 'all':  # Then resolve all
        status_text = 'Grouping All Particle Levels...'
        parts = data.particles

    elif mode == 'selected':  # Then resolve selected
        assert group_selected is not None, \
            'No selected particles provided.'
        status_text = 'Grouping Selected Particle Levels...'
        parts = group_selected

    try:
        status_sig.emit(status_text)
        start_progress_sig.emit(len(parts))
        for num, part in enumerate(parts):
            dbg.p(f'Busy Grouping Particle {num + 1}')
            part.ahca.run_grouping()
            progress_sig.emit()
        status_sig.emit('Done')
    except Exception as exc:
        raise RuntimeError("Couldn't group levels.") from exc

    # grou.emit()
    # data.makehistograms(progress=False)
    # reset_gui_sig.emit()


class WorkerGrouping(QRunnable):

    def __init__(self,
                 data: H5dataset,
                 grouping_func,
                 mode: str,
                 currentparticle: Particle = None,
                 group_selected=None) -> None:
        """
        Initiate Resolve Levels Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to resolve a single,
        the selected, or all the particles'.

        Parameters
        ----------
        resolve_levels_func : function
            The function that will be called to perform the resolving of the levels.
        mode : {'current', 'selected', 'all'}
            Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
        resolve_selected : list[smsh5.Particle], optional
            The provided instances of the class Particle in smsh5 will be resolved.
        """

        super(WorkerGrouping, self).__init__()
        self.mode = mode
        self.signals = WorkerSignals()
        self.grouping_func = grouping_func
        self.group_selected = group_selected
        self.data = data
        self.currentparticle = currentparticle
        # self.fitparam = fitparam

    @pyqtSlot()
    def run(self) -> None:
        """ The code that will be run when the thread is started. """

        try:
            self.grouping_func(start_progress_sig=self.signals.start_progress,
                               progress_sig=self.signals.progress,
                               status_sig=self.signals.status_message,
                               reset_gui_sig=self.signals.reset_gui,
                               data=self.data,
                               mode=self.mode,
                               currentparticle=self.currentparticle,
                               group_selected=self.group_selected)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.grouping_finished.emit(self.mode)
            pass


class DatasetTreeNode(object):
    """ Contains the files with their respective particles. Also seems to house the actual data objects. """

    def __init__(self, name, dataobj, datatype, checked=False) -> None:
        """
        TODO Docstring

        Parameters
        ----------
        name
        dataobj
        datatype
        """

        self._data = name
        if type(name) == tuple:
            self._data = list(name)
        if type(name) in (str, bytes) or not hasattr(name, '__getitem__'):
            self._data = [name]

        self._columncount = len(self._data)
        self._children = []
        self._parent = None
        self._row = 0

        if datatype == 'dataset':
            pass

        elif datatype == 'particle':
            pass

        self.dataobj = dataobj
        self.setChecked(checked)

    def checked(self):
        """
        Appears to be used internally.

        Returns
        -------
        Returns check status.
        """
        return self._checked

    def setChecked(self, checked=True):
        self._checked = bool(checked)

    def data(self, in_column):
        """ TODO: Docstring """

        if in_column >= 0 and in_column < len(self._data):
            return self._data[in_column]

    def columnCount(self):
        """ TODO: Docstring """

        return self._columncount

    def childCount(self):
        """ TODO: Docstring """

        return len(self._children)

    def child(self, in_row):
        """ TODO: Docstring """

        if in_row >= 0 and in_row < self.childCount():
            return self._children[in_row]

    def parent(self):
        """ TODO: Docstring """

        return self._parent

    def row(self):
        """ TODO: Docstring """

        return self._row

    def addChild(self, in_child):
        """
        TODO: Docstring

        Parameters
        ----------
        in_child
        """

        in_child._parent = self
        in_child._row = len(self._children)
        self._children.append(in_child)
        self._columncount = max(in_child.columnCount(), self._columncount)

        return in_child._row


class DatasetTreeModel(QAbstractItemModel):
    """ TODO: Docstring """

    def __init__(self):
        """ TODO: Docstring """

        QAbstractItemModel.__init__(self)
        self._root = DatasetTreeNode(None, None, None)
        # for node in in_nodes:
        #     self._root.addChild(node)

    def flags(self, index):
        # return self.flags(index) | Qt.ItemIsUserCheckable
        flags = Qt.ItemIsEnabled | Qt.ItemIsTristate | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        return flags

    def rowCount(self, in_index):
        """
        TODO: Docstring

        Parameters
        ----------
        in_index
        """

        if in_index.isValid():
            return in_index.internalPointer().childCount()
        return self._root.childCount()

    def addChild(self, in_node, in_parent=None, progress_sig=None):
        """
        TODO: Docstring

        Parameters
        ----------
        in_node
        in_parent
        progress_sig
        """

        self.layoutAboutToBeChanged.emit()
        if not in_parent or not in_parent.isValid():
            parent = self._root
        else:
            parent = in_parent.internalPointer()
        row = parent.addChild(in_node)
        self.layoutChanged.emit()
        self.modelReset.emit()
        if progress_sig is not None:
            progress_sig.emit()  # Increment progress bar on MainWindow GUI
        return self.index(row, 0)

    def index(self, in_row, in_column, in_parent=None):
        """
        TODO: Docstring

        Parameters
        ----------
        in_row
        in_column
        in_parent
        """

        if not in_parent or not in_parent.isValid():
            parent = self._root
        else:
            parent = in_parent.internalPointer()

        # if not QAbstractItemModel.hasIndex(self, in_row, in_column, in_parent):
        #     return QModelIndex()

        child = parent.child(in_row)
        if child:
            return QAbstractItemModel.createIndex(self, in_row, in_column, child)
        else:
            return QModelIndex()

    def parent(self, in_index):
        """
        TODO: Docstring

        Parameters
        ----------
        in_index
        """

        if in_index.isValid():
            p = in_index.internalPointer().parent()
            if p:
                return QAbstractItemModel.createIndex(self, p.row(), 0, p)
        return QModelIndex()

    def columnCount(self, in_index):
        """
        TODO: Docstring

        Parameters
        ----------
        in_index
        """

        if in_index.isValid():
            return in_index.internalPointer().columnCount()
        return self._root.columnCount()

    def get_particle(self, ind: int) -> smsh5.Particle:
        """
        Returns the smsh5.Particle object of the ind'th tree particle.

        Parameters
        ----------
        ind: int
            The index of the particle.

        Returns
        -------
        smsh5.Particle
        """
        return self.data(ind, Qt.UserRole)

    def data(self, in_index, role):
        """
        TODO: Docstring

        Parameters
        ----------
        in_index
        role
        """

        if not in_index.isValid():
            return None
        node = in_index.internalPointer()
        if role == Qt.DisplayRole:
            return node.data(in_index.column())
        if role == Qt.UserRole:
            return node.dataobj
        if role == Qt.CheckStateRole:
            if node.checked():
                return Qt.Checked
            return Qt.Unchecked
        return None

    def setData(self, index, value, role=Qt.EditRole):

        if index.isValid():
            if role == Qt.CheckStateRole:
                node = index.internalPointer()
                node.setChecked(not node.checked())
                return True
        return False


# noinspection PyUnresolvedReferences
class MainWindow(QMainWindow, UI_Main_Window):
    """
    Class for Full SMS application that returns QMainWindow object.

    This class uses a *.ui converted to a *.py script to generate g Be
    sure to run convert_py after having made changes to mainwindow.
    """

    def __init__(self):
        """Initialise MainWindow object.

        Creates and populates QMainWindow object as described by mainwindow.py
        as well as creates MplWidget
        """

        self.threadpool = QThreadPool()
        dbg.p("Multi-threading with maximum %d threads" % self.threadpool.maxThreadCount(), "MainWindow")

        self.confidence_index = {
            0: 99,
            1: 95,
            2: 90,
            3: 69}

        if system() == "Windows":
            dbg.p("System -> Windows", "MainWindow")
        elif system() == "Darwin":
            dbg.p("System -> Unix/Linus", "MainWindow")
        else:
            dbg.p("System -> Other", "MainWindow")

        QMainWindow.__init__(self)
        UI_Main_Window.__init__(self)
        self.setupUi(self)

        self.setWindowIcon(QIcon(rm.path('Full-SMS.ico', rm.RMType.Icons)))

        self.tabWidget.setCurrentIndex(0)

        self.setWindowTitle("Full SMS")

        self.pgIntensity.getPlotItem().getAxis('left').setLabel('Intensity', 'counts/100ms')
        self.pgIntensity.getPlotItem().getAxis('bottom').setLabel('Time', 's')
        self.pgIntensity.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)
        self.pgIntensity.getPlotItem().setContentsMargins(5, 5, 5, 5)

        self.pgLifetime_Int.getPlotItem().getAxis('left').setLabel('Intensity', 'counts/100ms')
        self.pgLifetime_Int.getPlotItem().getAxis('bottom').setLabel('Time', 's')
        # self.pgLifetime_Int.getPlotItem().getViewBox()\
        #     .setYLink(self.pgIntensity.getPlotItem().getAxis('left').getViewBox())
        # self.pgLifetime_Int.getPlotItem().getViewBox()\
        #     .setXLink(self.pgIntensity.getPlotItem().getAxis('bottom').getViewBox())
        self.pgLifetime_Int.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)
        self.pgLifetime_Int.getPlotItem().setContentsMargins(5, 5, 5, 5)

        self.pgLifetime.getPlotItem().getAxis('left').setLabel('Num. of occur.', 'counts/bin')
        self.pgLifetime.getPlotItem().getAxis('bottom').setLabel('Decay time', 'ns')
        self.pgLifetime.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)
        self.pgLifetime_Int.getPlotItem().setContentsMargins(5, 5, 5, 5)

        self.pgGroups_Int.getPlotItem().getAxis('left').setLabel('Intensity', 'counts/100ms')
        self.pgGroups_Int.getPlotItem().getAxis('bottom').setLabel('Time', 's')
        self.pgGroups_Int.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)
        self.pgGroups_Int.getPlotItem().setContentsMargins(5, 5, 1, 5)

        # self.pgGroups_Hist.getPlotItem().getAxis('left').setLabel('Time', 's')
        self.pgGroups_Hist.getPlotItem().getAxis('bottom').setLabel('Relative Frequency')
        self.pgGroups_Hist.getPlotItem().setYLink(self.pgGroups_Int.getPlotItem().getViewBox())
        self.pgGroups_Hist.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)
        self.pgGroups_Hist.getPlotItem().setContentsMargins(1, 5, 5, 25)
        self.pgGroups_Hist.getPlotItem().getAxis('bottom').setStyle(showValues=False)
        self.pgGroups_Hist.getPlotItem().getAxis('left').setStyle(showValues=False)
        self.pgGroups_Hist.getPlotItem().vb.setLimits(xMin=0, xMax=1)

        self.pgBIC.getPlotItem().getAxis('left').setLabel('BIC')
        self.pgBIC.getPlotItem().getAxis('bottom').setLabel('Number of State')
        self.pgBIC.getPlotItem().getViewBox().setLimits(xMin=0)
        self.pgBIC.getPlotItem().setContentsMargins(5, 5, 5, 5)

        self.pgSpectra.getPlotItem().getAxis('left').setLabel('X Range', 'um')
        self.pgSpectra.getPlotItem().getAxis('bottom').setLabel('Y Range', '<span>&#181;</span>m')
        self.pgSpectra.getPlotItem().getViewBox().setAspectLocked(lock=True, ratio=1)
        self.pgSpectra.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)
        self.pgSpectra.getPlotItem().setContentsMargins(5, 5, 5, 5)

        self.int_controller = IntController(self)
        self.lifetime_controller = LifetimeController(self)
        self.spectra_controller = SpectraController(self)
        self.grouping_controller = GroupingController(self)

        plots = [self.pgIntensity, self.pgLifetime_Int, self.pgLifetime,
                 self.pgGroups_Int, self.pgGroups_Hist, self.pgBIC, self.pgSpectra, self.lifetime_controller.fitparamdialog.pgFitParam]
        axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)
        for plot in plots:
            # Set background and axis line width
            plot.setBackground(background=None)
            plot_item = plot.getPlotItem()
            plot_item.getAxis('left').setPen(axis_line_pen)
            plot_item.getAxis('bottom').setPen(axis_line_pen)

            # Set axis label bold and size
            font = plot_item.getAxis('left').label.font()
            font.setBold(True)
            if plot == self.pgLifetime_Int:
                font.setPointSize(8)
            elif plot == self.pgGroups_Int or plot == self.pgGroups_Hist:
                font.setPointSize(10)
            else:
                font.setPointSize(12)
            plot_item.getAxis('left').label.setFont(font)
            plot_item.getAxis('bottom').label.setFont(font)

            plot.setAntialiasing(True)

        # Connect all GUI buttons with outside class functions
        self.btnApplyBin.clicked.connect(self.int_controller.gui_apply_bin)
        self.btnApplyBinAll.clicked.connect(self.int_controller.gui_apply_bin_all)
        self.btnResolve.clicked.connect(self.int_controller.gui_resolve)
        self.btnResolve_Selected.clicked.connect(self.int_controller.gui_resolve_selected)
        self.btnResolveAll.clicked.connect(self.int_controller.gui_resolve_all)
        self.actionTime_Resolve_Current.triggered.connect(self.int_controller.time_resolve_current)
        self.actionTime_Resolve_Selected.triggered.connect(self.int_controller.time_resolve_selected)
        self.actionTime_Resolve_All.triggered.connect(self.int_controller.time_resolve_all)

        self.btnPrevLevel.clicked.connect(self.lifetime_controller.gui_prev_lev)
        self.btnNextLevel.clicked.connect(self.lifetime_controller.gui_next_lev)
        self.btnWholeTrace.clicked.connect(self.lifetime_controller.gui_whole_trace)
        self.btnLoadIRF.clicked.connect(self.lifetime_controller.gui_load_irf)
        self.btnFitParameters.clicked.connect(self.lifetime_controller.gui_fit_param)
        self.btnFitCurrent.clicked.connect(self.lifetime_controller.gui_fit_current)
        self.btnFit.clicked.connect(self.lifetime_controller.gui_fit_levels)
        self.btnFitSelected.clicked.connect(self.lifetime_controller.gui_fit_selected)
        self.btnFitAll.clicked.connect(self.lifetime_controller.gui_fit_all)
        self.btnGroupCurrent.clicked.connect(self.grouping_controller.gui_group_current)
        self.btnGroupSelected.clicked.connect(self.grouping_controller.gui_group_selected)
        self.btnGroupAll.clicked.connect(self.grouping_controller.gui_group_all)

        self.btnSubBackground.clicked.connect(self.spectra_controller.gui_sub_bkg)

        self.actionOpen_h5.triggered.connect(self.act_open_h5)
        self.actionOpen_pt3.triggered.connect(self.act_open_pt3)
        self.actionSave_Selected.triggered.connect(self.act_save_selected)
        self.actionTrim_Dead_Traces.triggered.connect(self.act_trim)
        self.actionSwitch_All.triggered.connect(self.act_switch_all)
        self.actionSwitch_Selected.triggered.connect(self.act_switch_selected)
        self.actionSet_Startpoint.triggered.connect(self.act_set_startpoint)
        self.btnEx_Current.clicked.connect(self.gui_export_current)
        self.btnEx_Selected.clicked.connect(self.gui_export_selected)
        self.btnEx_All.clicked.connect(self.gui_export_all)

        # Create and connect model for dataset tree
        self.treemodel = DatasetTreeModel()
        self.treeViewParticles.setModel(self.treemodel)
        # Connect the tree selection to data display
        self.treeViewParticles.selectionModel().currentChanged.connect(self.display_data)

        self.part_nodes = list()
        self.part_index = list()

        self.tauparam = None
        self.ampparam = None
        self.shift = None
        self.decaybg = None
        self.irfbg = None
        self.start = None
        self.end = None
        self.addopt = None

        self.statusBar().showMessage('Ready...')
        self.progress = QProgressBar(self)
        self.progress.setMinimumSize(170, 19)
        self.progress.setVisible(False)
        self.progress.setValue(0)  # Range of values is from 0 to 100
        self.statusBar().addPermanentWidget(self.progress)
        self.data_loaded = False
        self.level_resolved = False
        self.irf_loaded = False
        self.has_spectra = False

        self._current_level = None

        self.tabWidget.currentChanged.connect(self.tab_change)

        self.reset_gui()
        self.repaint()

    """#######################################
    ######## GUI Housekeeping Methods ########
    #######################################"""

    def after_show(self):
        self.pgSpectra.resize(self.tabSpectra.size().height(),
                              self.tabSpectra.size().height() - self.btnSubBackground.size().height() - 40)

    def resizeEvent(self, a0: QResizeEvent):
        if self.tabSpectra.size().height() <= self.tabSpectra.size().width():
            self.pgSpectra.resize(self.tabSpectra.size().height(),
                                  self.tabSpectra.size().height() - self.btnSubBackground.size().height() - 40)
        else:
            self.pgSpectra.resize(self.tabSpectra.size().width(),
                                  self.tabSpectra.size().width() - 40)

    def check_all_sums(self) -> None:
        """
        Check if the all_sums.pickle file exists, and if it doesn't creates it
        """
        if (not os.path.exists(rm.path('all_sums.pickle'))) and \
                (not os.path.isfile(rm.path('all_sums.pickle'))):
            self.status_message('Calculating change point sums, this may take several minutes.')
            create_all_sums = CPSums(only_pickle=True, n_min=10, n_max=1000)
            del create_all_sums
            self.status_message('Ready...')

    def gui_export_current(self):

        self.export(mode='current')

    def gui_export_selected(self):

        self.export(mode='selected')

    def gui_export_all(self):

        self.export(mode='all')

    def act_open_h5(self):
        """ Allows the user to point to a h5 file and then starts a thread that reads and loads the file. """

        fname = QFileDialog.getOpenFileName(self, 'Open HDF5 file', '', "HDF5 files (*.h5)")
        if fname != ('', ''):  # fname will equal ('', '') if the user canceled.
            of_worker = WorkerOpenFile(fname)
            of_worker.signals.openfile_finished.connect(self.open_file_thread_complete)
            of_worker.signals.start_progress.connect(self.start_progress)
            of_worker.signals.progress.connect(self.update_progress)
            of_worker.signals.auto_progress.connect(self.update_progress)
            of_worker.signals.start_progress.connect(self.start_progress)
            of_worker.signals.status_message.connect(self.status_message)
            of_worker.signals.add_datasetindex.connect(self.add_dataset)
            of_worker.signals.add_particlenode.connect(self.add_node)
            of_worker.signals.reset_tree.connect(lambda: self.treemodel.modelReset.emit())
            of_worker.signals.data_loaded.connect(self.set_data_loaded)
            of_worker.signals.bin_size.connect(self.spbBinSize.setValue)
            of_worker.signals.set_start.connect(self.set_startpoint)
            of_worker.signals.set_tmin.connect(self.lifetime_controller.set_tmin)

            self.threadpool.start(of_worker)

    def act_open_pt3(self):
        """ Allows a user to load a group of .pt3 files that are in a folder and loads them. NOT YET IMPLEMENTED. """

        print("act_open_pt3")

    def act_save_selected(self):
        """" Saves selected particles into a new HDF5 file."""

        selected_nums = self.get_checked_nums()

        if not len(selected_nums):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle('Save Error')
            msg.setText('No particles selected.')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            return

        fname, _ = QFileDialog.getSaveFileName(self, 'New or Existing HDF5 file', '',
                                               'HDF5 files (*.h5)', options=QFileDialog.DontConfirmOverwrite)
        if os.path.exists(fname[0]):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle('Add To Existing File')
            msg.setText('Do you want to add selected particles to existing file?')
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            if msg.exec() == QMessageBox.Cancel:
                return

        if self.tree2dataset().name == fname:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle('Save Error')
            msg.setText('Can''t add particles to currently opened file.')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            return

        self.tree2dataset().save_particles(fname, selected_nums)

    def act_trim(self):
        """ Used to trim the 'dead' part of a trace as defined by two parameters. """

        print("act_trim")

    def act_switch_all(self):

        self.switching_frequency(all_selected='all')

    def act_switch_selected(self):

        self.switching_frequency(all_selected='selected')

    def act_set_startpoint(self):

        start, ok = QInputDialog.getInt(self, 'Input Dialog', 'Enter startpoint:')
        self.set_startpoint(start)

    def set_startpoint(self, start=None):
        if start is None:
            start = self.lifetime_controller.startpoint
        try:
            self.tree2dataset().makehistograms(remove_zeros=False, startpoint=start, channel=True)
        except Exception as exc:
            print(exc)
        if self.lifetime_controller.irf_loaded:
            self.lifetime_controller.change_irf_start(start)
        if self.lifetime_controller.startpoint is None:
            self.lifetime_controller.startpoint = start
        self.display_data()
        dbg.p('Set startpoint', 'MainWindow')

    """#######################################
    ############ Internal Methods ############
    #######################################"""

    def add_dataset(self, datasetnode):

        self.datasetindex = self.treemodel.addChild(datasetnode)

    def add_node(self, particlenode, progress_sig, i):

        index = self.treemodel.addChild(particlenode, self.datasetindex, progress_sig)
        if i == 1:
            self.treeViewParticles.expand(self.datasetindex)
            self.treeViewParticles.setCurrentIndex(index)

        self.part_nodes.append(particlenode)
        self.part_index.append(index)

    def tab_change(self, active_tab_index: int):
        if self.data_loaded and hasattr(self, 'currentparticle'):
            if self.tabWidget.currentIndex() in [0, 1, 2, 3]:
                self.display_data()

    def display_data(self, current=None, prev=None) -> None:
        """ Displays the intensity trace and the histogram of the current particle.

            Directly called by the tree signal currentChanged, thus the two arguments.

        Parameters
        ----------
        current : QtCore.QModelIndex
            The index of the current selected particle as defined by QtCore.QModelIndex.
        prev : QtCore.QModelIndex
            The index of the previous selected particle as defined by QtCore.QModelIndex.
        """

        # self.current_level = None

        self.current_ind = current
        self.pre_ind = prev
        if current is not None:
            if hasattr(self, 'currentparticle'):
                self.currentparticle = self.treemodel.get_particle(current)
            self.current_level = None  # Reset current level when particle changes.
        if hasattr(self, 'currentparticle') and type(self.currentparticle) is smsh5.Particle:
            cur_tab_name = self.tabWidget.currentWidget().objectName()

            if cur_tab_name == 'tabIntensity' or cur_tab_name == 'tabGrouping':
                self.int_controller.set_bin(self.currentparticle.bin_size)
                self.int_controller.plot_trace()

            if cur_tab_name == 'tabGrouping':
                self.grouping_controller.plot_hist()
                if self.currentparticle.has_groups:
                    self.grouping_controller.plot_groups()

            if self.currentparticle.has_levels:
                self.int_controller.plot_levels()
                self.btnGroupCurrent.setEnabled(True)
                self.btnGroupSelected.setEnabled(True)
                self.btnGroupAll.setEnabled(True)
            else:
                self.btnGroupCurrent.setEnabled(False)
                self.btnGroupSelected.setEnabled(False)
                self.btnGroupAll.setEnabled(False)

            if cur_tab_name == 'tabLifetime':
                self.lifetime_controller.plot_decay(remove_empty=False)
                self.lifetime_controller.plot_convd()
                self.lifetime_controller.update_results()
            dbg.p('Current data displayed', 'MainWindow')

    def status_message(self, message: str) -> None:
        """
        Updates the status bar with the provided message argument.

        Parameters
        ----------
        message : str
            The message that is to be displayed in the status bar.
        """

        if message != '':
            self.statusBar().showMessage(message)
            # self.statusBar().show()
        else:
            self.statusBar().clearMessage()

    def start_progress(self, max_num: int) -> None:
        """
        Sets the maximum value of the progress bar before use.

        reset parameter can be optionally set to False to prevent the setting of the progress bar value to 0.

        Parameters
        ----------
        max_num : int
            The number of iterations or steps that the complete process is made up of.
        """

        assert type(max_num) is int, "MainWindow:\tThe type of the 'max_num' parameter is not int."
        self.progress.setMaximum(max_num)
        self.progress.setValue(0)
        self.progress.setVisible(True)

    def update_progress(self, value: int = None, text: str = None) -> None:
        """ Used to update the progress bar by an increment of one. If at maximum sets progress bars visibility to False """

        # print("Update progress")
        if self.progress.isVisible():
            if value is not None:
                self.progress.setValue(value)
                if value == self.progress.maximum():
                    self.progress.setVisible(False)
            else:
                current_value = self.progress.value()
                self.progress.setValue(current_value + 1)
                if current_value + 1 == self.progress.maximum():
                    self.progress.setVisible(False)

        elif value is not None:
            if text is None:
                text = 'Progress.'
            self.status_message(text)
            self.progress.setMaximum(100)
            self.progress.setValue(value)
            self.progress.setVisible(True)

        self.progress.repaint()
        self.statusBar().repaint()
        self.repaint()

    def tree2particle(self, identifier):
        """ Returns the particle dataset for the identifier given.
        The identifier could be the number of the particle of the the datasetnode value.

        Parameters
        ----------
        identifier
            The integer number or a datasetnode object of the particle in question.
        Returns
        -------

        """
        if type(identifier) is int:
            return self.part_nodes[identifier].dataobj
        if type(identifier) is DatasetTreeNode:
            return identifier.dataobj

    def tree2dataset(self) -> smsh5.H5dataset:
        """ Returns the H5dataset object of the file loaded.

        Returns
        -------
        smsh5.H5dataset
        """
        return self.treemodel.data(self.treemodel.index(0, 0), Qt.UserRole)

    def set_data_loaded(self):
        self.data_loaded = True

    def open_file_thread_complete(self, irf=False) -> None:
        """ Is called as soon as all of the threads have finished. """

        if self.data_loaded and not irf:
            self.currentparticle = self.tree2particle(0)
            self.treeViewParticles.expandAll()
            self.treeViewParticles.setCurrentIndex(self.part_index[0])
            self.display_data(self.part_index[1])

            msgbx = TimedMessageBox(30)
            msgbx.setIcon(QMessageBox.Question)
            msgbx.setText("Would you like to resolve levels now?")
            msgbx.set_timeout_text(message_pretime="(Resolving levels in ", message_posttime=" seconds)")
            msgbx.setWindowTitle("Resolve Levels?")
            msgbx.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
            msgbx.setDefaultButton(QMessageBox.Yes)
            msgbx_result, timed_out = msgbx.exec()
            if msgbx_result == QMessageBox.Yes:
                confidences = ("0.99", "0.95", "0.90", "0.69")
                if timed_out:
                    index = 0
                else:
                    item, ok = QInputDialog.getItem(self, "Choose Confidence",
                                                    "Select confidence interval to use.", confidences, 0, False)
                    if ok:
                        index = list(self.confidence_index.values()).index(int(float(item) * 100))
                self.cmbConfIndex.setCurrentIndex(index)
                self.int_controller.start_resolve_thread('all')
        self.reset_gui()
        self.gbxExport_Int.setEnabled(True)
        self.chbEx_Trace.setEnabled(True)
        dbg.p('File opened', 'MainWindow')

    def start_binall_thread(self, bin_size) -> None:
        """

        Parameters
        ----------
        bin_size
        """

        dataset = self.tree2dataset()

        binall_thread = WorkerBinAll(dataset, bin_all, bin_size)
        binall_thread.signals.resolve_finished.connect(self.binall_thread_complete)
        binall_thread.signals.start_progress.connect(self.start_progress)
        binall_thread.signals.progress.connect(self.update_progress)
        binall_thread.signals.status_message.connect(self.status_message)

        self.threadpool.start(binall_thread)

    def binall_thread_complete(self):

        self.status_message('Done')
        self.plot_trace()
        dbg.p('Binnig all levels complete', 'MainWindow')

    def start_resolve_thread(self, mode: str = 'current', thread_finished=None) -> None:
        """
        Creates a worker to resolve levels.

        Depending on the ``current_selected_all`` parameter the worker will be
        given the necessary parameter to fit the current, selected or all particles.

        Parameters
        ----------
        thread_finished
        mode : {'current', 'selected', 'all'}
            Possible values are 'current' (default), 'selected', and 'all'.
        """

        if thread_finished is None:
            if self.data_loaded:
                thread_finished = self.resolve_thread_complete
            else:
                thread_finished = self.open_file_thread_complete

        selected = None
        if mode == 'selected':
            selected = self.get_checked_particles()

        resolve_thread = WorkerResolveLevels(resolve_levels,
                                             conf=self.confidence_index[self.cmbConfIndex.currentIndex()],
                                             data=self.tree2dataset(),
                                             currentparticle=self.currentparticle,
                                             mode=mode,
                                             resolve_selected=selected)
        resolve_thread.signals.resolve_finished.connect(self.resolve_thread_complete)
        resolve_thread.signals.start_progress.connect(self.start_progress)
        resolve_thread.signals.progress.connect(self.update_progress)
        resolve_thread.signals.status_message.connect(self.status_message)
        resolve_thread.signals.reset_gconnect(self.reset_gui)

        self.threadpool.start(resolve_thread)

    # @dbg.profile
    # TODO: remove this method as it has been replaced by function
    def resolve_levels(self, start_progress_sig: pyqtSignal,
                       progress_sig: pyqtSignal, status_sig: pyqtSignal,
                       mode: str,
                       resolve_selected=None) -> None:  # parallel: bool = False
        """
        Resolves the levels in particles by finding the change points in the
        abstimes data of a Particle instance.

        Parameters
        ----------
        start_progress_sig : pyqtSignal
            Used to call method to set up progress bar on G
        progress_sig : pyqtSignal
            Used to call method to increment progress bar on G
        status_sig : pyqtSignal
            Used to call method to show status bar message on G
        mode : {'current', 'selected', 'all'}
            Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
        resolve_selected : list[smsh5.Partilce]
            A list of Particle instances in smsh5, that isn't the current one, to be resolved.
        """

        assert mode in ['current', 'selected', 'all'], \
            "'resolve_all' and 'resolve_selected' can not both be given as parameters."

        if mode == 'current':  # Then resolve current
            _, conf = self.get_gui_confidence()
            self.currentparticle.cpts.run_cpa(confidence=conf, run_levels=True)

        elif mode == 'all':  # Then resolve all
            data = self.tree2dataset()
            _, conf = self.get_gui_confidence()
            try:
                status_sig.emit('Resolving All Particle Levels...')
                start_progress_sig.emit(data.numpart)
                # if parallel:
                #     self.conf_parallel = conf
                #     Parallel(n_jobs=-2, backend='threading')(
                #         delayed(self.run_parallel_cpa)
                #         (self.tree2particle(num)) for num in range(data.numpart)
                #     )
                #     del self.conf_parallel
                # else:
                for num in range(data.numpart):
                    data.particles[num].cpts.run_cpa(confidence=conf, run_levels=True)
                    progress_sig.emit()
                status_sig.emit('Done')
            except Exception as exc:
                raise RuntimeError("Couldn't resolve levels.") from exc

        elif mode == 'selected':  # Then resolve selected
            assert resolve_selected is not None, \
                'No selected particles provided.'
            try:
                _, conf = self.get_gui_confidence()
                status_sig.emit('Resolving Selected Particle Levels...')
                start_progress_sig.emit(len(resolve_selected))
                for particle in resolve_selected:
                    particle.cpts.run_cpa(confidence=conf, run_levels=True)
                    progress_sig.emit()
                status_sig.emit('Done')
            except Exception as exc:
                raise RuntimeError("Couldn't resolve levels.") from exc

    def run_parallel_cpa(self, particle):
        particle.cpts.run_cpa(confidence=self.conf_parallel, run_levels=True)

    # TODO: remove this function
    def resolve_thread_complete(self, mode: str):
        """
        Is performed after thread has been terminated.

        Parameters
        ----------
        mode : {'current', 'selected', 'all'}
            Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
        """

        if self.tree2dataset().cpa_has_run:
            self.tabGrouping.setEnabled(True)
        if self.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
            self.display_data()
        dbg.p('Resolving levels complete', 'MainWindow')
        self.check_remove_bursts(mode=mode)
        self.set_startpoint()
        self.chbEx_Levels.setEnabled(True)

    def check_remove_bursts(self, mode: str = None) -> None:
        if mode == 'current':
            particles = [self.currentparticle]
        elif mode == 'selected':
            particles = self.get_checked_particles()
        else:
            particles = self.tree2dataset().particles

        removed_bursts = False
        has_burst = [particle.has_burst for particle in particles]
        if sum(has_burst):
            if not removed_bursts:
                removed_bursts = True
            msgbx = TimedMessageBox(30)
            msgbx.setIcon(QMessageBox.Question)
            msgbx.setText("Would you like to remove the photon bursts?")
            msgbx.set_timeout_text(message_pretime="(Removing photon bursts in ", message_posttime=" seconds)")
            msgbx.setWindowTitle("Photon bursts detected")
            msgbx.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
            msgbx.setDefaultButton(QMessageBox.Yes)
            msgbx.show()
            msgbx_result, _ = msgbx.exec()
            if msgbx_result == QMessageBox.Yes:
                for num, particle in enumerate(particles):
                    if has_burst[num]:
                        particle.cpts.remove_bursts()
            self.tree2dataset().makehistograms()

            self.display_data()

    def run_parallel_cpa(self, particle):
        particle.cpts.run_cpa(confidence=self.conf_parallel, run_levels=True)

    def switching_frequency(self, all_selected: str = None):
        """
        Calculates and exports the accumulated switching frequency of either
        all the particles, or only the selected.

        Parameters
        ----------
        all_selected : {'all', 'selected'}
            Possible values are 'all' (default) or 'selected'.
        """
        try:
            if all_selected is None:
                all_selected = 'all'

            assert all_selected.lower() in ['all', 'selected'], "mode parameter must be either 'all' or 'selected'."

            if all_selected is 'all':
                data = self.treemodel.data(self.treemodel.index(0, 0), Qt.UserRole)
                # assert data.
        except Exception as exc:
            dbg.p('Switching frequency analysis failed: ' + exc, "MainWidnow")
        else:
            pass

    def get_checked(self):
        checked = list()
        for ind in range(self.treemodel.rowCount(self.datasetindex)):
            if self.part_nodes[ind].checked():
                checked.append((ind, self.part_nodes[ind]))
                # checked_nums.append(ind)
                # checked_particles.append(self.part_nodes[ind])
        return checked

    def get_checked_nums(self):
        checked_nums = list()
        for ind in range(self.treemodel.rowCount(self.datasetindex)):
            if self.part_nodes[ind].checked():
                checked_nums.append(ind + 1)
        return checked_nums

    def get_checked_particles(self):
        checked_particles = list()
        for ind in range(self.treemodel.rowCount(self.datasetindex)):
            if self.part_nodes[ind].checked():
                checked_particles.append(self.tree2particle(ind))
        return checked_particles

    def set_level_resolved(self):
        self.level_resolved = True
        print(self.level_resolved)

    def export(self, mode: str = None):

        assert mode in ['current', 'selected', 'all'], "MainWindow\tThe mode parameter is invalid"

        if mode == 'current':
            particles = [self.currentparticle]
        elif mode == 'selected':
            particles = self.get_checked_particles()
        else:
            particles = self.tree2dataset().particles

        # save_dlg = QFileDialog(self)
        # save_dlg.setAcceptMode(QFileDialog.AcceptSave)
        # save_dlg.setFileMode(QFileDialog.DirectoryOnly)
        # save_dlg.setViewMode(QFileDialog.List)
        # save_dlg.setOption(QFileDialog.DontUseNativeDialog)
        # save_dlg.findChild(QLineEdit, 'fileNameEdit').setProperty('Visable', False)
        # save_dlg.findChild(QComboBox, 'fileTypeCombo').setProperty('enabled', True)
        # # filters = QDir.Filter("Comma delimited (*.csv)")  #;;Space delimited (*.txt);;Tab delimited (.*txt)
        # save_dlg.setNameFilter("Comma delimited (*.csv);;Space delimited (*.txt);;Tab delimited (.*txt)")
        # test2 = save_dlg.exec()

        f_dir = QFileDialog.getExistingDirectory(self)

        if f_dir:
            ex_traces = self.chbEx_Trace.isChecked()
            ex_levels = self.chbEx_Levels.isChecked()
            ex_lifetime = self.chbEx_Lifetimes.isChecked()
            ex_hist = self.chbEx_Hist.isChecked()
            for num, p in enumerate(particles):
                if ex_traces:
                    tr_path = os.path.join(f_dir, p.name + ' trace.csv')
                    ints = p.binnedtrace.intdata
                    times = p.binnedtrace.inttimes / 1E3
                    rows = list()
                    rows.append(['Bin #', 'Bin Time (s)', f'Bin Int (counts/{p.bin_size}ms)'])
                    for i in range(len(ints)):
                        rows.append([str(i), str(times[i]), str(ints[i])])
                    with open(tr_path, 'w') as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

                if ex_levels:
                    if p.has_levels:
                        lvl_tr_path = os.path.join(f_dir, p.name + ' levels-plot.csv')
                        ints, times = p.levels2data()
                        rows = list()
                        rows.append(['Level #', 'Time (s)', 'Int (counts/s)'])
                        for i in range(len(ints)):
                            rows.append([str(i // 2), str(times[i]), str(ints[i])])
                        with open(lvl_tr_path, 'w') as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

                        lvl_path = os.path.join(f_dir, p.name + ' levels.csv')
                        rows = list()
                        rows.append(['Level #', 'Start Time (s)', 'End Time (s)', 'Dwell Time (/s)',
                                     'Int (counts/s)', 'Num of Photons'])
                        for i, l in enumerate(p.levels):
                            rows.append([str(i), str(l.times_s[0]), str(l.times_s[1]), str(l.dwell_time_s),
                                         str(l.int_p_s), str(l.num_photons)])
                        with open(lvl_path, 'w') as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

                if ex_lifetime:
                    if p.numexp == 1:
                        taucol = ['Lifetime (ns)']
                        ampcol = ['Amp']
                    elif p.numexp == 2:
                        taucol = ['Lifetime 1 (ns)', 'Lifetime 2 (ns)']
                        ampcol = ['Amp 1', 'Amp 2']
                    elif p.numexp == 3:
                        taucol = ['Lifetime 1 (ns)', 'Lifetime 2 (ns)', 'Lifetime 3 (ns)']
                        ampcol = ['Amp 1', 'Amp 2', 'Amp 3']
                    if p.has_levels:
                        lvl_path = os.path.join(f_dir, p.name + ' levels lifetimes.csv')
                        rows = list()
                        rows.append(['Level #', 'Start Time (s)', 'End Time (s)', 'Dwell Time (/s)',
                                     'Int (counts/s)', 'Num of Photons'] + taucol + ampcol +
                                    ['Av. Lifetime (ns)', 'IRF Shift (ns)', 'Decay BG', 'IRF BG'])
                        for i, l in enumerate(p.levels):
                            if l.histogram.tau is None or l.histogram.amp is None:  # Problem with fitting the level
                                tauexp = ['0' for i in range(p.numexp)]
                                ampexp = ['0' for i in range(p.numexp)]
                                other_exp = ['0', '0', '0', '0']
                            else:
                                if p.numexp == 1:
                                    tauexp = [str(l.histogram.tau)]
                                    ampexp = [str(l.histogram.amp)]
                                else:
                                    tauexp = [str(tau) for tau in l.histogram.tau]
                                    ampexp = [str(amp) for amp in l.histogram.amp]
                                other_exp = [str(l.histogram.avtau), str(l.histogram.shift), str(l.histogram.bg),
                                             str(l.histogram.irfbg)]

                            rows.append([str(i), str(l.times_s[0]), str(l.times_s[1]), str(l.dwell_time_s),
                                         str(l.int_p_s), str(l.num_photons)] + tauexp + ampexp + other_exp)

                        with open(lvl_path, 'w') as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

                if ex_hist:
                    tr_path = os.path.join(f_dir, p.name + ' histogram.csv')
                    times = p.histogram.convd_t
                    if times is not None:
                        decay = p.histogram.fit_decay
                        convd = p.histogram.convd
                        rows = list()
                        rows.append(['Time (ns)', 'Decay', 'Fitted'])
                        for i, time in enumerate(times):
                            rows.append([str(time), str(decay[i]), str(convd[i])])

                        with open(tr_path, 'w') as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

                    if p.has_levels:
                        dir_path = os.path.join(f_dir, p.name + ' histograms')
                        try:
                            os.mkdir(dir_path)
                        except FileExistsError:
                            pass
                        for i, l in enumerate(p.levels):
                            hist_path = os.path.join(dir_path, 'level ' + str(i) + ' histogram.csv')
                            times = l.histogram.convd_t
                            if times is None:
                                continue
                            decay = l.histogram.fit_decay
                            convd = l.histogram.convd
                            rows = list()
                            rows.append(['Time (ns)', 'Decay', 'Fitted'])
                            for j, time in enumerate(times):
                                rows.append([str(time), str(decay[j]), str(convd[j])])

                            with open(hist_path, 'w') as f:
                                writer = csv.writer(f, dialect=csv.excel)
                                writer.writerows(rows)

                    dbg.p('Exporting Finished', 'MainWindow')

    def reset_gui(self):
        """ Sets the GUI elements to enabled if it should be accessible. """

        dbg.p('Reset GUI', 'MainWindow')
        if self.data_loaded:
            new_state = True
        else:
            new_state = False

        # Intensity
        self.tabIntensity.setEnabled(new_state)
        self.btnApplyBin.setEnabled(new_state)
        self.btnApplyBinAll.setEnabled(new_state)
        self.btnResolve.setEnabled(new_state)
        self.btnResolve_Selected.setEnabled(new_state)
        self.btnResolveAll.setEnabled(new_state)
        self.cmbConfIndex.setEnabled(new_state)
        self.spbBinSize.setEnabled(new_state)
        self.actionReset_Analysis.setEnabled(new_state)
        self.actionSave_Selected.setEnabled(new_state)
        if new_state:
            enable_levels = self.level_resolved
        else:
            enable_levels = new_state
        self.actionTrim_Dead_Traces.setEnabled(enable_levels)

        # Lifetime
        self.tabLifetime.setEnabled(new_state)
        self.btnFitParameters.setEnabled(new_state)
        self.btnLoadIRF.setEnabled(new_state)
        if new_state:
            enable_fitting = self.lifetime_controller.irf_loaded
        else:
            enable_fitting = new_state
        self.btnFitCurrent.setEnabled(enable_fitting)
        self.btnFit.setEnabled(enable_fitting)
        self.btnFitAll.setEnabled(enable_fitting)
        self.btnFitSelected.setEnabled(enable_fitting)
        self.btnNextLevel.setEnabled(enable_levels)
        self.btnPrevLevel.setEnabled(enable_levels)
        # print(enable_levels)

        # Spectral
        if self.has_spectra:
            self.tabSpectra.setEnabled(True)
            self.btnSubBackground.setEnabled(new_state)
        else:
            self.tabSpectra.setEnabled(False)

    @property
    def current_level(self):
        return self._current_level

    @current_level.setter
    def current_level(self, value):
        if value is None:
            self._current_level = None
        else:
            try:
                # print(self.currentparticle.current2data(value))
                self._current_level = value
            except:
                pass


class IntController(QObject):

    def __init__(self, mainwindow):
        super().__init__()

        self.mainwindow = mainwindow

        self.confidence_index = {
            0: 99,
            1: 95,
            2: 90,
            3: 69}

    def gui_apply_bin(self):
        """ Changes the bin size of the data of the current particle and then displays the new trace. """

        # self.pgSpectra.centralWidget
        #
        # self.pgIntensity.getPlotItem().setFixedWidth(500)
        # self.pgSpectra.resize(100, 200)
        # self.pgIntensity.getPlotItem().getAxis('left').setRange(0, 100)
        # window_color = self.palette().color(QPalette.Window)
        # rgba_color = (window_color.red()/255, window_color.green()/255, window_color.blue()/255, 1)
        # self.pgIntensity.setBackground(background=rgba_color)
        # self.pgIntensity.setXRange(0, 10, 0)
        # self.pgIntensity.getPlotItem().plot(y=[1, 2, 3, 4, 5])
        try:
            self.mainwindow.currentparticle.binints(self.get_bin())
        except Exception as err:
            dbg.p('Error Occured:' + str(err), "IntController")
        else:
            self.mainwindow.display_data()
            self.mainwindow.repaint()
            dbg.p('Single trace binned', 'IntController')

    def get_bin(self) -> int:
        """ Returns current GUI value for bin size in ms.

        Returns
        -------
        int
            The value of the bin size on the GUI in spbBinSize.
        """

        return self.mainwindow.spbBinSize.value()

    def set_bin(self, new_bin: int):
        """ Sets the GUI value for the bin size in ms

        Parameters
        ----------
        new_bin: int
            Value to set bin size to, in ms.
        """
        self.mainwindow.spbBinSize.setValue(new_bin)

    def gui_apply_bin_all(self):
        """ Changes the bin size of the data of all the particles and then displays the new trace of the current particle. """

        try:
            self.mainwindow.start_binall_thread(self.get_bin())
        except Exception as err:
            dbg.p('Error Occured:' + str(err), "IntController")
        else:
            self.plot_trace()
            self.mainwindow.repaint()
            dbg.p('All traces binned', 'IntController')

    def ask_end_time(self):
        """ Prompts the user to supply an end time."""

        end_time_s, ok = QInputDialog.getDouble(self.mainwindow, 'End Time', 'Provide end time in seconds', 0, 1, 10000, 3)
        return end_time_s, ok

    def time_resolve_current(self):
        """ Resolves the levels of the current particle to an end time asked of the user."""

        end_time_s, ok = self.ask_end_time()
        if ok:
            self.gui_resolve(end_time_s=end_time_s)

    def time_resolve_selected(self):
        """ Resolves the levels of the selected particles to an end time asked of the user."""

        end_time_s, ok = self.ask_end_time()
        if ok:
            self.gui_resolve_selected(end_time_s=end_time_s)

    def time_resolve_all(self):
        """ Resolves the levels of all the particles to an end time asked of the user."""

        end_time_s, ok = self.ask_end_time()
        if ok:
            self.gui_resolve_all(end_time_s=end_time_s)

    def gui_resolve(self, end_time_s=None):
        """ Resolves the levels of the current particle and displays it. """

        self.start_resolve_thread(mode='current', end_time_s=end_time_s)

    def gui_resolve_selected(self, end_time_s=None):
        """ Resolves the levels of the selected particles and displays the levels of the current particle. """

        self.start_resolve_thread(mode='selected', end_time_s=end_time_s)

    def gui_resolve_all(self, end_time_s=None):
        """ Resolves the levels of the all the particles and then displays the levels of the current particle. """

        self.start_resolve_thread(mode='all', end_time_s=end_time_s)

    def plot_trace(self) -> None:
        """ Used to display the trace from the absolute arrival time data of the current particle. """

        try:
            # self.currentparticle = self.treemodel.data(self.current_ind, Qt.UserRole)
            trace = self.mainwindow.currentparticle.binnedtrace.intdata
            times = self.mainwindow.currentparticle.binnedtrace.inttimes / 1E3
        except AttributeError:
            dbg.p('No trace!', 'IntController')
        else:
            plot_pen = QPen()
            plot_pen.setCosmetic(True)
            cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()
            if cur_tab_name != 'tabSpectra':
                if cur_tab_name == 'tabIntensity':
                    plot_item = self.mainwindow.pgIntensity.getPlotItem()
                    plot_pen.setWidthF(1.5)
                    plot_pen.setColor(QColor('green'))
                elif cur_tab_name == 'tabLifetime':
                    plot_item = self.mainwindow.pgLifetime_Int.getPlotItem()
                    plot_pen.setWidthF(1.1)
                    plot_pen.setColor(QColor('green'))
                elif cur_tab_name == 'tabGrouping':
                    plot_item = self.mainwindow.pgGroups_Int
                    plot_pen.setWidthF(1.1)
                    plot_pen.setColor(QColor(0, 0, 0, 50))

                plot_pen.setJoinStyle(Qt.RoundJoin)

                plot_item.clear()
                unit = 'counts/' + str(self.get_bin()) + 'ms'
                plot_item.getAxis('left').setLabel(text='Intensity', units=unit)
                plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=times[-1])
                plot_item.plot(x=times, y=trace, pen=plot_pen, symbol=None)

    def plot_levels(self):
        """ Used to plot the resolved intensity levels of the current particle. """
        currentparticle = self.mainwindow.currentparticle
        # print('levels plto')
        try:
            level_ints, times = currentparticle.levels2data()
            level_ints = level_ints * self.get_bin() / 1E3
        except AttributeError:
            dbg.p('No levels!', 'IntController')
        else:
            cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()
            if cur_tab_name == 'tabIntensity':
                plot_item = self.mainwindow.pgIntensity.getPlotItem()
                # pen_width = 1.5
            elif cur_tab_name == 'tabLifetime':
                plot_item = self.mainwindow.pgLifetime_Int.getPlotItem()
                # pen_width = 1.1
            elif cur_tab_name == 'tabGrouping':
                plot_item = self.mainwindow.pgGroups_Int.getPlotItem()
            else:
                return

        plot_pen = QPen()
        plot_pen.setWidthF(2)
        plot_pen.brush()
        plot_pen.setJoinStyle(Qt.RoundJoin)
        plot_pen.setColor(QColor('black'))
        plot_pen.setCosmetic(True)

        plot_item.plot(x=times, y=level_ints, pen=plot_pen, symbol=None)

        if self.mainwindow.current_level is not None:
            current_ints, current_times = currentparticle.current2data(self.mainwindow.current_level)
            current_ints = current_ints * self.get_bin() / 1E3
            # print(current_ints, current_times)

            if not (current_ints[0] == np.inf or current_ints[1] == np.inf):
                plot_pen.setColor(QColor('red'))
                plot_pen.setWidthF(3)
                plot_item.plot(x=current_times, y=current_ints, pen=plot_pen, symbol=None)
            else:
                dbg.p('Infinity in level', 'IntController')

    def start_resolve_thread(self, mode: str = 'current', thread_finished=None, end_time_s=None) -> None:
        """
        Creates a worker to resolve levels.

        Depending on the ``current_selected_all`` parameter the worker will be
        given the necessary parameter to fit the current, selected or all particles.

        Parameters
        ----------
        end_time_s : float
        thread_finished
        mode : {'current', 'selected', 'all'}
            Possible values are 'current' (default), 'selected', and 'all'.
        """

        if thread_finished is None:
            if self.mainwindow.data_loaded:
                thread_finished = self.resolve_thread_complete
            else:
                thread_finished = self.mainwindow.open_file_thread_complete

        _, conf = self.get_gui_confidence()
        data = self.mainwindow.tree2dataset()
        currentparticle = self.mainwindow.currentparticle
        # print(currentparticle)

        # print(mode)
        if mode == 'current':
            # sig = WorkerSignals()
            # self.resolve_levels(sig.start_progress, sig.progress, sig.status_message)
            resolve_thread = WorkerResolveLevels(resolve_levels, conf, data, currentparticle, mode, end_time_s=end_time_s)
        elif mode == 'selected':
            resolve_thread = WorkerResolveLevels(resolve_levels, conf, data, currentparticle, mode,
                                                 resolve_selected=self.mainwindow.get_checked_particles(), end_time_s=end_time_s)
        elif mode == 'all':
            resolve_thread = WorkerResolveLevels(resolve_levels, conf, data, currentparticle, mode, end_time_s=end_time_s)
            # resolve_thread.signals.finished.connect(thread_finished)
            # resolve_thread.signals.start_progress.connect(self.start_progress)
            # resolve_thread.signals.progress.connect(self.update_progress)
            # resolve_thread.signals.status_message.connect(self.status_message)
            # self.resolve_levels(resolve_thread.signals.start_progress, resolve_thread.signals.progress,
            #                     resolve_thread.signals.status_message, resolve_all=True, parallel=True)

        resolve_thread.signals.resolve_finished.connect(self.resolve_thread_complete)
        resolve_thread.signals.start_progress.connect(self.mainwindow.start_progress)
        resolve_thread.signals.progress.connect(self.mainwindow.update_progress)
        resolve_thread.signals.status_message.connect(self.mainwindow.status_message)
        resolve_thread.signals.reset_gui.connect(self.mainwindow.reset_gui)
        resolve_thread.signals.level_resolved.connect(self.mainwindow.set_level_resolved)

        self.mainwindow.threadpool.start(resolve_thread)

    def resolve_thread_complete(self, mode):
        if self.mainwindow.tree2dataset().cpa_has_run:
            self.mainwindow.tabGrouping.setEnabled(True)
        if self.mainwindow.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
            self.mainwindow.display_data()
        self.mainwindow.check_remove_bursts(mode=mode)
        self.mainwindow.chbEx_Levels.setEnabled(True)
        self.mainwindow.set_startpoint()
        dbg.p('Resolving levels complete', 'IntController')

    def get_gui_confidence(self):
        """ Return current GUI value for confidence percentage. """

        return [self.mainwindow.cmbConfIndex.currentIndex(),
                self.confidence_index[self.mainwindow.cmbConfIndex.currentIndex()]]


class LifetimeController(QObject):

    def __init__(self, mainwindow):
        super().__init__()

        self.mainwindow = mainwindow
        self.fitparamdialog = FittingDialog(self.mainwindow, self)
        self.fitparam = FittingParameters(self)
        self.irf_loaded = False

        self.first = 0
        self.startpoint = None
        self.tmin = 0

    def gui_prev_lev(self):
        """ Moves to the previous resolves level and displays its decay curve. """

        if self.mainwindow.current_level is None:
            pass
        elif self.mainwindow.current_level == 0:
            self.mainwindow.current_level = None
        else:
            self.mainwindow.current_level -= 1
        self.mainwindow.display_data()

    def gui_next_lev(self):
        """ Moves to the next resolves level and displays its decay curve. """

        if self.mainwindow.current_level is None:
            self.mainwindow.current_level = 0
        else:
            self.mainwindow.current_level += 1
        self.mainwindow.display_data()

    def gui_whole_trace(self):
        "Unselects selected level and shows whole trace's decay curve"

        self.mainwindow.current_level = None
        self.mainwindow.display_data()

    def gui_load_irf(self):
        """ Allow the user to load a IRF instead of the IRF that has already been loaded. """

        fname = QFileDialog.getOpenFileName(self.mainwindow, 'Open HDF5 file', '', "HDF5 files (*.h5)")
        if fname != ('', ''):  # fname will equal ('', '') if the user canceled.
            of_worker = WorkerOpenFile(fname, irf=True, tmin=self.tmin)
            of_worker.signals.openfile_finished.connect(self.mainwindow.open_file_thread_complete)
            of_worker.signals.start_progress.connect(self.mainwindow.start_progress)
            of_worker.signals.progress.connect(self.mainwindow.update_progress)
            of_worker.signals.auto_progress.connect(self.mainwindow.update_progress)
            of_worker.signals.start_progress.connect(self.mainwindow.start_progress)
            of_worker.signals.status_message.connect(self.mainwindow.status_message)
            of_worker.signals.add_datasetindex.connect(self.mainwindow.add_dataset)
            of_worker.signals.add_particlenode.connect(self.mainwindow.add_node)
            of_worker.signals.reset_tree.connect(lambda: self.mainwindow.treemodel.modelReset.emit())
            of_worker.signals.data_loaded.connect(self.mainwindow.set_data_loaded)
            of_worker.signals.bin_size.connect(self.mainwindow.spbBinSize.setValue)
            of_worker.signals.add_irf.connect(self.add_irf)

            self.mainwindow.threadpool.start(of_worker)

    def add_irf(self, decay, t, irfdata):

        self.fitparam.irf = decay
        self.fitparam.irft = t
        self.fitparam.irfdata = irfdata
        self.irf_loaded = True
        self.mainwindow.set_startpoint()
        self.mainwindow.reset_gui
        self.fitparamdialog.updateplot()

    def gui_fit_param(self):
        """ Opens a dialog to choose the setting with which the decay curve will be fitted. """

        if self.fitparamdialog.exec():
            self.fitparam.getfromdialog()

    def gui_fit_current(self):
        """ Fits the currently selected level's decay curve using the provided settings. """

        if self.mainwindow.current_level is None:
            histogram = self.mainwindow.currentparticle.histogram
        else:
            level = self.mainwindow.current_level
            histogram = self.mainwindow.currentparticle.levels[level].histogram
        try:
            channelwidth = self.mainwindow.currentparticle.channelwidth
            shift = self.fitparam.shift / channelwidth
            # shift = self.fitparam.shift
            if self.fitparam.start is not None:
                start = int(self.fitparam.start / channelwidth)
            else:
                start = None
            if self.fitparam.end is not None:
                end = int(self.fitparam.end / channelwidth)
            else:
                end = None
            if not histogram.fit(self.fitparam.numexp, self.fitparam.tau, self.fitparam.amp,
                                 shift, self.fitparam.decaybg, self.fitparam.irfbg,
                                 start, end, self.fitparam.addopt,
                                 self.fitparam.irf, self.fitparam.shiftfix):
                return  # fit unsuccessful
        except AttributeError:
            dbg.p("No decay", "Lifetime Fitting")
        else:
            self.mainwindow.display_data()

    def gui_fit_selected(self):
        """ Fits the all the levels decay curves in the all the selected particles using the provided settings. """

        self.start_fitting_thread(mode='selected')

    def gui_fit_all(self):
        """ Fits the all the levels decay curves in the all the particles using the provided settings. """

        self.start_fitting_thread(mode='all')

    def gui_fit_levels(self):
        """ Fits the all the levels decay curves for the current particle. """

        self.start_fitting_thread()

    def update_results(self):

        currentparticle = self.mainwindow.currentparticle
        if self.mainwindow.current_level is None:
            histogram = currentparticle.histogram
        else:
            level = self.mainwindow.current_level
            histogram = currentparticle.levels[level].histogram
        if not histogram.fitted:
            return
        tau = histogram.tau
        amp = histogram.amp
        shift = histogram.shift
        bg = histogram.bg
        irfbg = histogram.irfbg
        try:
            taustring = 'Tau = ' + ' '.join('{:#.3g} ns'.format(F) for F in tau)
            ampstring = 'Amp = ' + ' '.join('{:#.3g} '.format(F) for F in amp)
        except TypeError:  # only one component
            taustring = 'Tau = {:#.3g} ns'.format(tau)
            ampstring = 'Amp = {:#.3g}'.format(amp)
        shiftstring = 'Shift = {:#.3g} ns'.format(shift)
        bgstring = 'Decay BG = {:#.3g}'.format(bg)
        irfbgstring = 'IRF BG = {:#.3g}'.format(irfbg)
        self.mainwindow.textBrowser.setText(
            taustring + '\n' + ampstring + '\n' + shiftstring + '\n' + bgstring + '\n' +
            irfbgstring)

    def plot_decay(self, remove_empty: bool = False) -> None:
        """ Used to display the histogram of the decay data of the current particle. """

        currentlevel = self.mainwindow.current_level
        # print(currentlevel)
        currentparticle = self.mainwindow.currentparticle
        if currentlevel is None:
            if currentparticle.histogram.fitted:
                decay = currentparticle.histogram.fit_decay
                t = currentparticle.histogram.convd_t
            else:
                try:
                    decay = currentparticle.histogram.decay
                    t = currentparticle.histogram.t

                except AttributeError:
                    dbg.p(debug_print='No Decay!', debug_from='LifetimeController')
                    return
        else:
            if currentparticle.levels[currentlevel].histogram.fitted:
                decay = currentparticle.levels[currentlevel].histogram.fit_decay
                t = currentparticle.levels[currentlevel].histogram.convd_t
            else:
                try:
                    decay = currentparticle.levels[currentlevel].histogram.decay
                    t = currentparticle.levels[currentlevel].histogram.t
                except ValueError:
                    return

        if decay.size == 0:
            return  # some levels have no photons

        if self.mainwindow.tabWidget.currentWidget().objectName() == 'tabLifetime':
            plot_item = self.mainwindow.pgLifetime.getPlotItem()
            plot_pen = QPen()
            plot_pen.setWidthF(1.5)
            plot_pen.setJoinStyle(Qt.RoundJoin)
            plot_pen.setColor(QColor('blue'))
            plot_pen.setCosmetic(True)

            if remove_empty:
                self.first = (decay > 4).argmax(axis=0)
                t = t[self.first:-1] - t[self.first]
                decay = decay[self.first:-1]
            else:
                self.first = 0

            # try:
            #     decay = decay / decay.max()
            # except ValueError:  # Empty decay
            #     return
            # print(decay.max())
            plot_item.clear()
            plot_item.plot(x=t, y=decay, pen=plot_pen, symbol=None)
            unit = 'ns with ' + str(currentparticle.channelwidth) + 'ns bins'
            plot_item.getAxis('bottom').setLabel('Decay time', unit)
            plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=t[-1])
            self.fitparamdialog.updateplot()

    def plot_convd(self, remove_empty: bool = False) -> None:
        """ Used to display the histogram of the decay data of the current particle. """

        currentlevel = self.mainwindow.current_level
        currentparticle = self.mainwindow.currentparticle
        if currentlevel is None:
            try:
                convd = currentparticle.histogram.convd
                t = currentparticle.histogram.convd_t

            except AttributeError:
                dbg.p(debug_print='No Decay!', debug_from='LifetimeController')
                return
        else:
            try:
                convd = currentparticle.levels[currentlevel].histogram.convd
                t = currentparticle.levels[currentlevel].histogram.convd_t
            except ValueError:
                return

        if convd is None or t is None:
            return

        # convd = convd / convd.max()

        if self.mainwindow.tabWidget.currentWidget().objectName() == 'tabLifetime':
            plot_item = self.mainwindow.pgLifetime.getPlotItem()
            plot_pen = QPen()
            plot_pen.setWidthF(4)
            plot_pen.setJoinStyle(Qt.RoundJoin)
            plot_pen.setColor(QColor('dark blue'))
            plot_pen.setCosmetic(True)

            # if remove_empty:
            #     first = (decay > 4).argmax(axis=0)
            #     t = t[first:-1] - t[first]
            #     decay = decay[first:-1]
            # convd = convd[self.first:-1]

            # plot_item.clear()
            plot_item.plot(x=t, y=convd, pen=plot_pen, symbol=None)
            unit = 'ns with ' + str(currentparticle.channelwidth) + 'ns bins'
            plot_item.getAxis('bottom').setLabel('Decay time', unit)
            plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=t[-1])

    def start_fitting_thread(self, mode: str = 'current', thread_finished=None) -> None:
        """
        Creates a worker to resolve levels.

        Depending on the ``current_selected_all`` parameter the worker will be
        given the necessary parameter to fit the current, selected or all particles.

        Parameters
        ----------
        thread_finished
        mode : {'current', 'selected', 'all'}
            Possible values are 'current' (default), 'selected', and 'all'.
        """

        if thread_finished is None:
            if self.mainwindow.data_loaded:
                thread_finished = self.fitting_thread_complete
            else:
                thread_finished = self.mainwindow.open_file_thread_complete

        data = self.mainwindow.tree2dataset()
        currentparticle = self.mainwindow.currentparticle

        print(mode)
        if mode == 'current':
            fitting_thread = WorkerFitLifetimes(fit_lifetimes, data, currentparticle, self.fitparam, mode)
        elif mode == 'selected':
            fitting_thread = WorkerFitLifetimes(fit_lifetimes, data, currentparticle, self.fitparam, mode,
                                                resolve_selected=self.mainwindow.get_checked_particles())
        elif mode == 'all':
            fitting_thread = WorkerFitLifetimes(fit_lifetimes, data, currentparticle, self.fitparam, mode)

        fitting_thread.signals.fitting_finished.connect(self.fitting_thread_complete)
        fitting_thread.signals.start_progress.connect(self.mainwindow.start_progress)
        fitting_thread.signals.progress.connect(self.mainwindow.update_progress)
        fitting_thread.signals.status_message.connect(self.mainwindow.status_message)
        fitting_thread.signals.reset_gui.connect(self.mainwindow.reset_gui)

        self.mainwindow.threadpool.start(fitting_thread)

    def fitting_thread_complete(self, mode):
        if self.mainwindow.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
            self.mainwindow.display_data()
        self.mainwindow.chbEx_Lifetimes.setEnabled(False)
        self.mainwindow.chbEx_Lifetimes.setEnabled(True)
        self.mainwindow.chbEx_Hist.setEnabled(True)
        print(self.mainwindow.chbEx_Lifetimes.isChecked())
        dbg.p('Fitting levels complete', 'Fitting Thread')

    def change_irf_start(self, start):
        dataset = self.fitparam.irfdata

        dataset.makehistograms(remove_zeros=False, startpoint=start, channel=True)
        irfhist = dataset.particles[0].histogram
        # irfhist.t -= irfhist.t.min()
        self.fitparam.irf = irfhist.decay
        self.fitparam.irft = irfhist.t
        # ind = np.searchsorted(self.fitparam.irft, start)
        # print(self.fitparam.irft)
        # print(ind)
        # self.fitparam.irft = self.fitparam.irft[ind:]
        # self.fitparam.irf = self.fitparam.irf[ind:]
        # print(self.fitparam.irft)

    def set_tmin(self, tmin=0):
        self.tmin = tmin


class GroupingController(QObject):

    def __init__(self, mainwidow: MainWindow):
        super().__init__()

        self.mainwindow = mainwidow

    def plot_hist(self):
        try:
            int_data = self.mainwindow.currentparticle.binnedtrace.intdata
        except AttributeError:
            dbg.p('No trace!', 'GroupingController')
        else:
            plot_pen = QPen()
            plot_pen.setColor(QColor(0, 0, 0, 0))
            plot_item = self.mainwindow.pgGroups_Hist.getPlotItem()
            plot_item.clear()

            bin_edges = np.histogram_bin_edges(np.negative(int_data), bins='auto')
            freq, hist_bins = np.histogram(np.negative(int_data), bins=bin_edges, density=True)
            freq /= np.max(freq)
            int_hist = pg.PlotCurveItem(x=hist_bins, y=freq, pen=plot_pen,
                                        stepMode=True, fillLevel=0, brush=(0, 0, 0, 50))
            int_hist.rotate(-90)
            plot_item.addItem(int_hist)

            if self.mainwindow.currentparticle.has_levels:
                level_ints = self.mainwindow.currentparticle.level_ints

                level_ints *= self.mainwindow.currentparticle.bin_size/1000
                dwell_times = [level.dwell_time_s for level in self.mainwindow.currentparticle.levels]
                level_freq, level_hist_bins = np.histogram(np.negative(level_ints), bins=bin_edges,
                                                           weights=dwell_times, density=True)
                level_freq /= np.max(level_freq)
                level_hist = pg.PlotCurveItem(x=level_hist_bins, y=level_freq, stepMode=True,
                                              pen=plot_pen, fillLevel=0, brush=(0, 0, 0, 255))

                level_hist.rotate(-90)
                plot_item.addItem(level_hist)

    def plot_groups(self):
        currentparticle = self.mainwindow.currentparticle
        # print('levels plto')
        try:
            groups = currentparticle.groups
            num_groups = currentparticle.num_groups
            num_levels = currentparticle.num_levels
        except AttributeError:
            dbg.p('No groups!', 'GroupingController')
        else:
            cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()
            if cur_tab_name == 'tabGrouping':
                plot_item = self.mainwindow.pgGroups_Int.getPlotItem()
            else:
                return

        plot_pen = QPen()
        plot_pen.setWidthF(2)
        plot_pen.brush()
        plot_pen.setJoinStyle(Qt.RoundJoin)
        plot_pen.setColor(QColor('black'))
        plot_pen.setCosmetic(True)

        # plot_item.plot(x=times, y=level_ints, pen=plot_pen, symbol=None)
        pass

    def gui_group_current(self):
        self.start_grouping_thread(mode='current')

    def gui_group_selected(self):
        self.start_grouping_thread(mode='selected')

    def gui_group_all(self):
        self.start_grouping_thread(mode='all')

    def start_grouping_thread(self, mode: str = 'current') -> None:
        """
        Creates a worker to resolve levels.

        Depending on the ``current_selected_all`` parameter the worker will be
        given the necessary parameter to fit the current, selected or all particles.

        Parameters
        ----------
        thread_finished
        mode : {'current', 'selected', 'all'}
            Possible values are 'current' (default), 'selected', and 'all'.
        """

        data = self.mainwindow.tree2dataset()

        print(mode)
        if mode == 'current':
            grouping_worker = WorkerGrouping(data=data,
                                             grouping_func=group_levels,
                                             currentparticle=self.mainwindow.currentparticle,
                                             mode='current')
        elif mode == 'selected':
            grouping_worker = WorkerGrouping(data=data,
                                             grouping_func=group_levels,
                                             mode='selected',
                                             group_selected=self.mainwindow.get_checked_particles())
        elif mode == 'all':
            grouping_worker = WorkerGrouping(data=data,
                                             grouping_func=group_levels,
                                             mode='all')

        grouping_worker.signals.grouping_finished.connect(self.grouping_thread_complete)
        grouping_worker.signals.start_progress.connect(self.mainwindow.start_progress)
        grouping_worker.signals.progress.connect(self.mainwindow.update_progress)
        grouping_worker.signals.status_message.connect(self.mainwindow.status_message)
        grouping_worker.signals.reset_gui.connect(self.mainwindow.reset_gui)

        self.mainwindow.threadpool.start(grouping_worker)

    def grouping_thread_complete(self, mode):
        if self.mainwindow.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
            self.mainwindow.display_data()
        dbg.p('Grouping levels complete', 'Grouping Thread')


class SpectraController(QObject):

    def __init__(self, mainwindow: MainWindow):
        super().__init__()

        self.mainwindow = mainwindow

    def gui_sub_bkg(self):
        """ Used to subtract the background TODO: Explain the sub_background """

        print("gui_sub_bkg")


class FittingDialog(QDialog, UI_Fitting_Dialog):
    """Class for dialog that is used to choose lifetime fit parameters."""

    def __init__(self, mainwindow, lifetime_controller):
        QDialog.__init__(self)
        UI_Fitting_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow
        self.lifetime_controller = lifetime_controller
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self.updateplot)
        for widget in self.findChildren(QCheckBox):
            widget.stateChanged.connect(self.updateplot)
        for widget in self.findChildren(QComboBox):
            widget.currentTextChanged.connect(self.updateplot)
        self.updateplot()

        # self.lineStartTime.setValidator(QIntValidator())
        # self.lineEndTime.setValidator(QIntValidator())

    def updateplot(self, *args):

        try:
            model = self.make_model()
        except Exception as err:
            dbg.p(debug_print='Error Occured:' + str(err), debug_from='FittingDialog')
            return

        fp = self.lifetime_controller.fitparam
        try:
            irf = fp.irf
            irft = fp.irft
        except AttributeError:
            dbg.p(debug_print='No IRF!', debug_from='FittingDialog')
            return

        shift, decaybg, irfbg, start, end = self.getparams()

        channelwidth = self.mainwindow.currentparticle.channelwidth
        shift = shift / channelwidth
        start = int(start / channelwidth)
        end = int(end / channelwidth)
        irf = tcspcfit.colorshift(irf, shift)
        convd = scipy.signal.convolve(irf, model)
        convd = convd[:np.size(irf)]
        convd = convd / convd.max()

        try:
            if self.mainwindow.current_level is None:
                histogram = self.mainwindow.currentparticle.histogram
            else:
                level = self.mainwindow.current_level
                histogram = self.mainwindow.currentparticle.levels[level].histogram
            decay = histogram.decay
            decay = decay / decay.max()
            t = histogram.t

            # decay, t = start_at_value(decay, t)
            end = min(end, np.size(t) - 1)  # Make sure endpoint is not bigger than size of t

            convd = convd[irft > 0]
            irft = irft[irft > 0]

        except AttributeError:
            dbg.p(debug_print='No Decay!', debug_from='FittingDialog')
        else:
            plot_item = self.pgFitParam.getPlotItem()
            plot_item.setLogMode(y=True)
            plot_pen = QPen()
            plot_pen.setWidthF(3)
            plot_pen.setJoinStyle(Qt.RoundJoin)
            plot_pen.setColor(QColor('blue'))
            plot_pen.setCosmetic(True)

            plot_item.clear()
            plot_item.plot(x=t, y=np.clip(decay, a_min=0.001, a_max=None), pen=plot_pen, symbol=None)
            plot_pen.setWidthF(4)
            plot_pen.setColor(QColor('dark blue'))
            plot_item.plot(x=irft, y=np.clip(convd, a_min=0.001, a_max=None), pen=plot_pen, symbol=None)
            # unit = 'ns with ' + str(currentparticle.channelwidth) + 'ns bins'
            plot_item.getAxis('bottom').setLabel('Decay time (ns)')
            # plot_item.getViewBox().setLimits(xMin=0, yMin=0.1, xMax=t[-1], yMax=1)
            # plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=t[-1])
            # self.MW_fitparam.axes.clear()
            # self.MW_fitparam.axes.semilogy(t, decay, color='xkcd:dull blue')
            # self.MW_fitparam.axes.semilogy(irft, convd, color='xkcd:marine blue', linewidth=2)
            # self.MW_fitparam.axes.set_ylim(bottom=1e-2)

        try:
            plot_pen.setColor(QColor('gray'))
            plot_pen.setWidth(3)
            startline = pg.InfiniteLine(angle=90, pen=plot_pen, movable=False, pos=t[start])
            endline = pg.InfiniteLine(angle=90, pen=plot_pen, movable=False, pos=t[end])
            plot_item.addItem(startline)
            plot_item.addItem(endline)
            # self.MW_fitparam.axes.axvline(t[start])
            # self.MW_fitparam.axes.axvline(t[end])
        except IndexError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText('Value out of bounds!')
            msg.exec_()

    def getparams(self):
        fp = self.lifetime_controller.fitparam
        irf = fp.irf
        shift = fp.shift
        if shift is None:
            shift = 0
        decaybg = fp.decaybg
        if decaybg is None:
            decaybg = 0
        irfbg = fp.irfbg
        if irfbg is None:
            irfbg = 0
        start = fp.start
        if start is None:
            start = 0
        end = fp.end
        if end is None:
            end = np.size(irf)
        return shift, decaybg, irfbg, start, end

    def make_model(self):
        fp = self.lifetime_controller.fitparam
        t = self.mainwindow.currentparticle.histogram.t
        fp.getfromdialog()
        if fp.numexp == 1:
            tau = fp.tau[0][0]
            model = np.exp(-t / tau)
        elif fp.numexp == 2:
            tau1 = fp.tau[0][0]
            tau2 = fp.tau[1][0]
            amp1 = fp.amp[0][0]
            amp2 = fp.amp[1][0]
            # print(amp1, amp2, tau1, tau2)
            model = amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2)
        elif fp.numexp == 3:
            tau1 = fp.tau[0][0]
            tau2 = fp.tau[1][0]
            tau3 = fp.tau[2][0]
            amp1 = fp.amp[0][0]
            amp2 = fp.amp[1][0]
            amp3 = fp.amp[2][0]
            model = amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2) + amp3 * np.exp(-t / tau3)
        return model


class FittingParameters:
    def __init__(self, parent):
        self.parent = parent
        self.fpd = self.parent.fitparamdialog
        self.irf = None
        self.tau = None
        self.amp = None
        self.shift = None
        self.shiftfix = None
        self.decaybg = None
        self.irfbg = None
        self.start = None
        self.end = None
        self.numexp = None
        self.addopt = None

    def getfromdialog(self):
        self.numexp = int(self.fpd.combNumExp.currentText())
        if self.numexp == 1:
            self.tau = [[self.get_from_gui(i) for i in
                         [self.fpd.line1Init, self.fpd.line1Min, self.fpd.line1Max, self.fpd.check1Fix]]]
            self.amp = [[self.get_from_gui(i) for i in
                         [self.fpd.line1AmpInit, self.fpd.line1AmpMin, self.fpd.line1AmpMax, self.fpd.check1AmpFix]]]

        elif self.numexp == 2:
            self.tau = [[self.get_from_gui(i) for i in
                         [self.fpd.line2Init1, self.fpd.line2Min1, self.fpd.line2Max1, self.fpd.check2Fix1]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line2Init2, self.fpd.line2Min2, self.fpd.line2Max2, self.fpd.check2Fix2]]]
            self.amp = [[self.get_from_gui(i) for i in
                         [self.fpd.line2AmpInit1, self.fpd.line2AmpMin1, self.fpd.line2AmpMax1,
                          self.fpd.check2AmpFix1]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line2AmpInit2, self.fpd.line2AmpMin2, self.fpd.line2AmpMax2,
                          self.fpd.check2AmpFix2]]]

        elif self.numexp == 3:
            self.tau = [[self.get_from_gui(i) for i in
                         [self.fpd.line3Init1, self.fpd.line3Min1, self.fpd.line3Max1, self.fpd.check3Fix1]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line3Init2, self.fpd.line3Min2, self.fpd.line3Max2, self.fpd.check3Fix2]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line3Init3, self.fpd.line3Min3, self.fpd.line3Max3, self.fpd.check3Fix3]]]
            self.amp = [[self.get_from_gui(i) for i in
                         [self.fpd.line3AmpInit1, self.fpd.line3AmpMin1, self.fpd.line3AmpMax1,
                          self.fpd.check3AmpFix1]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line3AmpInit2, self.fpd.line3AmpMin2, self.fpd.line3AmpMax2,
                          self.fpd.check3AmpFix2]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line3AmpInit3, self.fpd.line3AmpMin3, self.fpd.line3AmpMax3,
                          self.fpd.check3AmpFix3]]]

        self.shift = self.get_from_gui(self.fpd.lineShift)
        self.shiftfix = self.get_from_gui(self.fpd.checkFixIRF)
        self.decaybg = self.get_from_gui(self.fpd.lineDecayBG)
        self.irfbg = self.get_from_gui(self.fpd.lineIRFBG)
        self.start = self.get_from_gui(self.fpd.lineStartTime)
        self.end = self.get_from_gui(self.fpd.lineEndTime)
        # try:
        #     self.start = int(self.get_from_gui(self.fpd.lineStartTime))
        # except TypeError:
        #     self.start = self.get_from_gui(self.fpd.lineStartTime)
        # try:
        #     self.end = int(self.get_from_gui(self.fpd.lineEndTime))
        # except TypeError:
        #     self.end = self.get_from_gui(self.fpd.lineEndTime)

        if self.fpd.lineAddOpt.text() != '':
            self.addopt = self.fpd.lineAddOpt.text()
        else:
            self.addopt = None

    @staticmethod
    def get_from_gui(guiobj):
        if type(guiobj) == QLineEdit:
            if guiobj.text() == '':
                return None
            else:
                return float(guiobj.text())
        elif type(guiobj) == QCheckBox:
            return float(guiobj.isChecked())


def main():
    """
    Creates QApplication and runs MainWindow().
    """
    # convert_convert_ui()
    app = QApplication([])
    print('Currently used style:', app.style().metaObject().className())
    print('Available styles:', QStyleFactory.keys())
    dbg.p(debug_print='App created', debug_from='Main')
    main_window = MainWindow()
    dbg.p(debug_print='Main Window created', debug_from='Main')
    main_window.show()
    main_window.after_show()
    main_window.tabSpectra.repaint()
    dbg.p(debug_print='Main Window shown', debug_from='Main')
    app.exec_()
    dbg.p(debug_print='App excuted', debug_from='Main')


if __name__ == '__main__':
    main()
