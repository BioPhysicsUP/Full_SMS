"""Module for analysis of SMS data from HDF5 files

Bertus van Heerden and Joshua Botha
University of Pretoria
2019
"""

__docformat__ = 'NumPy'

import ui.convert_ui as convert_ui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, pyqtSignal, QAbstractItemModel, \
    QModelIndex, Qt, QThreadPool, QRunnable, pyqtSlot, QItemSelectionModel, QSize
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from platform import system
import sys
import os
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
# from matplotlib.axes.subplots import Axes
import numpy as np
import scipy

import matplotlib as mpl

import random
# import matplotlib as mpl
# from matplotlib import figure as Figure
import dbg
import traceback
import smsh5
import tcspcfit
from smsh5 import start_at_nonzero

from ui.mainwindow import Ui_MainWindow
from ui.fitting_dialog import Ui_Dialog
from generate_sums import CPSums
from joblib import Parallel, delayed
import pyqtgraph as pg


class WorkerSignals(QObject):
    """ A QObject with attributes  of pyqtSignal's that can be used
    to communicate between worker threads and the main thread. """

    finished = pyqtSignal()
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

    add_irf = pyqtSignal(np.ndarray, np.ndarray)

    reset_gui = pyqtSignal()


class WorkerOpenFile(QRunnable):
    """ A QRunnable class to create a worker thread for opening h5 file. """

    # def __init__(self, fn, *args, **kwargs):
    def __init__(self, fname, openfile_func):
        """
        Initiate Open File Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to open a h5 file
        and populate the tree in the mainwindow gui.

        Parameters
        ----------
        fname : str
            The name of the file.
        openfile_func : function
            Function to be called that will read the h5 file and populate the tree on the gui.
        """

        super(WorkerOpenFile, self).__init__()
        self.openfile_func = openfile_func
        self.signals = WorkerSignals()
        self.fname = fname

    @pyqtSlot()
    def run(self) -> None:
        """ The code that will be run when the thread is started. """

        try:
            self.openfile_func(self.fname, self.signals)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


# TODO: These functions should probably be in a class since they share the signals.
def open_h5(fname, signals) -> None:
    """
    Read the selected h5 file and populates the tree on the gui with the file and the particles.

    Accepts a function that will be used to indicate the current progress.

    As this function is designed to be called from a thread other than the main one, no GUI code
    should be called here.

    Parameters
    ----------
    fname : str
        Path name to h5 file.
    start_progress_sig : pyqtSignal
        Used to call method to set up progress bar on GUI.
    progress_sig : pyqtSignal
        Used to call method to increment progress bar on GUI.
    status_sig : pyqtSignal
        Used to call method to show status bar message on GUI.
    """

    # print("Open_h5 called from thread")

    start_progress_sig = signals.start_progress
    auto_prog_sig = signals.auto_progress
    progress_sig = signals.progress
    status_sig = signals.status_message
    add_dataset_sig = signals.add_datasetindex
    add_node_sig = signals.add_particlenode
    reset_tree_sig = signals.reset_tree
    data_loaded_sig = signals.data_loaded
    bin_size_sig = signals.bin_size

    try:
        status_sig.emit("Opening file...")
        dataset = smsh5.H5dataset(fname[0], progress_sig, auto_prog_sig)
        bin_all(dataset, 100, start_progress_sig, progress_sig, status_sig, bin_size_sig)
        start_progress_sig.emit(dataset.numpart)
        status_sig.emit("Opening file: Building decay histograms...")
        dataset.makehistograms()

        datasetnode = DatasetTreeNode(fname[0][fname[0].rfind('/') + 1:-3], dataset, 'dataset')
        add_dataset_sig.emit(datasetnode)

        start_progress_sig.emit(dataset.numpart)
        status_sig.emit("Opening file: Adding particles...")
        for i, particle in enumerate(dataset.particles):
            particlenode = DatasetTreeNode(particle.name, particle, 'particle')
            add_node_sig.emit(particlenode, progress_sig, i)
            progress_sig.emit()
        reset_tree_sig.emit()
        status_sig.emit("Done")
        data_loaded_sig.emit()
    except Exception as exc:
        raise RuntimeError("h5 data file was not loaded successfully.") from exc


def open_irf(fname, signals) -> None:
    """
    Read the selected h5 file and populates the tree on the gui with the file and the particles.

    Accepts a function that will be used to indicate the current progress.

    As this function is designed to be called from a thread other than the main one, no GUI code
    should be called here.

    Parameters
    ----------
    fname : str
        Path name to h5 file.
    start_progress_sig : pyqtSignal
        Used to call method to set up progress bar on GUI.
    progress_sig : pyqtSignal
        Used to call method to increment progress bar on GUI.
    status_sig : pyqtSignal
        Used to call method to show status bar message on GUI.
    """

    # print("Open_h5 called from thread")
    # TODO: cleanup this function and the one above to remove duplicate code

    start_progress_sig = signals.start_progress
    auto_prog_sig = signals.auto_progress
    progress_sig = signals.progress
    status_sig = signals.status_message
    bin_size_sig = signals.bin_size
    add_irf_sig = signals.add_irf

    try:
        status_sig.emit("Opening file...")
        dataset = smsh5.H5dataset(fname[0], progress_sig, auto_prog_sig)
        bin_all(dataset, 100, start_progress_sig, progress_sig, status_sig, bin_size_sig)
        start_progress_sig.emit(dataset.numpart)
        status_sig.emit("Opening file: Building decay histograms...")
        dataset.makehistograms()

        # datasetnode = DatasetTreeNode(fname[0][fname[0].rfind('/') + 1:-3], dataset, 'dataset')
        # add_dataset_sig.emit(datasetnode)
        irfhist = dataset.particles[0].histogram
        add_irf_sig.emit(irfhist.decay, irfhist.t)

        start_progress_sig.emit(dataset.numpart)
        status_sig.emit("Opening file: Adding particles...")
        # for i, particle in enumerate(dataset.particles):
        #     particlenode = DatasetTreeNode(particle.name, particle, 'particle')
        #     add_node_sig.emit(particlenode, progress_sig, i)
        #     progress_sig.emit()
        # reset_tree_sig.emit()
        status_sig.emit("Done")
    except Exception as exc:
        raise RuntimeError("h5 data file was not loaded successfully.") from exc


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
    dataset.binints(bin_size, progress_sig)
    bin_size_sig.emit(bin_size)


def resolve_levels(start_progress_sig: pyqtSignal,
                   progress_sig: pyqtSignal, status_sig: pyqtSignal,
                   conf, data, currentparticle, resolve_all: bool = None,
                   resolve_selected=None) -> None:  #  parallel: bool = False
    """
    Resolves the levels in particles by finding the change points in the
    abstimes data of a Particle instance.

    If no parameter are given the current particle will be resolved. If
    the ``resolve_all`` parameter is given **all** the loaded particles
    will be resolved. If the ``resolve_selected`` parameter is provided
    the selection of particles will be resolved.

    Parameters
    ----------
    parallel : Bool, False
        If True, parallel is used.
    start_progress_sig : pyqtSignal
        Used to call method to set up progress bar on GUI.
    progress_sig : pyqtSignal
        Used to call method to increment progress bar on GUI.
    status_sig : pyqtSignal
        Used to call method to show status bar message on GUI.
    resolve_all : bool
        If True all the particle instances available will be resolved.
    resolve_selected : list[smsh5.Partilce]
        A list of Particle instances in smsh5, that isn't the current one, to be resolved.
    """

    print(currentparticle)
    assert not (resolve_all is not None and resolve_selected is not None), \
        "'resolve_all' and 'resolve_selected' can not both be given as parameters."

    if resolve_all is None and resolve_selected is None:  # Then resolve current
        currentparticle.cpts.run_cpa(confidence=conf / 100, run_levels=True)

    elif resolve_all is not None and resolve_selected is None:  # Then resolve all
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
            status_sig.emit('Ready...')
        except Exception as exc:
            raise RuntimeError("Couldn't resolve levels.") from exc
    elif resolve_selected is not None:  # Then resolve selected
        try:
            status_sig.emit('Resolving Selected Particle Levels...')
            start_progress_sig.emit(len(resolve_selected))
            for particle in resolve_selected:
                particle.cpts.run_cpa(confidence=conf, run_levels=True)
                progress_sig.emit()
            status_sig.emit('Ready...')
        except Exception as exc:
            raise RuntimeError("Couldn't resolve levels.") from exc


class WorkerBinAll(QRunnable):
    """ A QRunnable class to create a worker thread for binning all the data. """

    def __init__(self, dataset, binall_func, bin_size):
        """
        Initiate Open File Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to open a h5 file
        and populate the tree in the mainwindow gui.

        Parameters
        ----------
        fname : str
            The name of the file.
        binall_func : function
            Function to be called that will read the h5 file and populate the tree on the gui.
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
                             self.signals.progress, self.signals.status_message)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


class WorkerResolveLevels(QRunnable):
    """ A QRunnable class to create a worker thread for resolving levels. """

    def __init__(self, resolve_levels_func, conf, data, currentparticle,
                 resolve_all: bool = None,
                 resolve_selected=None) -> None:
        """
        Initiate Resolve Levels Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to resolve a single,
        the selected, or all the particles'.

        Parameters
        ----------
        resolve_levels_func : function
            The function that will be called to perform the resolving of the levels.
        resolve_all : bool, optional
            If true then all particle levels will be resolved.
        resolve_selected : list[smsh5.Particle], optional
            The provided instances of the class Particle in smsh5 will be resolved.
        """

        super(WorkerResolveLevels, self).__init__()
        self.signals = WorkerSignals()
        self.resolve_levels_func = resolve_levels_func
        self.resolve_all = resolve_all
        self.resolve_selected = resolve_selected
        self.conf = conf
        self.data = data
        self.currentparticle = currentparticle
        print(self.currentparticle)

    @pyqtSlot()
    def run(self) -> None:
        """ The code that will be run when the thread is started. """

        try:
            self.resolve_levels_func(self.signals.start_progress, self.signals.progress,
                                     self.signals.status_message, self.conf, self.data, self.currentparticle,
                                     self.resolve_all, self.resolve_selected)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


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
        # if progress_sig is not None:
        #     progress_sig.emit()  # Increment progress bar on MainWindow GUI
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


class MainWindow(QMainWindow):
    """
    Class for Full SMS application that returns QMainWindow object.

    This class uses a *.ui converted to a *.py script to generate gui. Be
    sure to run convert_ui.py after having made changes to mainwindow.ui.
    """

    def __init__(self):
        """Initialise MainWindow object.

        Creates and populates QMainWindow object as described by mainwindow.py
        as well as creates MplWidget
        """

        self.threadpool = QThreadPool()
        dbg.p("Multi-threading with maximum %d threads" % self.threadpool.maxThreadCount(), "Main")

        self.confidence_index = {
            0: 99,
            1: 95,
            2: 90,
            3: 69}

        if system() == "win32" or system() == "win64":
            dbg.p("System -> Windows", "Main")
        elif system() == "Darwin":
            dbg.p("System -> Unix/Linus", "Main")
        else:
            dbg.p("System -> Other", "Main")

        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.fitparamdialog = FittingDialog(self)
        # print(self.MW_Intensity.figure.get_dpi())

        self.ui.tabWidget.setCurrentIndex(0)

        self.setWindowTitle("Full SMS")

        self.ui.pgIntensity.getPlotItem().getAxis('left').setLabel('Intensity', 'counts/100ms')
        self.ui.pgIntensity.getPlotItem().getAxis('bottom').setLabel('Time', 's')
        self.ui.pgIntensity.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)

        self.ui.pgLifetime_Int.getPlotItem().getAxis('left').setLabel('Intensity', 'counts/100ms')
        self.ui.pgLifetime_Int.getPlotItem().getAxis('bottom').setLabel('Time', 's')
        # self.ui.pgLifetime_Int.getPlotItem().getViewBox()\
        #     .setYLink(self.ui.pgIntensity.getPlotItem().getAxis('left').getViewBox())
        # self.ui.pgLifetime_Int.getPlotItem().getViewBox()\
        #     .setXLink(self.ui.pgIntensity.getPlotItem().getAxis('bottom').getViewBox())
        self.ui.pgLifetime_Int.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)

        self.ui.pgLifetime.getPlotItem().getAxis('left').setLabel('Num. of occur.', 'counts/bin')
        self.ui.pgLifetime.getPlotItem().getAxis('bottom').setLabel('Decay time', 'ns')
        self.ui.pgLifetime.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)

        self.ui.pgGroups.getPlotItem().getAxis('left').setLabel('Intensity', 'counts/100ms')
        self.ui.pgGroups.getPlotItem().getAxis('bottom').setLabel('Time', 's')
        self.ui.pgGroups.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)

        self.ui.pgBIC.getPlotItem().getAxis('left').setLabel('BIC')
        self.ui.pgBIC.getPlotItem().getAxis('bottom').setLabel('Number of State')
        self.ui.pgBIC.getPlotItem().getViewBox().setLimits(xMin=0)

        self.ui.pgSpectra.getPlotItem().getAxis('left').setLabel('X Range', 'um')
        self.ui.pgSpectra.getPlotItem().getAxis('bottom').setLabel('Y Range', '<span>&#181;</span>m')
        self.ui.pgSpectra.getPlotItem().getViewBox().setAspectLocked(lock=True, ratio=1)
        self.ui.pgLifetime_Int.getPlotItem().getViewBox().setLimits(xMin=0, yMin=0)

        plots = [self.ui.pgIntensity, self.ui.pgLifetime_Int, self.ui.pgLifetime,
                 self.ui.pgGroups, self.ui.pgBIC, self.ui.pgSpectra]
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
            if plot == self.ui.pgLifetime_Int:
                font.setPointSize(8)
            elif plot == self.ui.pgGroups:
                font.setPointSize(10)
            else:
                font.setPointSize(12)
            plot_item.getAxis('left').label.setFont(font)
            plot_item.getAxis('bottom').label.setFont(font)

            plot.setAntialiasing(True)

        # Connect all GUI buttons with outside class functions
        self.ui.btnApplyBin.clicked.connect(self.gui_apply_bin)
        self.ui.btnApplyBinAll.clicked.connect(self.gui_apply_bin_all)
        self.ui.btnResolve.clicked.connect(self.gui_resolve)
        self.ui.btnResolve_Selected.clicked.connect(self.gui_resolve_selected)
        self.ui.btnResolveAll.clicked.connect(self.gui_resolve_all)
        self.ui.btnPrevLevel.clicked.connect(self.gui_prev_lev)
        self.ui.btnNextLevel.clicked.connect(self.gui_next_lev)
        self.ui.btnLoadIRF.clicked.connect(self.gui_load_irf)
        self.ui.btnFitParameters.clicked.connect(self.gui_fit_param)
        self.ui.btnFit.clicked.connect(self.gui_fit_current)
        self.ui.btnFitSelected.clicked.connect(self.gui_fit_selected)
        self.ui.btnFitAll.clicked.connect(self.gui_fit_all)
        self.ui.btnSubBackground.clicked.connect(self.gui_sub_bkg)
        self.ui.actionOpen_h5.triggered.connect(self.act_open_h5)
        self.ui.actionOpen_pt3.triggered.connect(self.act_open_pt3)
        self.ui.actionTrim_Dead_Traces.triggered.connect(self.act_trim)
        self.ui.actionSwitch_All.triggered.connect(self.act_switch_all)
        self.ui.actionSwitch_Selected.triggered.connect(self.act_switch_selected)

        # Create and connect model for dataset tree
        self.treemodel = DatasetTreeModel()
        self.ui.treeViewParticles.setModel(self.treemodel)
        # Connect the tree selection to data display
        self.ui.treeViewParticles.selectionModel().currentChanged.connect(self.display_data)

        self.part_nodes = dict()

        self.tauparam = None
        self.ampparam = None
        self.shift = None
        self.decaybg = None
        self.irfbg = None
        self.start = None
        self.end = None
        self.addopt = None
        self.fitparam = FittingParameters(self)

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

        self.current_level = None

        self.ui.tabWidget.currentChanged.connect(self.tab_change)

        self.reset_gui()
        self.repaint()

    """#######################################
    ######## GUI Housekeeping Methods ########
    #######################################"""

    def after_show(self):
        self.ui.pgSpectra.resize(self.ui.tabSpectra.size().height(),
                                 self.ui.tabSpectra.size().height()-self.ui.btnSubBackground.size().height()-40)

    def resizeEvent(self, a0: QResizeEvent):
        if self.ui.tabSpectra.size().height() <= self.ui.tabSpectra.size().width():
            self.ui.pgSpectra.resize(self.ui.tabSpectra.size().height(),
                                     self.ui.tabSpectra.size().height()-self.ui.btnSubBackground.size().height()-40)
        else:
            self.ui.pgSpectra.resize(self.ui.tabSpectra.size().width(),
                                     self.ui.tabSpectra.size().width()-40)


    def check_all_sums(self) -> None:
        """
        Check if the all_sums.pickle file exists, and if it doesn't creates it
        """
        if (not os.path.exists(os.getcwd()+'\\all_sums.pickle')) and\
                (not os.path.isfile(os.getcwd()+'\\all_sums.pickle')):
            self.status_message('Calculating change point sums, this may take several minutes.')
            create_all_sums = CPSums(only_pickle=True, n_min=10, n_max=1000)
            del create_all_sums
            self.status_message('Ready...')

    def get_bin(self) -> int:
        """ Returns current GUI value for bin size in ms.

        Returns
        -------
        int
            The value of the bin size on the GUI in spbBinSize.
        """

        return self.ui.spbBinSize.value()

    def set_bin(self, new_bin: int):
        """ Sets the GUI value for the bin size in ms

        Parameters
        ----------
        new_bin: int
            Value to set bin size to, in ms.
        """
        self.ui.spbBinSize.setValue(new_bin)

    def get_gui_confidence(self):
        """ Return current GUI value for confidence percentage. """

        return [self.ui.cmbConfIndex.currentIndex(), self.confidence_index[self.ui.cmbConfIndex.currentIndex()]]

    def gui_apply_bin(self):
        """ Changes the bin size of the data of the current particle and then displays the new trace. """

        # self.ui.pgSpectra.centralWidget
        #
        # self.ui.pgIntensity.getPlotItem().setFixedWidth(500)
        # self.ui.pgSpectra.resize(100, 200)
        # self.ui.pgIntensity.getPlotItem().getAxis('left').setRange(0, 100)
        # window_color = self.palette().color(QPalette.Window)
        # rgba_color = (window_color.red()/255, window_color.green()/255, window_color.blue()/255, 1)
        # self.ui.pgIntensity.setBackground(background=rgba_color)
        # self.ui.pgIntensity.setXRange(0, 10, 0)
        # self.ui.pgIntensity.getPlotItem().plot(y=[1, 2, 3, 4, 5])
        try:
            self.currentparticle.binints(self.get_bin())
        except Exception as err:
            print('Error Occured:' + str(err))
        else:
            self.display_data()
            self.repaint()
            dbg.p('Single trace binned', 'Main')

    def gui_apply_bin_all(self):
        """ Changes the bin size of the data of all the particles and then displays the new trace of the current particle. """

        try:
            self.start_binall_thread(self.get_bin())
        except Exception as err:
            print('Error Occured:' + str(err))
        else:
            self.plot_trace()
            self.repaint()
            dbg.p('All traces binned', 'Main')

    def gui_resolve(self):
        """ Resolves the levels of the current particle and displays it. """

        self.start_resolve_thread(mode='current')

    def gui_resolve_selected(self):
        """ Resolves the levels of the selected particles and displays the levels of the current particle. """

        self.start_resolve_thread(mode='selected')

    def gui_resolve_all(self):
        """ Resolves the levels of the all the particles and then displays the levels of the current particle. """

        self.start_resolve_thread(mode='all')

    def gui_prev_lev(self):
        """ Moves to the previous resolves level and displays its decay curve. """

        if self.current_level is None:
            pass
        elif self.current_level == 0:
            self.current_level = None
        else:
            self.current_level -= 1
        self.plot_levels()
        self.plot_decay()

    def gui_next_lev(self):
        """ Moves to the next resolves level and displays its decay curve. """

        if self.current_level is None:
            self.current_level = 0
        else:
            self.current_level += 1
        self.plot_levels()
        self.plot_decay()

    def gui_load_irf(self):
        """ Allow the user to load a IRF instead of the IRF that has already been loaded. """

        fname = QFileDialog.getOpenFileName(self, 'Open HDF5 file', '', "HDF5 files (*.h5)")
        if fname != ('', ''):  # fname will equal ('', '') if the user canceled.
            of_worker = WorkerOpenFile(fname, open_irf)
            of_worker.signals.finished.connect(self.open_file_thread_complete)
            of_worker.signals.start_progress.connect(self.start_progress)
            of_worker.signals.progress.connect(self.update_progress)
            of_worker.signals.auto_progress.connect(self.update_progress)
            of_worker.signals.start_progress.connect(self.start_progress)
            of_worker.signals.status_message.connect(self.status_message)
            of_worker.signals.add_datasetindex.connect(self.add_dataset)
            of_worker.signals.add_particlenode.connect(self.add_node)
            of_worker.signals.reset_tree.connect(lambda: self.treemodel.modelReset.emit())
            of_worker.signals.data_loaded.connect(self.set_data_loaded)
            of_worker.signals.bin_size.connect(self.ui.spbBinSize.setValue)
            of_worker.signals.add_irf.connect(self.add_irf)

            self.threadpool.start(of_worker)

    def gui_fit_param(self):
        """ Opens a dialog to choose the setting with which the decay curve will be fitted. """

        print("gui_fit_param")
        if self.fitparamdialog.exec():
            self.fitparam.getfromdialog()

    def gui_fit_current(self):
        """ Fits the all the levels decay curves in the current particle using the provided settings. """

        print("gui_fit_current")
        try:
            if not self.currentparticle.histogram.fit(self.fitparam.numexp, self.fitparam.tau, self.fitparam.amp,
                                                      self.fitparam.shift, self.fitparam.decaybg, self.fitparam.irfbg,
                                                      self.fitparam.start, self.fitparam.end, self.fitparam.addopt,
                                                      self.fitparam.irf):
                return  # fit unsuccessful
        except AttributeError:
            raise
            print("No decay")
        else:
            self.plot_convd()
            self.update_results()

    def update_results(self):
        tau = self.currentparticle.histogram.tau
        amp = self.currentparticle.histogram.amp
        shift = self.currentparticle.histogram.shift
        bg = self.currentparticle.histogram.bg
        irfbg = self.currentparticle.histogram.irfbg
        try:
            taustring = 'Tau = ' + ' '.join('{:#.3g} ns'.format(F) for F in tau)
            ampstring = 'Amp = ' + ' '.join('{:#.3g} '.format(F) for F in amp)
        except TypeError:  # only one component
            taustring = 'Tau = {:#.3g} ns'.format(tau)
            ampstring = 'Amp = {:#.3g}'.format(amp)
        shiftstring = 'Shift = {:#.3g} ns'.format(shift)
        bgstring = 'Decay BG = {:#.3g}'.format(bg)
        irfbgstring = 'IRF BG = {:#.3g}'.format(irfbg)
        self.ui.textBrowser.setText(taustring + '\n' + ampstring + '\n' + shiftstring + '\n' + bgstring + '\n' +
                                    irfbgstring)

    def gui_fit_selected(self):
        """ Fits the all the levels decay curves in the all the selected particles using the provided settings. """

        print("gui_fit_selected")

    def gui_fit_all(self):
        """ Fits the all the levels decay curves in the all the particles using the provided settings. """

        print("gui_fit_all")

    def gui_sub_bkg(self):
        """ Used to subtract the background TODO: Explain the sub_background """

        print("gui_sub_bkg")

    def act_open_h5(self):
        """ Allows the user to point to a h5 file and then starts a thread that reads and loads the file. """

        fname = QFileDialog.getOpenFileName(self, 'Open HDF5 file', '', "HDF5 files (*.h5)")
        if fname != ('', ''):  # fname will equal ('', '') if the user canceled.
            of_worker = WorkerOpenFile(fname, open_h5)
            of_worker.signals.finished.connect(self.open_file_thread_complete)
            of_worker.signals.start_progress.connect(self.start_progress)
            of_worker.signals.progress.connect(self.update_progress)
            of_worker.signals.auto_progress.connect(self.update_progress)
            of_worker.signals.start_progress.connect(self.start_progress)
            of_worker.signals.status_message.connect(self.status_message)
            of_worker.signals.add_datasetindex.connect(self.add_dataset)
            of_worker.signals.add_particlenode.connect(self.add_node)
            of_worker.signals.reset_tree.connect(lambda: self.treemodel.modelReset.emit())
            of_worker.signals.data_loaded.connect(self.set_data_loaded)
            of_worker.signals.bin_size.connect(self.ui.spbBinSize.setValue)

            self.threadpool.start(of_worker)

    def act_open_pt3(self):
        """ Allows a user to load a group of .pt3 files that are in a folder and loads them. NOT YET IMPLEMENTED. """

        print("act_open_pt3")

    def act_trim(self):
        """ Used to trim the 'dead' part of a trace as defined by two parameters. """

        print("act_trim")

    def act_switch_all(self):

        self.switching_frequency(all_selected='all')

    def act_switch_selected(self):

        self.switching_frequency(all_selected='selected')

    """#######################################
    ############ Internal Methods ############
    #######################################"""

    def add_irf(self, decay, t):

        self.fitparam.irf = decay
        self.fitparam.irft = t
        self.irf_loaded = True
        self.reset_gui

    def add_dataset(self, datasetnode):

        self.datasetindex = self.treemodel.addChild(datasetnode)

    def add_node(self, particlenode, progress_sig, i):

        index = self.treemodel.addChild(particlenode, self.datasetindex, progress_sig)
        self.part_nodes[i] = particlenode

    def tab_change(self, active_tab_index:int):
        if self.data_loaded and hasattr(self, 'currentparticle'):
            self.display_data()

    def display_data(self, current=None, prev=None) -> None:  # TODO: What is previous for?
        """ Displays the intensity trace and the histogram of the current particle.

        Parameters
        ----------
        current : QtCore.QModelIndex
            The index of the current selected particle as defined by QtCore.QModelIndex.
        prev : QtCore.QModelIndex
            The index of the previous selected particle as defined by QtCore.QModelIndex.
        """

        self.current_level = None
        self.current_ind = current
        self.pre_ind = prev
        if current is not None:
            self.currentparticle = self.treemodel.get_particle(current)
        if type(self.currentparticle) is smsh5.Particle:
            self.set_bin(self.currentparticle.bin_size)
            self.plot_trace()
            if self.currentparticle.has_levels:
                print('bla')
                self.plot_levels()
                self.ui.btnGroup.setEnabled(True)
                self.ui.btnGroup_Selected.setEnabled(True)
                self.ui.btnGroup_All.setEnabled(True)
            else:
                self.ui.btnGroup.setEnabled(False)
                self.ui.btnGroup_Selected.setEnabled(False)
                self.ui.btnGroup_All.setEnabled(False)
            self.plot_decay(remove_empty=True)
            dbg.p('Current data displayed', 'Main')

    def plot_decay(self, remove_empty: bool = False) -> None:
        """ Used to display the histogram of the decay data of the current particle. """

        if self.current_level is None:
            try:
                decay = self.currentparticle.histogram.decay
                t = self.currentparticle.histogram.t

            except AttributeError:
                dbg.p(debug_print='No Decay!', debug_from='Main')
                return
        else:
            try:
                decay = self.currentparticle.levels[self.current_level].histogram.decay
                t = self.currentparticle.levels[self.current_level].histogram.t
            except ValueError:
                return

        if self.ui.tabWidget.currentWidget().objectName() == 'tabLifetime':
            plot_item = self.ui.pgLifetime.getPlotItem()
            plot_pen = QPen()
            plot_pen.setWidthF(1.5)
            plot_pen.setJoinStyle(Qt.RoundJoin)
            plot_pen.setColor(QColor('blue'))
            plot_pen.setCosmetic(True)

            if remove_empty:
                first = (decay > 4).argmax(axis=0)
                t = t[first:-1] - t[first]
                decay = decay[first:-1]

            plot_item.clear()
            plot_item.plot(x=t, y=decay, pen=plot_pen, symbol=None)
            unit = 'ns with ' + str(self.currentparticle.channelwidth) + 'ns bins'
            plot_item.getAxis('bottom').setLabel('Decay time', unit)
            plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=t[-1])

    def plot_trace(self) -> None:
        """ Used to display the trace from the absolute arrival time data of the current particle. """

        try:
            # self.currentparticle = self.treemodel.data(self.current_ind, Qt.UserRole)
            trace = self.currentparticle.binnedtrace.intdata
            times = self.currentparticle.binnedtrace.inttimes / 1E3
        except AttributeError:
            print('No trace!')
        else:
            plot_pen = QPen()
            plot_pen.setCosmetic(True)
            cur_tab_name = self.ui.tabWidget.currentWidget().objectName()
            if cur_tab_name != 'tabSpectra':
                if cur_tab_name == 'tabIntensity':
                    plot_item = self.ui.pgIntensity.getPlotItem()
                    plot_pen.setWidthF(1.5)
                    plot_pen.setColor(QColor('green'))
                elif cur_tab_name == 'tabLifetime':
                    plot_item = self.ui.pgLifetime_Int.getPlotItem()
                    plot_pen.setWidthF(1.1)
                    plot_pen.setColor(QColor('green'))
                elif cur_tab_name == 'tabGrouping':
                    plot_item = self.ui.pgGroups
                    plot_pen.setWidthF(1.1)
                    plot_pen.setColor(QColor(0, 0, 0, 50))

                plot_pen.setJoinStyle(Qt.RoundJoin)

                plot_item.clear()
                unit = 'counts/'+str(self.get_bin())+'ms'
                plot_item.getAxis('left').setLabel(text='Intensity', units=unit)
                plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=times[-1])
                plot_item.plot(x=times, y=trace, pen=plot_pen, symbol=None)

    def plot_levels(self):
        """ Used to plot the resolved intensity levels of the current particle. """
        try:
            # self.currentparticle = self.treemodel.data(self.current_ind, Qt.UserRole)
            level_ints, times = self.currentparticle.levels2data()
            level_ints = level_ints*self.get_bin()/1E3
            print(level_ints, times)

        except AttributeError:
            print('No levels!')
        else:
            # if self.ui.tabIntensity.isActiveWindow():
            #     plot_item = self.ui.pgIntensity.getPlotItem()
            #     print('int')
            # elif self.ui.tabLifetime.isActiveWindow():
            #     print('life')
            #     plot_item = self.ui.pgLifetime_Int.getPlotItem()
            # else:
            #     return
            if self.ui.tabWidget.currentWidget().objectName() == 'tabIntensity':
                plot_item = self.ui.pgIntensity.getPlotItem()
                # pen_width = 1.5
            elif self.ui.tabWidget.currentWidget().objectName() == 'tabLifetime':
                plot_item = self.ui.pgLifetime_Int.getPlotItem()
                # pen_width = 1.1
            else:
                return

        plot_pen = QPen()
        plot_pen.setWidthF(2)
        plot_pen.brush()
        plot_pen.setJoinStyle(Qt.RoundJoin)
        plot_pen.setColor(QColor('black'))
        plot_pen.setCosmetic(True)

        plot_item.plot(x=times, y=level_ints, pen=plot_pen, symbol=None)

        if self.current_level is not None:
            current_ints, current_times = self.currentparticle.current2data(self.current_level)
            current_ints = current_ints*self.get_bin()/1E3
            print(current_ints, current_times)

            if not (current_ints[0] == np.inf or current_ints[1] == np.inf):
                plot_pen.setColor(QColor('red'))
                plot_pen.setWidthF(3)
                plot_item.plot(x=current_times, y=current_ints, pen=plot_pen, symbol=None)
            else:
                print('infinity in level')

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

    def open_h5(self, fname, start_progress_sig, auto_prog_sig, progress_sig, status_sig) -> None:
        """
        Read the selected h5 file and populates the tree on the gui with the file and the particles.

        Accepts a function that will be used to indicate the current progress.

        Parameters
        ----------
        fname : str
            Path name to h5 file.
        start_progress_sig : pyqtSignal
            Used to call method to set up progress bar on GUI.
        progress_sig : pyqtSignal
            Used to call method to increment progress bar on GUI.
        status_sig : pyqtSignal
            Used to call method to show status bar message on GUI.
        """

        # print("Open_h5 called from thread")
        # TODO: Remove this method as it is not needed
        try:
            status_sig.emit("Opening file...")
            dataset = smsh5.H5dataset(fname[0], progress_sig, auto_prog_sig)
            self.bin_all(dataset, 100, start_progress_sig, progress_sig, status_sig)
            start_progress_sig.emit(dataset.numpart)
            status_sig.emit("Opening file: Building decay histograms...")
            dataset.makehistograms()

            datasetnode = DatasetTreeNode(fname[0][fname[0].rfind('/') + 1:-3], dataset, 'dataset')
            self.datasetindex = self.treemodel.addChild(datasetnode)
            # print(datasetindex)

            start_progress_sig.emit(dataset.numpart)
            status_sig.emit("Opening file: Adding particles...")
            self.part_nodes = dict()
            for i, particle in enumerate(dataset.particles):
                particlenode = DatasetTreeNode(particle.name, particle, 'particle')
                index = self.treemodel.addChild(particlenode, self.datasetindex, progress_sig)
                self.part_nodes[i] = particlenode
                # self.treemodel.index(index, self.datasetindex)
            self.treemodel.modelReset.emit()
            status_sig.emit("Done")
            self.data_loaded = True
        except Exception as exc:
            raise RuntimeError("h5 data file was not loaded successfully.") from exc

    def set_data_loaded(self):
        self.data_loaded = True

    def open_file_thread_complete(self) -> None:
        """ Is called as soon as all of the threads have finished. """

        if self.data_loaded:
            msgbx = QMessageBox()
            msgbx.setIcon(QMessageBox.Question)
            msgbx.setText("Resolve Levels Now?")
            msgbx.setInformativeText("Would you like to resolve the levels now?")
            msgbx.setWindowTitle("Resolve Levels?")
            msgbx.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
            msgbx.setDefaultButton(QMessageBox.Yes)
            if msgbx.exec() == QMessageBox.Yes:
                confidences = ("0.99", "0.95", "0.90", "0.69")
                item, ok = QInputDialog.getItem(self, "Choose Confidence",
                                                "Select confidence interval to use.", confidences, 1, False)
                if ok:
                    index = list(self.confidence_index.values()).index(int(float(item) * 100))
                    self.ui.cmbConfIndex.setCurrentIndex(index)
                    self.start_resolve_thread('all')
        self.reset_gui()
        dbg.p('File opened', 'Main')


    def start_binall_thread(self, bin_size) -> None:
        """

        Parameters
        ----------
        bin_size
        """

        dataset = self.treemodel.data(self.datasetindex, Qt.UserRole)

        binall_thread = WorkerBinAll(dataset, self.bin_all, bin_size)
        binall_thread.signals.finished.connect(self.binall_thread_complete)
        binall_thread.signals.start_progress.connect(self.start_progress)
        binall_thread.signals.progress.connect(self.update_progress)
        binall_thread.signals.status_message.connect(self.status_message)

        self.threadpool.start(binall_thread)

    def bin_all(self, dataset, bin_size, start_progress_sig, progress_sig, status_sig) -> None:
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
        if not self.data_loaded:
            part = "Opening file: "
        else:
            part = ""
        status_sig.emit(part + "Binning traces...")
        dataset.binints(bin_size, progress_sig)
        self.ui.spbBinSize.setValue(bin_size)

    def binall_thread_complete(self):

        self.status_message('Done')
        self.plot_trace()
        dbg.p('Binnig all levels complete', 'BinAll Thread')

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

        _, conf = self.get_gui_confidence()
        data = self.tree2dataset()
        currentparticle = self.currentparticle
        print(currentparticle)

        if mode == 'current':
            # sig = WorkerSignals()
            # self.resolve_levels(sig.start_progress, sig.progress, sig.status_message)
            resolve_thread = WorkerResolveLevels(resolve_levels, conf, data, currentparticle)
        elif mode == 'selected':
            resolve_thread = WorkerResolveLevels(resolve_levels, conf, data, currentparticle, resolve_selected=self.get_checked_particles())
        elif mode == 'all':
            resolve_thread = WorkerResolveLevels(resolve_levels, conf, data, currentparticle, resolve_all=True)
            # resolve_thread.signals.finished.connect(thread_finished)
            # resolve_thread.signals.start_progress.connect(self.start_progress)
            # resolve_thread.signals.progress.connect(self.update_progress)
            # resolve_thread.signals.status_message.connect(self.status_message)
            # self.resolve_levels(resolve_thread.signals.start_progress, resolve_thread.signals.progress,
            #                     resolve_thread.signals.status_message, resolve_all=True, parallel=True)

        resolve_thread.signals.finished.connect(thread_finished)
        resolve_thread.signals.start_progress.connect(self.start_progress)
        resolve_thread.signals.progress.connect(self.update_progress)
        resolve_thread.signals.status_message.connect(self.status_message)
        resolve_thread.signals.reset_gui.connect(self.reset_gui)

        self.threadpool.start(resolve_thread)

    # @dbg.profile
    def resolve_levels(self, start_progress_sig: pyqtSignal,
                       progress_sig: pyqtSignal, status_sig: pyqtSignal,
                       reset_gui_sig: pyqtSignal, resolve_all: bool = None,
                       resolve_selected=None) -> None:  #  parallel: bool = False
        """
        Resolves the levels in particles by finding the change points in the
        abstimes data of a Particle instance.

        If no parameter are given the current particle will be resolved. If
        the ``resolve_all`` parameter is given **all** the loaded particles
        will be resolved. If the ``resolve_selected`` parameter is provided
        the selection of particles will be resolved.
        """
        assert not (resolve_all is not None and resolve_selected is not None), \
            "'resolve_all' and 'resolve_selected' can not both be given as parameters."

        data = self.tree2dataset()
        if resolve_all is None and resolve_selected is None:  # Then resolve current
            _, conf = self.get_gui_confidence()
            self.currentparticle.cpts.run_cpa(confidence=conf / 100, run_levels=True)
            print('resolved')

        elif resolve_all is not None and resolve_selected is None:  # Then resolve all
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
                status_sig.emit('Ready...')
            except Exception as exc:
                raise RuntimeError("Couldn't resolve levels.") from exc
        elif resolve_selected is not None:  # Then resolve selected
            try:
                _, conf = self.get_gui_confidence()
                status_sig.emit('Resolving Selected Particle Levels...')
                start_progress_sig.emit(len(resolve_selected))
                for particle in resolve_selected:
                    particle.cpts.run_cpa(confidence=conf, run_levels=True)
                    progress_sig.emit()
                status_sig.emit('Ready...')
            except Exception as exc:
                raise RuntimeError("Couldn't resolve levels.") from exc

        # print('bla')
        self.level_resolved = True
        data.makehistograms()
        reset_gui_sig.emit()

    def run_parallel_cpa(self, particle):
        particle.cpts.run_cpa(confidence=self.conf_parallel, run_levels=True)

    def resolve_thread_complete(self):
        if self.tree2dataset().cpa_has_run:
            self.ui.tabGrouping.setEnabled(True)
        if self.ui.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
            self.display_data()
        dbg.p('Resolving levels complete', 'Resolve Thread')

        ###############################################################################################################
        self.currentparticle.ahca.run_grouping()
        ###############################################################################################################

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
            print('Switching frequency analysis failed: ' + exc)
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
                checked_nums.append(ind+1)
        return checked_nums

    def get_checked_particles(self):
        checked_particles = list()
        for ind in range(self.treemodel.rowCount(self.datasetindex)):
            if self.part_nodes[ind].checked():
                checked_particles.append(self.tree2particle(ind))
        return checked_particles

    def reset_gui(self):
        """ Sets the GUI elements to enabled if it should be accessible. """

        print('reset')
        if self.data_loaded:
            enabled = True
        else:
            enabled = False

        # Intensity
        self.ui.tabIntensity.setEnabled(enabled)
        self.ui.btnApplyBin.setEnabled(enabled)
        self.ui.btnApplyBinAll.setEnabled(enabled)
        self.ui.btnResolve.setEnabled(enabled)
        self.ui.btnResolve_Selected.setEnabled(enabled)
        self.ui.btnResolveAll.setEnabled(enabled)
        self.ui.cmbConfIndex.setEnabled(enabled)
        self.ui.spbBinSize.setEnabled(enabled)
        self.ui.actionReset_Analysis.setEnabled(enabled)
        if enabled:
            enable_levels = self.level_resolved
        else:
            enable_levels = enabled
        self.ui.actionTrim_Dead_Traces.setEnabled(enable_levels)

        # Lifetime
        self.ui.tabLifetime.setEnabled(enabled)
        self.ui.btnFitParameters.setEnabled(enabled)
        self.ui.btnLoadIRF.setEnabled(enabled)
        if enabled:
            enable_fitting = self.irf_loaded
        else:
            enable_fitting = enabled
        self.ui.btnFit.setEnabled(enable_fitting)
        self.ui.btnFitAll.setEnabled(enable_fitting)
        self.ui.btnFitSelected.setEnabled(enable_fitting)
        self.ui.btnNextLevel.setEnabled(enable_levels)
        self.ui.btnPrevLevel.setEnabled(enable_levels)
        print(enable_levels)

        # Spectral
        if self.has_spectra:
            self.ui.tabSpectra.setEnabled(True)
            self.ui.btnSubBackground.setEnabled(enabled)
        else:
            self.ui.tabSpectra.setEnabled(False)


class FittingDialog(QDialog, Ui_Dialog):
    def __init__(self, parent):
        self.parent = parent
        QDialog.__init__(self, parent)
        self.setupUi(self)
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self.updateplot)
        for widget in self.findChildren(QCheckBox):
            widget.stateChanged.connect(self.updateplot)
        for widget in self.findChildren(QComboBox):
            widget.currentTextChanged.connect(self.updateplot)

        self.lineStartTime.setValidator(QIntValidator())
        self.lineEndTime.setValidator(QIntValidator())

    def updateplot(self, *args):

        try:
            model = self.make_model()
        except Exception as err:
            dbg.p(debug_print='Error Occured:' + str(err), debug_from='Fitting Parameters')
            return

        fp = self.parent.fitparam
        try:
            irf = fp.irf
            irft = fp.irft
        except AttributeError:
            dbg.p(debug_print='No IRF!', debug_from='Fitting Parameters')
            return

        shift, decaybg, irfbg, start, end = self.getparams()

        irf = tcspcfit.colorshift(irf, shift)
        convd = scipy.signal.convolve(irf, model)
        convd = convd[:np.size(irf)]
        convd = convd / convd.max()

        try:
            decay = self.parent.currentparticle.histogram.decay
            decay = decay / decay.max()
            t = self.parent.currentparticle.histogram.t

            decay, t = start_at_nonzero(decay, t)
            end = min(end, np.size(t) - 1)  # Make sure endpoint is not bigger than size of t

            convd = convd[irft > 0]
            irft = irft[irft > 0]

        except AttributeError:
            dbg.p(debug_print='No Decay!', debug_from='Fitting Parameters')
        else:
            self.MW_fitparam.axes.clear()
            self.MW_fitparam.axes.semilogy(t, decay, color='xkcd:dull blue')
            self.MW_fitparam.axes.semilogy(irft, convd, color='xkcd:marine blue', linewidth=2)
            self.MW_fitparam.axes.set_ylim(bottom=1e-3)

            self.MW_fitparam.axes.axvline(t[start])
            self.MW_fitparam.axes.axvline(t[end])

            self.MW_fitparam.draw()

    def getparams(self):
        fp = self.parent.fitparam
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
        fp = self.parent.fitparam
        t = self.parent.currentparticle.histogram.t
        fp.getfromdialog()
        if fp.numexp == 1:
            tau = fp.tau[0][0]
            model = np.exp(-t / tau)
        elif fp.numexp == 2:
            tau1 = fp.tau[0][0]
            tau2 = fp.tau[1][0]
            amp1 = fp.amp[0][0]
            amp2 = fp.amp[1][0]
            print(amp1, amp2, tau1, tau2)
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
        self.decaybg = None
        self.irfbg = None
        self.start = None
        self.end = None
        self.numexp = None
        self.addopt = None

    def getfromdialog(self):
        self.numexp = int(self.fpd.combNumExp.currentText())
        if self.numexp == 1:
            self.tau = [[self.get_from_gui(i) for i in [self.fpd.line1Init, self.fpd.line1Min, self.fpd.line1Max, self.fpd.check1Fix]]]
            self.amp = [[self.get_from_gui(i) for i in [self.fpd.line1AmpInit, self.fpd.line1AmpMin, self.fpd.line1AmpMax, self.fpd.check1AmpFix]]]

        elif self.numexp == 2:
            self.tau = [[self.get_from_gui(i) for i in [self.fpd.line2Init1, self.fpd.line2Min1, self.fpd.line2Max1, self.fpd.check2Fix1]],
                        [self.get_from_gui(i) for i in [self.fpd.line2Init2, self.fpd.line2Min2, self.fpd.line2Max2, self.fpd.check2Fix2]]]
            self.amp = [[self.get_from_gui(i) for i in [self.fpd.line2AmpInit1, self.fpd.line2AmpMin1, self.fpd.line2AmpMax1, self.fpd.check2AmpFix1]],
                        [self.get_from_gui(i) for i in [self.fpd.line2AmpInit2, self.fpd.line2AmpMin2, self.fpd.line2AmpMax2, self.fpd.check2AmpFix2]]]

        elif self.numexp == 3:
            self.tau = [[self.get_from_gui(i) for i in [self.fpd.line3Init1, self.fpd.line3Min1, self.fpd.line3Max1, self.fpd.check3Fix1]],
                        [self.get_from_gui(i) for i in [self.fpd.line3Init2, self.fpd.line3Min2, self.fpd.line3Max2, self.fpd.check3Fix2]],
                        [self.get_from_gui(i) for i in [self.fpd.line3Init3, self.fpd.line3Min3, self.fpd.line3Max3, self.fpd.check3Fix3]]]
            self.amp = [[self.get_from_gui(i) for i in [self.fpd.line3AmpInit1, self.fpd.line3AmpMin1, self.fpd.line3AmpMax1, self.fpd.check3AmpFix1]],
                        [self.get_from_gui(i) for i in [self.fpd.line3AmpInit2, self.fpd.line3AmpMin2, self.fpd.line3AmpMax2, self.fpd.check3AmpFix2]],
                        [self.get_from_gui(i) for i in [self.fpd.line3AmpInit3, self.fpd.line3AmpMin3, self.fpd.line3AmpMax3, self.fpd.check3AmpFix3]]]

        self.shift = self.get_from_gui(self.fpd.lineShift)
        self.decaybg = self.get_from_gui(self.fpd.lineDecayBG)
        self.irfbg = self.get_from_gui(self.fpd.lineIRFBG)
        try:
            self.start = int(self.get_from_gui(self.fpd.lineStartTime))
        except TypeError:
            self.start = self.get_from_gui(self.fpd.lineStartTime)
        try:
            self.end = int(self.get_from_gui(self.fpd.lineEndTime))
        except TypeError:
            self.end = self.get_from_gui(self.fpd.lineEndTime)

        self.addopt = self.get_from_gui(self.fpd.lineAddOpt)

    @staticmethod
    def get_from_gui(guiobj):
        if type(guiobj) == QLineEdit:
            if guiobj.text() == '':
                return None
            else:
                return float(guiobj.text())
        elif type(guiobj) == QCheckBox:
            return float(guiobj.isChecked())


# class DatasetTreeNode():
#     def __init__(self, name, dataobj, datatype):
#         self._data = name
#         if type(name) == tuple:
#             self._data = list(name)
#         if type(name) in (str, bytes) or not hasattr(name, '__getitem__'):
#             self._data = [name]
#


def main():
    """
    Creates QApplication and runs MainWindow().
    """
    convert_ui.convert_ui()
    app = QApplication([])
    dbg.p(debug_print='App created', debug_from='Main')
    main_window = MainWindow()
    dbg.p(debug_print='Main Window created', debug_from='Main')
    main_window.show()
    main_window.after_show()
    main_window.ui.tabSpectra.repaint()
    dbg.p(debug_print='Main Window shown', debug_from='Main')
    app.exec_()
    dbg.p(debug_print='App excuted', debug_from='Main')


if __name__ == '__main__':
    main()
