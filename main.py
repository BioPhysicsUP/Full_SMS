"""Module for analysis of SMS data from HDF5 files

Bertus van Heerden and Joshua Botha
University of Pretoria
2019
"""

__docformat__ = 'NumPy'

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, pyqtSignal, QAbstractItemModel, \
    QModelIndex, Qt, QThreadPool, QRunnable, pyqtSlot, QItemSelectionModel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from platform import system
import sys
import os
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Axes
import numpy as np
import random
import matplotlib as mpl
from matplotlib import figure as Figure
import dbg
import traceback
import smsh5
from ui.mainwindow import Ui_MainWindow
from generate_sums import CPSums
from joblib import Parallel, delayed


# mpl.use("Qt5Agg")


# Default settings for matplotlib plots
# *************************************
# mpl.rcParams['figure.dpi'] = 120
# mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
# mpl.rcParams['savefig.dpi'] = 400
# mpl.rcParams['font.size'] = 10
# # mpl.rcParams['legend.fontsize'] = 'small'
# # mpl.rcParams['legend.fontsize'] = 'small'
# mpl.rcParams['lines.linewidth'] = 1.0
# # mpl.rcParams['errorbar.capsize'] = 3


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
    add_particlenode = pyqtSignal(object)


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

        # print("Hello from thread!!!!")
        # self.signals.progress.emit()
        try:
            self.openfile_func(self.fname, self.signals.start_progress,self.signals.auto_progress,
                               self.signals.progress, self.signals.status_message)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


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

        # print("Hello from thread!!!!")
        # self.signals.progress.emit()
        try:
            self.binall_func(self.dataset, self.bin_size, self.signals.start_progress, self.signals.progress, self.signals.status_message)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


class WorkerResolveLevels(QRunnable):
    """ A QRunnable class to create a worker thread for resolving levels. """

    def __init__(self, resolve_levels_func,
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

    @pyqtSlot()
    def run(self) -> None:
        """ The code that will be run when the thread is started. """

        try:
            self.resolve_levels_func(self.signals.start_progress, self.signals.progress,
                                     self.signals.status_message, self.resolve_all, self.resolve_selected)
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

        # Set defaults for figures depending on system
        if system() == "win32" or system() == "win64":
            dbg.p("System -> Windows", "Main")
            mpl.rcParams['figure.dpi'] = 120
            mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
            mpl.rcParams['savefig.dpi'] = 400
            mpl.rcParams['font.size'] = 10
            mpl.rcParams['lines.linewidth'] = 1.0
            self.fig_pos = [0.09, 0.12, 0.89, 0.85]  # [left, bottom, right, top]
            self.fig_life_int_pos = [0.12, 0.2, 0.85, 0.75]
            self.fig_lifetime_pos = [0.12, 0.22, 0.85, 0.75]
        elif system() == "Darwin":
            dbg.p("System -> Unix/Linus", "Main")
            mpl.rcParams['figure.dpi'] = 100
            mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
            mpl.rcParams['savefig.dpi'] = 400
            mpl.rcParams['font.size'] = 10
            mpl.rcParams['lines.linewidth'] = 1.0
            self.fig_pos = [0.1, 0.12, 0.8, 0.85]  # [left, bottom, right, top]
            self.fig_life_int_pos = [0.17, 0.2, 0.8, 0.75]
            self.fig_lifetime_pos = [0.15, 0.22, 0.8, 0.75]
        else:
            dbg.p("System -> Other", "Main")
            mpl.rcParams['figure.dpi'] = 120
            mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
            mpl.rcParams['savefig.dpi'] = 400
            mpl.rcParams['font.size'] = 10
            mpl.rcParams['lines.linewidth'] = 1.0
            self.fig_pos = [0.09, 0.12, 0.89, 0.85]  # [left, bottom, right, top]
            self.fig_life_int_pos = [0.12, 0.2, 0.85, 0.75]
            self.fig_lifetime_pos = [0.12, 0.22, 0.85, 0.75]

        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # print(self.MW_Intensity.figure.get_dpi())

        self.setWindowTitle("Full SMS")

        self.ui.MW_Intensity.axes.set_xlabel('Time (s)')
        self.ui.MW_Intensity.axes.set_ylabel('Bin Intensity (counts/bin)')
        self.ui.MW_Intensity.axes.patch.set_linewidth(0.1)
        self.ui.MW_Intensity.figure.tight_layout()
        self.ui.MW_Intensity.axes.set_position(self.fig_pos)
        # print(self.MW_Intensity.figure.get_dpi())

        self.ui.MW_LifetimeInt.axes.set_xlabel('Time (s)')
        self.ui.MW_LifetimeInt.axes.set_ylabel('Bin Intensity\n(counts/bin)')
        self.ui.MW_LifetimeInt.figure.tight_layout()
        self.ui.MW_LifetimeInt.axes.set_position(self.fig_life_int_pos)

        self.ui.MW_Lifetime.axes.set_xlabel('Time (ns)')
        self.ui.MW_Lifetime.axes.set_ylabel('Bin frequency\n(counts/bin)')
        self.ui.MW_Lifetime.figure.tight_layout()
        self.ui.MW_Lifetime.axes.set_position(self.fig_lifetime_pos)

        self.ui.MW_Spectra.axes.set_xlabel('Time (s)')
        self.ui.MW_Spectra.axes.set_ylabel('Wavelength (nm)')
        self.ui.MW_Spectra.figure.tight_layout()
        self.ui.MW_Spectra.axes.set_position(self.fig_pos)

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

        self.reset_gui()
        

    """#######################################
    ######## GUI Housekeeping Methods ########
    #######################################"""

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

        self.ui.spbBinSize.setValue(new_bin)

    def get_gui_confidence(self):
        """ Return current GUI value for confidence percentage. """

        return [self.ui.cmbConfIndex.currentIndex(), self.confidence_index[self.ui.cmbConfIndex.currentIndex()]]

    def gui_apply_bin(self):
        """ Changes the bin size of the data of the current particle and then displays the new trace. """

        try:
            self.currentparticle.binints(self.get_bin())
        except Exception as err:
            print('Error Occured:' + str(err))
        else:
            self.plot_trace()
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

        self.start_resolve_thread()

    def gui_resolve_selected(self):
        """ Resolves the levels of the selected particles and displays the levels of the current particle. """

        self.start_resolve_thread('selected')

    def gui_resolve_all(self):
        """ Resolves the levels of the all the particles and then displays the levels of the current particle. """

        self.start_resolve_thread('all')

    def gui_prev_lev(self):
        """ Moves to the previous resolves level and displays its decay curve. """

        print("gui_prev_lev")

    def gui_next_lev(self):
        """ Moves to the next resolves level and displays its decay curve. """

        print("gui_next_lev")

    def gui_load_irf(self):
        """ Allow the user to load a IRF instead of the IRF that has already been loaded. """

        print("gui_load_irf")

    def gui_fit_param(self):
        """ Opens a dialog to choose the setting with which the decay curve will be fitted. """

        print("gui_fit_param")

    def gui_fit_current(self):
        """ Fits the all the levels decay curves in the current particle using the provided settings. """

        print("gui_fit_current")

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
            of_worker = WorkerOpenFile(fname, self.open_h5)
            of_worker.signals.finished.connect(self.open_file_thread_complete)
            of_worker.signals.start_progress.connect(self.start_progress)
            of_worker.signals.progress.connect(self.update_progress)
            of_worker.signals.auto_progress.connect(self.update_progress)
            of_worker.signals.start_progress.connect(self.start_progress)
            of_worker.signals.status_message.connect(self.status_message)

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

    def display_data(self, current=None, prev=None) -> None:  # TODO: What is previous for?
        """ Displays the intensity trace and the histogram of the current particle.

        Parameters
        ----------
        current : QtCore.QModelIndex
            The index of the current selected particle as defined by QtCore.QModelIndex.
        prev : QtCore.QModelIndex
            The index of the previous selected particle as defined by QtCore.QModelIndex.
        """

        self.current_ind = current
        self.pre_ind = prev
        if current is not None:
            self.currentparticle = self.treemodel.data(current, Qt.UserRole)
        if type(self.currentparticle) is smsh5.Particle:
            self.set_bin(self.currentparticle.bin_size)
            self.plot_trace()
            self.currentparticle
            if self.currentparticle.has_levels:
                self.plot_levels()
            self.plot_decay()
            dbg.p('Current data displayed', 'Main')

    def plot_decay(self) -> None:
        """ Used to display the histogram of the decay data of the current particle. """

        try:
            decay = self.currentparticle.histogram.decay
            t = self.currentparticle.histogram.t

            trace = self.currentparticle.binnedtrace.intdata
        except AttributeError:
            print('No decay, or no trace!')
        else:
            self.ui.MW_Lifetime.axes.clear()
            self.ui.MW_Lifetime.axes.set_xlabel('Time (ns)')
            self.ui.MW_Lifetime.axes.set_ylabel('Bin frequency\n(counts/bin)')
            self.ui.MW_Lifetime.figure.tight_layout()
            self.ui.MW_Lifetime.axes.set_position(self.fig_lifetime_pos)
            self.ui.MW_Lifetime.axes.semilogy(t, decay)
            self.ui.MW_Lifetime.draw()

    def plot_trace(self) -> None:
        """ Used to display the trace from the absolute arrival time data of the current particle. """

        try:
            # self.currentparticle = self.treemodel.data(self.current_ind, Qt.UserRole)
            trace = self.currentparticle.binnedtrace.intdata
            times = self.currentparticle.binnedtrace.inttimes / 1E3
        except AttributeError:
            print('No trace!')
        else:
            self.ui.MW_Intensity.axes.clear()
            self.ui.MW_Intensity.axes.set_xlabel('Time (s)')
            self.ui.MW_Intensity.axes.set_ylabel('Bin Intensity (counts/{0}ms)'.format(self.get_bin()))
            self.ui.MW_Intensity.axes.patch.set_linewidth(0.1)
            self.ui.MW_Intensity.figure.tight_layout()
            self.ui.MW_Intensity.axes.set_position(self.fig_pos)
            self.ui.MW_Intensity.axes.plot(times, trace)
            self.ui.MW_Intensity.axes.set_xlim(0, times[-1])
            self.ui.MW_Intensity.draw()

            self.ui.MW_LifetimeInt.axes.clear()
            self.ui.MW_LifetimeInt.axes.set_xlabel('Time (s)')
            self.ui.MW_Intensity.axes.set_ylabel('Bin Intensity (counts/{0}ms)'.format(self.get_bin()))
            self.ui.MW_LifetimeInt.axes.patch.set_linewidth(0.1)
            self.ui.MW_LifetimeInt.figure.tight_layout()
            self.ui.MW_LifetimeInt.axes.set_position(self.fig_life_int_pos)
            self.ui.MW_LifetimeInt.axes.plot(times, trace)
            self.ui.MW_LifetimeInt.axes.set_xlim(0, times[-1])
            self.ui.MW_LifetimeInt.draw()

    def plot_levels(self):
        # self.currentparticle
        data, times = self.currentparticle.levels2data()
        data = data * self.get_bin()/1E3
        self.ui.MW_Intensity.axes.step(times, data, where='post')
        self.ui.MW_Intensity.draw()
        self.ui.MW_LifetimeInt.axes.step(times, data, where='post')
        self.ui.MW_LifetimeInt.draw()

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

        if type(identifier) is int:
            return self.part_nodes[identifier].dataobj
        if type(identifier) is DatasetTreeNode:
            return identifier.dataobj

    def tree2dataset(self):
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

    def start_resolve_thread(self, current_selected_all: str = 'current', thread_finished=None) -> None:
        """
        Creates a worker to resolve levels.

        Depending on the ``current_selected_all`` parameter the worker will be
        given the necessary parameter to fit the current, selected or all particles.

        Parameters
        ----------
        thread_finished
        current_selected_all : {'current', 'selected', 'all'}
            Possible values are 'current' (default), 'selected', and 'all'.
        """

        if thread_finished is None:
            if self.data_loaded:
                thread_finished = self.resolve_thread_complete
            else:
                thread_finished = self.open_file_thread_complete

        if current_selected_all == 'current':
            # sig = WorkerSignals()
            # self.resolve_levels(sig.start_progress, sig.progress, sig.status_message)
            resolve_thread = WorkerResolveLevels(self.resolve_levels)
        if current_selected_all == 'selected':
            resolve_thread = WorkerResolveLevels(self.resolve_levels, resolve_selected=self.get_checked_particles())
        elif current_selected_all == 'all':
            resolve_thread = WorkerResolveLevels(self.resolve_levels, resolve_all=True)

        # resolve_thread.signals.finished.connect(thread_finished)
        # resolve_thread.signals.start_progress.connect(self.start_progress)
        # resolve_thread.signals.progress.connect(self.update_progress)
        # resolve_thread.signals.status_message.connect(self.status_message)
        #
        # self.threadpool.start(resolve_thread)

    @dbg.profile
    def resolve_levels(self, start_progress_sig: pyqtSignal,
                       progress_sig: pyqtSignal, status_sig: pyqtSignal,
                       resolve_all: bool = None,
                       resolve_selected=None, parallel: bool = True) -> None:
        """
        Resolves the levels in particles by finding the change points in the
        abstimes data of a Particle instance.

        If no parameter are given the current particle will be resolved. If
        the ``resolve_all`` parameter is given **all** the loaded particles
        will be resolved. If the ``resolve_selected`` parameter is provided
        the selection of particles will be resolved.

        Parameters
        ----------
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

        assert not (resolve_all is not None and resolve_selected is not None), \
            "'resolve_all' and 'resolve_selected' can not both be given as parameters."

        if resolve_all is None and resolve_selected is None:  # Then resolve current
            data = self.currentparticle
            _, conf = self.get_gui_confidence()
            data.cpts.run_cpa(confidence=conf / 100, run_levels=True)
            self.display_data()
        elif resolve_all is not None and resolve_selected is None:  # Then resolve all
            data = self.tree2dataset()
            _, conf = self.get_gui_confidence()
            try:
                status_sig.emit('Resolving All Particle Levels...')
                start_progress_sig.emit(data.numpart)
                if parallel:
                    self.conf_parallel = conf
                    Parallel(n_jobs=-1, backend='threading')(
                        delayed(self.run_parallel_cpa)(self.tree2particle(num)) for num in range(data.numpart)
                    )
                    del self.conf_parallel
                else:
                    for num in range(data.numpart):
                        data.particles[num].cpts.run_cpa(confidence=conf, run_levels=True)
                        progress_sig.emit()
                status_sig.emit('Ready...')
                # test = self.ui.treeViewParticles.currentIndex()
                # index = self.ui.treeViewParticles.selectionModel().model().index(1,0)
                # self.ui.treeViewParticles.selectionModel().setCurrentIndex(index, QItemSelectionModel.NoUpdate)
                # self.ui.treeViewParticles.repaint()
                # self.repaint()
                if self.ui.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
                    self.display_data()
            except Exception as exc:
                raise RuntimeError("Couldn't resolve levels.") from exc
        else:
            pass

    def run_parallel_cpa(self, particle):
        particle.cpts.run_cpa(confidence=self.conf_parallel, run_levels=True)

    def resolve_thread_complete(self):
        dbg.p('Resolving levels complete', 'Resolve Thread')

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
                checked_particles.append(self.tree2particle(ind+1))
        return checked_particles

    def reset_gui(self):
        """ Sets the GUI elements to enabled if it should be accessible. """

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

        # Spectral
        if self.has_spectra:
            self.ui.tabSpectra.setEnabled(True)
            self.ui.btnSubBackground.setEnabled(enabled)
        else:
            self.ui.tabSpectra.setEnabled(False)


def main():
    app = QApplication([])
    dbg.p(debug_print='App created', debug_from='Main')
    main_window = MainWindow()
    dbg.p(debug_print='Main Window created', debug_from='Main')
    main_window.show()
    dbg.p(debug_print='Main Window shown', debug_from='Main')
    app.exec_()
    dbg.p(debug_print='App excuted', debug_from='Main')


if __name__ == '__main__':
    main()

""" Testing

class Test:
    def __init__(self):
        self.a = 1

    def new_a(self, value=int):
        self.a = value

test1 = Test()
test1_copy = test1
test1_copy.new_a(2)
print(test1.a)
"""
