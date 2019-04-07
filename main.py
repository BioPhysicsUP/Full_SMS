"""Module for analysis of SMS data from HDF5 files

Bertus van Heerden and Joshua Botha
University of Pretoria
2019
"""

__docformat__ = 'reStructuredText'

from PyQt5.QtWidgets import*
from PyQt5.QtCore import QObject, pyqtSignal, QAbstractItemModel, QModelIndex, Qt, QThreadPool, QRunnable, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from platform import system
import sys
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Axes
import numpy as np
import random
import matplotlib as mpl
import dbg
import traceback
import smsh5
from ui.mainwindow import Ui_MainWindow

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
    progress = pyqtSignal(int)
    set_progress = pyqtSlot(int, bool)
    add_datasetindex = pyqtSignal(object)
    add_particlenode = pyqtSignal(object)


class WorkerOpenFile(QRunnable):
    """ A QRunnable class to create a worker thread for opening h5 file. """
    # def __init__(self, fn, *args, **kwargs):
    def __init__(self, openfile_func):
        """
        Initiate Open File Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to open a h5 file
        and populate the tree in the mainwindow gui.

        :param openfile_func: Function to be called that will read the h5 file and populate the tree on the gui.
        :type openfile_func: function
        """

        super(WorkerOpenFile, self).__init__()
        self.openfile_func = openfile_func
        self.signals = WorkerSignals()
        # self.args = args
        # self.kwargs = kwargs
        # Add the callback to our kwargs

    @pyqtSlot()
    def run(self):
        """ The code that will be run when the thread is started. """

        # print("Hello from thread!!!!")
        # self.signals.progress.emit()
        try:
            self.openfile_func(self.signals.progress)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()
        # print(self.fname)
        # if fname != ('', ''):  # fname will equal ('', '') if the user canceled.
        # self.open_func(self.fname)

        # self.main_window.statusBar().showMessage("...binning traces, building histograms and preparing spectra...")
        # self.statusBar().show()

        # self.progress.setValue(0)
        # self.progress.setVisible(True)
        #
        # total = len(dataset.particles)
        # self.progress.setValue(100 * num / total)
        # self.progress.repaint()
        #
        # self.progress.setValue(0)
        # self.progress.setVisible(False)

        # dataset = smsh5.H5dataset(fname[0])
        # dataset.binints(100)
        # self.main_window.ui.spbBinSize.setValue(100)
        # dataset.makehistograms()
        #
        # datasetnode = DatasetTreeNode(fname[0][fname[0].rfind('/')+1:-3], dataset, 'dataset')
        # datasetindex = self.main_window.treemodel.addChild(datasetnode)
        # print(datasetindex)
        #
        # for particle in dataset.particles:
        #     particlenode = DatasetTreeNode(particle.name, particle, 'particle')
        #     self.main_window.treemodel.addChild(particlenode, datasetindex)


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
        print("Multi-threading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.confidence_index = {
            0: 99,
            1: 95,
            2: 90,
            3: 69
        }

        # Set defaults for figures depending on system
        if system() == "win32" or system() == "win64":
            dbg.p(debug_print="System: Windows", debug_from="Main")
            dpi = 120
            mpl.rcParams['figure.dpi'] = dpi
            mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
            mpl.rcParams['savefig.dpi'] = 400
            mpl.rcParams['font.size'] = 10
            mpl.rcParams['lines.linewidth'] = 1.0
            fig_pos = [0.09, 0.12, 0.89, 0.85]  # [left, bottom, right, top]
            fig_life_int_pos = [0.12, 0.2, 0.85, 0.75]
            fig_lifetime_pos = [0.12, 0.22, 0.85, 0.75]
        elif system() == "Darwin":
            dbg.p("System: Unix/Linus", "Main")
            dpi = 100
            mpl.rcParams['figure.dpi'] = dpi
            mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
            mpl.rcParams['savefig.dpi'] = 400
            mpl.rcParams['font.size'] = 10
            mpl.rcParams['lines.linewidth'] = 1.0
            fig_pos = [0.1, 0.12, 0.89, 0.85]  # [left, bottom, right, top]
            fig_life_int_pos = [0.17, 0.2, 0.8, 0.75]
            fig_lifetime_pos = [0.12, 0.22, 0.85, 0.75]
        else:
            dbg.p("System: Other", "Main")
            dpi = 120
            mpl.rcParams['figure.dpi'] = dpi
            mpl.rcParams['axes.linewidth'] = 1.0  # set  the value globally
            mpl.rcParams['savefig.dpi'] = 400
            mpl.rcParams['font.size'] = 10
            mpl.rcParams['lines.linewidth'] = 1.0
            fig_pos = [0.09, 0.12, 0.89, 0.85]  # [left, bottom, right, top]
            fig_life_int_pos = [0.12, 0.2, 0.85, 0.75]
            fig_lifetime_pos = [0.12, 0.22, 0.85, 0.75]

        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # print(self.MW_Intensity.figure.get_dpi())

        self.setWindowTitle("Full SMS")

        self.ui.MW_Intensity.axes.set_xlabel('Time (s)')
        self.ui.MW_Intensity.axes.set_ylabel('Bin Intensity (counts/bin)')
        self.ui.MW_Intensity.axes.patch.set_linewidth(0.1)
        self.ui.MW_Intensity.figure.tight_layout()
        self.ui.MW_Intensity.axes.set_position(fig_pos)
        # print(self.MW_Intensity.figure.get_dpi())

        self.ui.MW_LifetimeInt.axes.set_xlabel('Time (s)')
        self.ui.MW_LifetimeInt.axes.set_ylabel('Bin Intensity\n(counts/bin)')
        self.ui.MW_LifetimeInt.figure.tight_layout()
        self.ui.MW_LifetimeInt.axes.set_position(fig_life_int_pos)

        self.ui.MW_Lifetime.axes.set_xlabel('Time (ns)')
        self.ui.MW_Lifetime.axes.set_ylabel('Bin frequency\n(counts/bin)')
        self.ui.MW_Lifetime.figure.tight_layout()
        self.ui.MW_Lifetime.axes.set_position(fig_lifetime_pos)

        self.ui.MW_Spectra.axes.set_xlabel('Time (s)')
        self.ui.MW_Spectra.axes.set_ylabel('Wavelength (nm)')
        self.ui.MW_Spectra.figure.tight_layout()
        self.ui.MW_Spectra.axes.set_position(fig_pos)

        # Connect all GUI buttons with outside class functions
        self.ui.btnApplyBin.clicked.connect(self.gui_apply_bin)
        self.ui.btnApplyBinAll.clicked.connect(self.gui_apply_bin_all)
        self.ui.btnResolve.clicked.connect(self.gui_resolve)
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

        # Create and connect model for dataset tree
        self.treemodel = DatasetTreeModel()
        self.ui.treeViewParticles.setModel(self.treemodel)
        # Connect the tree selection to data display
        self.ui.treeViewParticles.selectionModel().currentChanged.connect(self.display_data)

        self.statusBar().showMessage('Load File')
        self.progress = QProgressBar(self)
        self.progress.setMinimumSize(170, 19)
        self.progress.setVisible(False)
        self.progress.setValue(0)  # Range of values is from 0 to 100
        self.statusBar().addPermanentWidget(self.progress)

    def get_bin(self):
        """ Returns current GUI value for bin size in ms.

        :return: Size of bin in ns.
        :rtype: int
        """

        return self.ui.spbBinSize.value()

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

    def gui_apply_bin_all(self):
        """ Changes the bin size of the data of all the particles and then displays the new trace of the current particle. """

        try:
            self.currentparticle.dataset.binints(self.get_bin())
        except Exception as err:
            print('Error Occured:' + str(err))
        else:
            self.plot_trace()
            self.repaint()

    def gui_resolve(self):
        """ Resolves the levels of the current particle and displays them. """

        print("gui_resolve")
        print(self.get_gui_confidence())

    def gui_resolve_all(self):
        """ Resolves the levels of the all the particles and then displays the levels of the current particle. """

        print("gui_resolve_all")

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

        open_file_worker = WorkerOpenFile(self.open_h5)
        open_file_worker.signals.finished.connect(self.thread_complete)
        open_file_worker.signals.set_progress.connect(self.set_progress)
        open_file_worker.signals.progress.connect(self.update_progress)

        self.threadpool.start(open_file_worker)

    def thread_complete(self):
        """ Is called as soon as one of the threads have finished. """

        print("THREAD COMPLETE!")

    def act_open_pt3(self):
        """ Allows a user to load a group of .pt3 files that are in a folder and loads them. NOT YET IMPLEMENTED. """

        print("act_open_pt3")

    def act_trim(self):
        """ Used to trim the 'dead' part of a trace as defined by two parameters. """

        print("act_trim")

    def display_data(self, current, prev):  # TODO: WHat is previous for?
        """ Displays the intensity trace and the histogram of the current particle. """

        self.currentparticle = self.treemodel.data(current, Qt.UserRole)
        self.plot_trace()
        self.plot_decay()

    def plot_decay(self):
        """ Used to display the histogram of the decay data of the current particle. """

        try:
            decay = self.currentparticle.histogram.decay
            t = self.currentparticle.histogram.t
        except AttributeError:
            print('No decay!')
        else:
            self.ui.MW_Lifetime.axes.clear()
            self.ui.MW_Lifetime.axes.semilogy(t, decay)
            self.ui.MW_Lifetime.draw()

    def plot_trace(self):
        """ Used to display the trace from the absolute arrival time data of the current particle. """

        try:
            trace = self.currentparticle.binnedtrace.intdata
        except AttributeError:
            print('No trace!')
        else:
            self.ui.MW_Intensity.axes.clear()
            self.ui.MW_Intensity.axes.plot(trace)
            self.ui.MW_Intensity.draw()

    def status_message(self, message):
        """
        Updates the status bar with the provided message argument.

        :param message: Message to be displayed in the status bar.
        :type message: str
        """

        pass


    def set_progress(self, max_num=None, reset=True):
        """
        Sets the maximum value of the progress bar before use.

        reset parameter can be optionally set to False.

        :param max_num: The maximum of steps to set the progress bar to.
        :type max_num: int
        :param reset: If true the current value of the progress bar will be set to 0.
        :type reset: bool
        """

    def update_progress(self):
        """ Used to update the progress bar by an increment of one. If at maximum sets progress bars visibility to False """

        print("update_progress called")

    def open_h5(self, progress_sig):
        """
        Read the selected h5 file and populates the tree on the gui with the file and the particles.

        Accepts a function that will be used to indicate the current progress.

        :param progress_sig: The function that will be called to update the gui with the progress.
        :type progress_sig: pyqtSignal
        """

        print("Open_h5 called from thread")
        progress_sig.emit()

        fname = QFileDialog.getOpenFileName(self, 'Open HDF5 file', '', "HDF5 files (*.h5)")
        if fname != ('', ''):  # fname will equal ('', '') if the user canceled.
            self.statusBar().showMessage("...binning traces, building histograms and preparing spectra...")
            # self.statusBar().show()

            # self.progress.setValue(0)
            # self.progress.setVisible(True)
            #
            # self.progress.setValue(100 * num / total)
            # self.progress.repaint()
            #
            # self.progress.setValue(0)
            # self.progress.setVisible(False)

            dataset = smsh5.H5dataset(fname[0])
            dataset.binints(100)
            self.ui.spbBinSize.setValue(100)
            dataset.makehistograms()

            datasetnode = DatasetTreeNode(fname[0][fname[0].rfind('/')+1:-3], dataset, 'dataset')
            datasetindex = self.treemodel.addChild(datasetnode)
            print(datasetindex)

            total = len(dataset.particles)
            for particle in dataset.particles:
                particlenode = DatasetTreeNode(particle.name, particle, 'particle')
                self.treemodel.addChild(particlenode, datasetindex)


class DatasetTreeNode:
    """ Contains the files with their respective particles. Also seems to house the actual data objects. """

    def __init__(self, name, dataobj, datatype):
        """
        TODO Docstring

        :param name:
        :type name:
        :param dataobj:
        :type dataobj:
        :param datatype:
        :type datatype:
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

        :param in_child:
        :type in_child:
        :return:
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

    def rowCount(self, in_index):
        """
        TODO: Docstring

        :param in_index:
        :type in_index:
        :return:
        """

        if in_index.isValid():
            return in_index.internalPointer().childCount()
        return self._root.childCount()

    def addChild(self, in_node, in_parent=None):
        """
        TODO: Docstring

        :param in_node:
        :type in_node:
        :param in_parent:
        :type in_parent:
        :return:
        """

        self.layoutAboutToBeChanged.emit()
        if not in_parent or not in_parent.isValid():
            parent = self._root
        else:
            parent = in_parent.internalPointer()
        row = parent.addChild(in_node)
        self.layoutChanged.emit()
        return self.index(row, 0)

    def index(self, in_row, in_column, in_parent=None):
        """
        TODO: Docstring

        :param in_row:
        :type in_row:
        :param in_column:
        :type in_column:
        :param in_parent:
        :type in_parent:
        :return:
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

        :param in_index:
        :type in_index:
        :return:
        """

        if in_index.isValid():
            p = in_index.internalPointer().parent()
            if p:
                return QAbstractItemModel.createIndex(self, p.row(), 0, p)
        return QModelIndex()

    def columnCount(self, in_index):
        """
        TODO: Docstring

        :param in_index:
        :type in_index:
        :return:
        """

        if in_index.isValid():
            return in_index.internalPointer().columnCount()
        return self._root.columnCount()

    def data(self, in_index, role):
        """
        TODO: Docstring

        :param in_index:
        :type in_index:
        :param role:
        :type role:
        :return:
        """

        if not in_index.isValid():
            return None
        node = in_index.internalPointer()
        if role == Qt.DisplayRole:
            return node.data(in_index.column())
        if role == Qt.UserRole:
            return node.dataobj
        return None


def main():
    app = QApplication([])
    dbg.p(debug_print='App created', debug_from='Main')
    main_window = MainWindow()
    dbg.p(debug_print='Main Window created', debug_from='Main')
    main_window.show()
    # print(main_window.f)
    dbg.p(debug_print='Main Window shown', debug_from='Main')
    # main_window.MW_Intensity.figure.set_dpi(100)
    # main_window.MW_Intensity.draw()
    # print(main_window.MW_Intensity.figure.get_dpi())
    app.exec_()
    dbg.p(debug_print='App excuted', debug_from='Main')


if __name__ == '__main__':
    main()
