# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------

from PyQt5.QtWidgets import*
from PyQt5.QtCore import QObject, pyqtSignal, QAbstractItemModel, QModelIndex, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from platform import system

from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Axes

import numpy as np
import random

import matplotlib as mpl

import dbg

import smsh5

from ui.mainwindow import Ui_MainWindow
from ui.fitting_dialog import Ui_Dialog

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
        self.fitparamdialog = FittingDialog(self)
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

        self.tauparam = None
        self.ampparam = None
        self.shift = None
        self.decaybg = None
        self.irfbg = None
        self.start = None
        self.end = None
        self.addopt = None
        self.fitparam = FittingParameters(self)

    def get_bin(self):
        """Returns current GUI value for bin size in ms."""
        return self.ui.spbBinSize.value()

    def get_gui_confidence(self):
        """Return current GUI value for confidence percentage."""
        return [self.ui.cmbConfIndex.currentIndex(), self.confidence_index[self.ui.cmbConfIndex.currentIndex()]]

    def gui_apply_bin(self):
        try:
            self.currentparticle.binints(self.get_bin())
        except Exception as err:
            print('Error Occured:' + str(err))
        else:
            self.plot_trace()

    def gui_apply_bin_all(self):
        try:
            self.currentparticle.dataset.binints(self.get_bin())
        except Exception as err:
            print('Error Occured:' + str(err))
        else:
            self.plot_trace()

    def gui_resolve(self):
        print("gui_resolve")
        print(main_window.get_gui_confidence())

    def gui_resolve_all(self):
        print("gui_resolve_all")

    def gui_prev_lev(self):
        print("gui_prev_lev")

    def gui_next_lev(self):
        print("gui_next_lev")

    def gui_load_irf(self):
        dataset, fname = self.open_h5_dataset()
        self.fitparam.irf = dataset.particles[0].histogram.decay

    def gui_fit_param(self):
        if self.fitparamdialog.exec():
            self.fitparam.getfromdialog()

    def gui_fit_current(self):
        try:
            self.currentparticle.histogram.fit(self.fitparam.tau, self.fitparam.amp, self.fitparam.shift,
                                               self.fitparam.decaybg, self.fitparam.irfbg, self.fitparam.start,
                                               self.fitparam.end, self.fitparam.addopt, self.fitparam.irf)
        except AttributeError:
            raise
            print("No decay")

    def gui_fit_selected(self):
        print("gui_fit_selected")

    def gui_fit_all(self):
        print("gui_fit_all")

    def gui_sub_bkg(self):
        print("gui_sub_bkg")

    def act_open_h5(self):
        dataset, fname = self.open_h5_dataset()

        datasetnode = DatasetTreeNode(fname[0], dataset, 'dataset')
        datasetindex = self.treemodel.addChild(datasetnode)
        print(datasetindex)

        for particle in dataset.particles:
            particlenode = DatasetTreeNode(particle.name, particle, 'particle')
            self.treemodel.addChild(particlenode, datasetindex)

    @staticmethod
    def open_h5_dataset():
        fname = QFileDialog.getOpenFileName(main_window, 'Open HDF5 file', '', "HDF5 files (*.h5)")
        try:
            dataset = smsh5.H5dataset(fname[0])
        except ValueError:
            dataset = None
        else:
            dataset.binints(100)
            dataset.makehistograms()
        return dataset, fname

    def act_open_pt3(self):
        print("act_open_pt3")

    def act_trim(self):
        print("act_trim")

    def display_data(self, current, prev):
        self.currentparticle = self.treemodel.data(current, Qt.UserRole)
        self.plot_trace()
        self.plot_decay()
        
    def plot_decay(self):
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
        try:
            trace = self.currentparticle.binnedtrace.intdata
        except AttributeError:
            print('No trace!')
        else:
            self.ui.MW_Intensity.axes.clear()
            self.ui.MW_Intensity.axes.plot(trace)
            self.ui.MW_Intensity.draw()


class FittingDialog(QDialog, Ui_Dialog):
    def __init__(self, parent):
        QDialog.__init__(self, parent)
        self.setupUi(self)


class FittingParameters:
    def __init__(self, parent):
        self.parent = parent
        self.fp = self.parent.fitparamdialog

        self.irf = None

    def getfromdialog(self):
        if int(self.fp.combNumExp.currentText()) == 1:
            self.tau = [[self.get_from_gui(i) for i in [self.fp.line1Init, self.fp.line1Min, self.fp.line1Max, self.fp.check1Fix]]]
            self.amp = [[self.get_from_gui(i) for i in [self.fp.line1AmpInit, self.fp.line1AmpMin, self.fp.line1AmpMax, self.fp.check1AmpFix]]]

        elif self.fp.combNumExp == 2:
            self.tau = [[self.get_from_gui(i) for i in [self.fp.line1Init, self.fp.line1Min, self.fp.line1Max, self.fp.check1Fix]],
                        [self.get_from_gui(i) for i in [self.fp.line2Init, self.fp.line2Min, self.fp.line2Max, self.fp.check2Fix]]]
            self.amp = [[self.get_from_gui(i) for i in [self.fp.line1AmpInit, self.fp.line1AmpMin, self.fp.line1AmpMax, self.fp.check1AmpFix]],
                        [self.get_from_gui(i) for i in [self.fp.line2AmpInit, self.fp.line2AmpMin, self.fp.line2AmpMax, self.fp.check2AmpFix]]]

        elif self.fp.combNumExp == 3:
            self.tau = [[self.get_from_gui(i) for i in [self.fp.line1Init, self.fp.line1Min, self.fp.line1Max, self.fp.check1Fix]],
                        [self.get_from_gui(i) for i in [self.fp.line2Init, self.fp.line2Min, self.fp.line2Max, self.fp.check2Fix]],
                        [self.get_from_gui(i) for i in [self.fp.line3Init, self.fp.line3Min, self.fp.line3Max, self.fp.check3Fix]]]
            self.amp = [[self.get_from_gui(i) for i in [self.fp.line1AmpInit, self.fp.line1AmpMin, self.fp.line1AmpMax, self.fp.check1AmpFix]],
                        [self.get_from_gui(i) for i in [self.fp.line2AmpInit, self.fp.line2AmpMin, self.fp.line2AmpMax, self.fp.check2AmpFix]],
                        [self.get_from_gui(i) for i in [self.fp.line3AmpInit, self.fp.line3AmpMin, self.fp.line3AmpMax, self.fp.check3AmpFix]]]

        self.shift = self.get_from_gui(self.fp.lineShift)
        self.decaybg = self.get_from_gui(self.fp.lineDecayBG)
        self.irfbg = self.get_from_gui(self.fp.lineIRFBG)
        self.start = self.get_from_gui(self.fp.lineStartTime)
        self.end = self.get_from_gui(self.fp.lineEndTime)

        self.addopt = self.get_from_gui(self.fp.lineAddOpt)
        
    def get_from_gui(self, guiobj):
        if type(guiobj) == QLineEdit:
            if guiobj.text() == '':
                return None
            else:
                return float(guiobj.text())
        elif type(guiobj) == QCheckBox:
            return float(guiobj.isChecked())


class DatasetTreeNode():
    def __init__(self, name, dataobj, datatype):
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
        if in_column >= 0 and in_column < len(self._data):
            return self._data[in_column]

    def columnCount(self):
        return self._columncount

    def childCount(self):
        return len(self._children)

    def child(self, in_row):
        if in_row >= 0 and in_row < self.childCount():
            return self._children[in_row]

    def parent(self):
        return self._parent

    def row(self):
        return self._row

    def addChild(self, in_child):
        in_child._parent = self
        in_child._row = len(self._children)
        self._children.append(in_child)
        self._columncount = max(in_child.columnCount(), self._columncount)

        return in_child._row


class DatasetTreeModel(QAbstractItemModel):
    def __init__(self):
        QAbstractItemModel.__init__(self)
        self._root = DatasetTreeNode(None, None, None)
        # for node in in_nodes:
        #     self._root.addChild(node)

    def rowCount(self, in_index):
        if in_index.isValid():
            return in_index.internalPointer().childCount()
        return self._root.childCount()

    def addChild(self, in_node, in_parent=None):
        self.layoutAboutToBeChanged.emit()
        if not in_parent or not in_parent.isValid():
            parent = self._root
        else:
            parent = in_parent.internalPointer()
        row = parent.addChild(in_node)
        self.layoutChanged.emit()
        return self.index(row, 0)

    def index(self, in_row, in_column, in_parent=None):
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
        if in_index.isValid():
            p = in_index.internalPointer().parent()
            if p:
                return QAbstractItemModel.createIndex(self, p.row(), 0, p)
        return QModelIndex()

    def columnCount(self, in_index):
        if in_index.isValid():
            return in_index.internalPointer().columnCount()
        return self._root.columnCount()

    def data(self, in_index, role):
        if not in_index.isValid():
            return None
        node = in_index.internalPointer()
        if role == Qt.DisplayRole:
            return node.data(in_index.column())
        if role == Qt.UserRole:
            return node.dataobj
        return None


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
