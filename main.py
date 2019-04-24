# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------

from PyQt5.QtWidgets import*
from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt
from platform import system

from PyQt5.QtGui import *

import numpy as np
import scipy

import matplotlib as mpl

import dbg

import smsh5
import tcspcfit
from smsh5 import start_at_nonzero

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
        self.fitparam.irft = dataset.particles[0].histogram.t
        self.plot_irf()

    def gui_fit_param(self):
        if self.fitparamdialog.exec():
            self.fitparam.getfromdialog()

    def gui_fit_current(self):
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
        except AttributeError:
            dbg.p(debug_print='No Decay!', debug_from='Main')
        else:

            # Todo: Normalisation needs to happen in the fitting code
            decay = decay / decay.max()  # Normalise
            t = self.currentparticle.histogram.t

            decay, t = start_at_nonzero(decay, t)
            self.ui.MW_Lifetime.axes.clear()
            self.ui.MW_Lifetime.axes.semilogy(t, decay, color='xkcd:dull blue')
            self.ui.MW_Lifetime.axes.set_ylim(bottom=1e-3)
            self.ui.MW_Lifetime.draw()
            self.plot_irf()
            self.plot_convd()

    def plot_irf(self):
        if self.fitparam.irf is not None:
            irf = self.fitparam.irf / self.fitparam.irf.max()
            t = self.fitparam.irft

            irf, t = start_at_nonzero(irf, t)

            self.ui.MW_Lifetime.axes.semilogy(t, irf, color='xkcd:gray')
            self.ui.MW_Lifetime.draw()

    def plot_convd(self):
        try:
            hist = self.currentparticle.histogram
        except AttributeError:
            pass
        else:
            if hist.convd is not None:
                self.ui.MW_Lifetime.axes.semilogy(hist.convd_t, hist.convd, color='xkcd:marine blue', linewidth=2)
                self.ui.MW_Lifetime.draw()

    def plot_trace(self):
        try:
            trace = self.currentparticle.binnedtrace.intdata
        except AttributeError:
            dbg.p(debug_print='No Trace!', debug_from='Main')
        else:
            self.ui.MW_Intensity.axes.clear()
            self.ui.MW_Intensity.axes.plot(trace)
            self.ui.MW_Intensity.draw()


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
        if 0 <= in_column < len(self._data):
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
