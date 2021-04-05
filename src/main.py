"""Module for analysis of SMS data from HDF5 files

Bertus van Heerden and Joshua Botha
University of Pretoria
2020
"""

__docformat__ = 'NumPy'

import csv
import os
import sys
from platform import system
import ctypes

import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, Qt, QThreadPool, pyqtSlot, QTimer, QEvent
from PyQt5.QtGui import QIcon, QResizeEvent
from PyQt5.QtWidgets import QMainWindow, QProgressBar, QFileDialog, QMessageBox, QInputDialog, \
    QApplication, QStyleFactory
from PyQt5 import uic
from typing import Union
import time

from controllers import IntController, LifetimeController, GroupingController, SpectraController, \
    resolve_levels
from thread_tasks import bin_all, OpenFile
from threads import ProcessThread, WorkerResolveLevels, WorkerBinAll
from tree_model import DatasetTreeNode, DatasetTreeModel
from signals import worker_sig_pass

try:
    import pkg_resources.py2_warn
except ImportError:
    pass

import smsh5
from generate_sums import CPSums
from custom_dialogs import TimedMessageBox
import file_manager as fm
from my_logger import setup_logger
import processes as prcs

#  TODO: Needs to rather be reworked not to use recursion, but rather a loop of some sort

sys.setrecursionlimit(1000 * 10)

main_window_file = fm.path(name="mainwindow.ui", file_type=fm.Type.UI)
UI_Main_Window, _ = uic.loadUiType(main_window_file)

logger = setup_logger(__name__, is_main=True)


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
        logger.info(f"Multi-threading with maximum {self.threadpool.maxThreadCount()} threads")
        self.active_threads = []

        self.confidence_index = {
            0: 99,
            1: 95,
            2: 90,
            3: 69}

        if system() == "Windows":
            logger.info("System -> Windows")
        elif system() == "Darwin":
            logger.info("System -> Unix/Linus")
            os.environ["QT_MAC_WANTS_LAYER"] = '1'
        else:
            logger.info("System -> Other")

        QMainWindow.__init__(self)
        UI_Main_Window.__init__(self)
        self.setupUi(self)

        self.setWindowIcon(QIcon(fm.path('Full-SMS.ico', fm.Type.Icons)))

        self.tabWidget.setCurrentIndex(0)

        self.setWindowTitle("Full SMS")

        pg.setConfigOption('antialias', True)
        # pg.setConfigOption('leftButtonPan', False)

        self.chbInt_Disp_Resolved.hide()
        self.chbInt_Disp_Photon_Bursts.hide()
        self.chbInt_Disp_Grouped.hide()
        self.chbInt_Disp_Using_Groups.hide()
        self.chbInt_Show_Groups.setEnabled(False)

        self.int_controller = IntController(self, int_widget=self.pgIntensity_PlotWidget,
                                            int_hist_container=self.wdgInt_Hist_Container,
                                            int_hist_line=self.lineInt_Hist,
                                            int_hist_widget=self.pgInt_Hist_PlotWidget,
                                            lifetime_widget=self.pgLifetime_Int_PlotWidget,
                                            groups_int_widget=self.pgGroups_Int_PlotWidget,
                                            groups_hist_widget=self.pgGroups_Hist_PlotWidget)
        self.lifetime_controller = \
            LifetimeController(self, lifetime_hist_widget=self.pgLifetime_Hist_PlotWidget)
        self.spectra_controller = SpectraController(self, spectra_widget=self.pgSpectra_ImageView)
        self.grouping_controller = \
            GroupingController(self, bic_plot_widget=self.pgGroups_BIC_PlotWidget)

        # Connect all GUI buttons with outside class functions
        i_c = self.int_controller
        self.btnApplyBin.clicked.connect(i_c.gui_apply_bin)
        self.btnApplyBinAll.clicked.connect(i_c.gui_apply_bin_all)
        self.btnResolve.clicked.connect(i_c.gui_resolve)
        self.btnResolve_Selected.clicked.connect(i_c.gui_resolve_selected)
        self.btnResolveAll.clicked.connect(i_c.gui_resolve_all)
        self.chbInt_Show_Hist.stateChanged.connect(i_c.hide_unhide_hist)
        self.chbInt_Show_Groups.stateChanged.connect(i_c.plot_all)
        self.actionTime_Resolve_Current.triggered.connect(i_c.time_resolve_current)
        self.actionTime_Resolve_Selected.triggered.connect(i_c.time_resolve_selected)
        self.actionTime_Resolve_All.triggered.connect(i_c.time_resolve_all)

        l_c = self.lifetime_controller
        self.btnPrevLevel.clicked.connect(l_c.gui_prev_lev)
        self.btnNextLevel.clicked.connect(l_c.gui_next_lev)
        self.btnWholeTrace.clicked.connect(l_c.gui_whole_trace)
        self.btnLoadIRF.clicked.connect(l_c.gui_load_irf)
        self.btnFitParameters.clicked.connect(l_c.gui_fit_param)
        self.btnFitCurrent.clicked.connect(l_c.gui_fit_current)
        self.btnFit.clicked.connect(l_c.gui_fit_levels)
        self.btnFitSelected.clicked.connect(l_c.gui_fit_selected)
        self.btnFitAll.clicked.connect(l_c.gui_fit_all)

        g_c = self.grouping_controller
        self.btnGroupCurrent.clicked.connect(g_c.gui_group_current)
        self.btnGroupSelected.clicked.connect(g_c.gui_group_selected)
        self.btnGroupAll.clicked.connect(g_c.gui_group_all)
        self.btnApplyGroupsCurrent.clicked.connect(g_c.gui_apply_groups_current)
        self.btnApplyGroupsSelected.clicked.connect(g_c.gui_apply_groups_selected)
        self.btnApplyGroupsAll.clicked.connect(g_c.gui_apply_groups_all)

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
        self.current_progress = float()
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
        # self.pgSpectra.resize(self.tabSpectra.size().height(),
        #                       self.tabSpectra.size().height() - self.btnSubBackground.size().height() - 40)
        # QEvent.
        # QTimer.singleShot(1000)
        self.calc_store_sums()
        for i in range(100):
            time.sleep(1)
            print(i)
        pass

    def resizeEvent(self, a0: QResizeEvent):
        # if self.tabSpectra.size().height() <= self.tabSpectra.size().width():
        #     self.pgSpectra.resize(self.tabSpectra.size().height(),
        #                           self.tabSpectra.size().height() - self.btnSubBackground.size().height() - 40)
        # else:
        #     self.pgSpectra.resize(self.tabSpectra.size().width(),
        #                           self.tabSpectra.size().width() - 40)
        pass

    def sums_file_check(self) -> bool:

        should_calc = False
        sums_path = fm.path(name="all_sums.pickle", file_type=fm.Type.Data)
        if (not os.path.exists(sums_path)) and \
                (not os.path.isfile(sums_path)):
            self.status_message('Calculating change point sums, this may take several minutes.')
            should_calc = True

        return should_calc

    def calc_store_sums(self) -> None:
        """
        Check if the all_sums.pickle file exists, and if it doesn't creates it
        """

        create_all_sums = CPSums(only_pickle=True, n_min=10, n_max=1000)
        del create_all_sums
        self.status_message('Ready...')

    def gui_export_current(self):
        self.export(mode='current')

    def gui_export_selected(self):
        self.export(mode='selected')

    def gui_export_all(self):
        self.export(mode='all')

    def set_bin_size(self, bin_size: int):
        self.spbBinSize.setValue(bin_size)

    def act_open_h5(self):
        """ Allows the user to point to a h5 file and then starts a thread that reads and loads the file. """

        file_path = QFileDialog.getOpenFileName(self, 'Open HDF5 file', '', "HDF5 files (*.h5)")
        if file_path != ('', ''):  # fname will equal ('', '') if the user canceled.
            self.status_message(message="Opening file...")
            of_process_thread = ProcessThread(num_processes=1)
            of_process_thread.worker_signals.add_datasetindex.connect(self.add_dataset)
            of_process_thread.worker_signals.add_particlenode.connect(self.add_node)
            of_process_thread.worker_signals.add_all_particlenodes.connect(self.add_all_nodes)
            of_process_thread.worker_signals.bin_size.connect(self.set_bin_size)
            of_process_thread.worker_signals.data_loaded.connect(self.set_data_loaded)
            of_process_thread.signals.status_update.connect(self.status_message)
            of_process_thread.signals.start_progress.connect(self.start_progress)
            of_process_thread.signals.set_progress.connect(self.set_progress)
            of_process_thread.signals.step_progress.connect(self.update_progress)
            of_process_thread.signals.add_progress.connect(self.update_progress)
            of_process_thread.signals.end_progress.connect(self.end_progress)
            of_process_thread.signals.error.connect(self.error_handler)
            of_process_thread.signals.finished.connect(self.open_file_thread_complete)

            of_obj = OpenFile(file_path=file_path)  # , progress_tracker=of_progress_tracker)
            of_process_thread.add_tasks_from_methods(of_obj, 'open_h5')
            self.threadpool.start(of_process_thread)
            self.active_threads.append(of_process_thread)


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
                                               'HDF5 files (*.h5)',
                                               options=QFileDialog.DontConfirmOverwrite)
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
            # self.tree2dataset().makehistograms(remove_zeros=False, startpoint=start, channel=True)
            dataset = self.currentparticle.dataset
            dataset.makehistograms(remove_zeros=False, startpoint=start, channel=True)
        except Exception as exc:
            print(exc)
        if self.lifetime_controller.irf_loaded:
            self.lifetime_controller.change_irf_start(start)
        if self.lifetime_controller.startpoint is None:
            self.lifetime_controller.startpoint = start
        self.display_data()
        logger.info('Set startpoint')

    """#######################################
    ############ Internal Methods ############
    #######################################"""

    def add_dataset(self, datasetnode):
        self.datasetindex = self.treemodel.addChild(datasetnode)

    def add_node(self, particlenode, num):
        index = self.treemodel.addChild(particlenode, self.datasetindex)  #, progress_sig)
        if num == 0:
            self.treeViewParticles.expand(self.datasetindex)
            self.treeViewParticles.setCurrentIndex(index)

        self.part_nodes.append(particlenode)
        self.part_index.append(index)

    def add_all_nodes(self, all_particlenodes):
        for particlenode, num in all_particlenodes:
            index = self.treemodel.addChild(particlenode, self.datasetindex)  # , progress_sig)
            if num == 0:
                self.treeViewParticles.expand(self.datasetindex)
                self.treeViewParticles.setCurrentIndex(index)
            self.part_nodes.append(particlenode)
            self.part_index.append(index)

    def tab_change(self, active_tab_index: int):
        if self.data_loaded and hasattr(self, 'currentparticle'):
            if self.tabWidget.currentIndex() in [0, 1, 2, 3]:
                self.display_data()

    def update_int_gui(self):
        cur_part = self.currentparticle

        if cur_part.has_levels:
            self.chbInt_Disp_Resolved.show()
        else:
            self.chbInt_Disp_Resolved.hide()

        if cur_part.has_burst:
            self.chbInt_Disp_Photon_Bursts.show()
        else:
            self.chbInt_Disp_Photon_Bursts.hide()

        if cur_part.has_groups:
            self.chbInt_Disp_Grouped.show()
            self.chbInt_Show_Groups.setEnabled(True)
        else:
            self.chbInt_Disp_Grouped.hide()
            self.chbInt_Show_Groups.setEnabled(False)

        if cur_part.using_group_levels:
            self.chbInt_Disp_Using_Groups.show()
        else:
            self.chbInt_Disp_Using_Groups.hide()

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

        # self.current_ind = current
        # self.pre_ind = prev
        if current is not None:
            if hasattr(self, 'currentparticle'):
                self.currentparticle = self.treemodel.get_particle(current)
            self.current_level = None  # Reset current level when particle changes.
        if hasattr(self, 'currentparticle') and type(self.currentparticle) is smsh5.Particle:
            cur_tab_name = self.tabWidget.currentWidget().objectName()

            if cur_tab_name in ['tabIntensity', 'tabGrouping', 'tabLifetime']:
                if cur_tab_name == 'tabIntensity':
                    self.update_int_gui()
                self.int_controller.set_bin(self.currentparticle.bin_size)
                self.int_controller.plot_trace()
                if cur_tab_name != 'tabLifetime':
                    self.int_controller.plot_hist()
                else:
                    self.lifetime_controller.plot_decay(remove_empty=False)
                    self.lifetime_controller.plot_convd()
                    self.lifetime_controller.update_results()

                if self.currentparticle.has_groups:
                    self.int_controller.plot_group_bounds()
                    if cur_tab_name == 'tabGrouping':
                        self.grouping_controller.plot_group_bic()
                else:
                    self.grouping_controller.clear_bic()

            if cur_tab_name == 'tabSpectra':
                self.spectra_controller.plot_spectra()

            # Set Enables
            set_apply_groups = False
            if self.currentparticle.has_levels:
                self.int_controller.plot_levels()
                set_group = True
                if self.currentparticle.has_groups:
                    set_apply_groups = True
                else:
                    set_apply_groups = False
            else:
                set_group = False
            self.btnGroupCurrent.setEnabled(set_group)
            self.btnGroupSelected.setEnabled(set_group)
            self.btnGroupAll.setEnabled(set_group)
            self.btnApplyGroupsCurrent.setEnabled(set_apply_groups)
            self.btnApplyGroupsSelected.setEnabled(set_apply_groups)
            self.btnApplyGroupsAll.setEnabled(set_apply_groups)

            logger.info('Current data displayed')

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

    def start_progress(self, max_num: int = None) -> None:
        """
        Sets the maximum value of the progress bar before use.

        reset parameter can be optionally set to False to prevent the setting of the progress bar value to 0.

        Parameters
        ----------
        max_num : int
            The number of iterations or steps that the complete process is made up of.
        """

        if max_num:
            assert type(max_num) is int, "MainWindow:\tThe type of the 'max_num' parameter is not int."
            self.progress.setMaximum(max_num)
            # print(max_num)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.current_progress = 0

        self.progress.repaint()
        self.statusBar().repaint()
        self.repaint()

    def set_progress(self, progress_value: int) -> None:
        """
        Sets the maximum value of the progress bar before use.

        reset parameter can be optionally set to False to prevent the setting of the progress bar value to 0.

        Parameters
        ----------
        progress_value : int
            The number of iterations or steps that the complete process is made up of.
        """

        assert type(progress_value) is int,\
            "MainWindow:\tThe type of the 'max_num' parameter is not int."
        self.progress.setValue(progress_value)

        self.progress.repaint()
        self.statusBar().repaint()
        self.repaint()

    def update_progress(self, value: Union[int, float] = None) -> None:
        """ Used to update the progress bar by an increment of one. If at maximum sets progress bars visibility to False """

        if not value:
            value = 1.

        if self.progress.isVisible():
            self.current_progress += value
            new_show_value = int(self.current_progress//1)
            self.progress.setValue(new_show_value)
            # print(self.current_progress)
            if self.current_progress >= self.progress.maximum():
                self.end_progress()

        self.progress.repaint()
        self.statusBar().repaint()
        self.repaint()

    def end_progress(self):
        self.current_progress = 0
        self.progress.setValue(0)
        self.progress.setMaximum(0)
        self.progress.setVisible(False)
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
            return self.datasetindex.child(identifier,0).data(Qt.UserRole)
        if type(identifier) is DatasetTreeNode:
            return identifier.dataobj

    def tree2dataset(self) -> smsh5.H5dataset:
        """ Returns the H5dataset object of the file loaded.

        Returns
        -------
        smsh5.H5dataset
        """
        # return self.treemodel.data(self.treemodel.index(0, 0), Qt.UserRole)
        return self.datasetindex.data(Qt.UserRole)

    def set_data_loaded(self):
        self.data_loaded = True

    def open_file_thread_complete(self, thread: ProcessThread, irf=False) -> None:
        """ Is called as soon as all of the threads have finished. """

        if self.data_loaded and not irf:
            self.currentparticle = self.tree2particle(1)
            self.treeViewParticles.expandAll()
            self.treeViewParticles.setCurrentIndex(self.part_index[1])
            self.display_data()

            msgbx = TimedMessageBox(30, parent=self)
            msgbx.setIcon(QMessageBox.Question)
            msgbx.setText("Would you like to resolve levels now?")
            msgbx.set_timeout_text(message_pretime="(Resolving levels in ",
                                   message_posttime=" seconds)")
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
                                                    "Select confidence interval to use.",
                                                    confidences, 0, False)
                    if ok:
                        index = list(self.confidence_index.values()).index(int(float(item) * 100))
                self.cmbConfIndex.setCurrentIndex(index)
                self.int_controller.start_resolve_thread('all')
        self.reset_gui()
        self.gbxExport_Int.setEnabled(True)
        self.chbEx_Trace.setEnabled(True)
        logger.info('File opened')

    @pyqtSlot(Exception)
    def open_file_error(self, err: Exception):
        # logger.error(err)
        pass

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
        logger.info('Binnig all levels complete')

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
                                             conf=self.confidence_index[
                                                 self.cmbConfIndex.currentIndex()],
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
                start_progress_sig.emit(data.num_parts)
                # if parallel:
                #     self.conf_parallel = conf
                #     Parallel(n_jobs=-2, backend='threading')(
                #         delayed(self.run_parallel_cpa)
                #         (self.tree2particle(num)) for num in range(data.numpart)
                #     )
                #     del self.conf_parallel
                # else:
                for num in range(data.num_parts):
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
        logger.info('Resolving levels complete')
        self.check_remove_bursts(mode=mode)
        self.set_startpoint()
        self.chbEx_Levels.setEnabled(True)

    def check_remove_bursts(self, mode: str = None) -> None:
        if mode == 'current':
            particles = [self.currentparticle]
        elif mode == 'selected':
            particles = self.get_checked_particles()
        else:
            # particles = self.tree2dataset().particles
            particles = self.currentparticle.dataset.particles  #TODO: This needs to change.

        removed_bursts = False  # TODO: Remove
        has_burst = [particle.has_burst for particle in particles]
        if sum(has_burst):
            if not removed_bursts:
                removed_bursts = True
            msgbx = TimedMessageBox(30, parent=self)
            msgbx.setIcon(QMessageBox.Question)
            msgbx.setText("Would you like to remove the photon bursts?")
            msgbx.set_timeout_text(message_pretime="(Removing photon bursts in ",
                                   message_posttime=" seconds)")
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

            assert all_selected.lower() in ['all',
                                            'selected'], "mode parameter must be either 'all' or 'selected'."

            if all_selected == 'all':
                data = self.treemodel.data(self.treemodel.index(0, 0), Qt.UserRole)
                # assert data.
        except Exception as exc:
            logger.info('Switching frequency analysis failed: ')
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
                            rows.append(
                                [str(i), str(l.times_s[0]), str(l.times_s[1]), str(l.dwell_time_s),
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
                                other_exp = [str(l.histogram.avtau), str(l.histogram.shift),
                                             str(l.histogram.bg),
                                             str(l.histogram.irfbg)]

                            rows.append(
                                [str(i), str(l.times_s[0]), str(l.times_s[1]), str(l.dwell_time_s),
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

                    logger.info('Exporting Finished')

    def reset_gui(self):
        """ Sets the GUI elements to enabled if it should be accessible. """
        logger.info('Reset GUI')
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
            
    def error_handler(self, e: Exception):
        # logger(e)
        raise e


def display_on():
    print("Always On")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

def display_reset():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    sys.exit(0)


def main():
    """
    Creates QApplication and runs MainWindow().
    """
    # convert_convert_ui()
    app = QApplication([])
    print('Currently used style:', app.style().metaObject().className())
    print('Available styles:', QStyleFactory.keys())
    logger.info('App created')
    main_window = MainWindow()
    logger.info('Main Window created')
    main_window.show()
    should_calc = main_window.sums_file_check()
    if should_calc:
        app.processEvents()
        main_window.calc_store_sums()
        app.processEvents()
    # main_window.tabSpectra.repaint()
    logger.info('Main Window shown')
    if system() == "Windows":
        display_on()
    app.instance().exec_()
    if system() == "Windows":
        display_reset()
    logger.info('App excuted')


if __name__ == '__main__':
    main()
