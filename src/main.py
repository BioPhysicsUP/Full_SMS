"""Module for analysis of SMS data from HDF5 files

Bertus van Heerden and Joshua Botha
University of Pretoria
2020
"""

from __future__ import annotations

__docformat__ = 'NumPy'

# import csv
import os
import sys
from platform import system
import ctypes

from PyQt5.QtCore import Qt, QThreadPool, pyqtSlot
from PyQt5.QtGui import QIcon  # , QResizeEvent
from PyQt5.QtWidgets import QMainWindow, QProgressBar, QFileDialog, QMessageBox, QInputDialog, \
    QApplication, QStyleFactory  # , QTreeWidget
from PyQt5 import uic
import pyqtgraph as pg
from typing import Union
import time
from multiprocessing import Process, freeze_support
from threading import Lock

from controllers import IntController, LifetimeController, GroupingController, SpectraController, \
    RasterScanController, AntibunchingController
from thread_tasks import OpenFile
from threads import ProcessThread
from tree_model import DatasetTreeNode, DatasetTreeModel
# import save_analysis
from settings_dialog import SettingsDialog, Settings

try:
    import pkg_resources.py2_warn
except ImportError:
    pass

import smsh5
from generate_sums import CPSums
from custom_dialogs import TimedMessageBox
import file_manager as fm
from my_logger import setup_logger
from convert_pt3 import ConvertPt3Dialog
from exporting import export_data, ExportWorker, DATAFRAME_FORMATS
from save_analysis import SaveAnalysisWorker, LoadAnalysisWorker
from selection import RangeSelectionDialog
import smsh5_file_reader

SMS_VERSION = "0.4.0"

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

        self.settings_dialog = SettingsDialog(self, get_saved_settings=True)
        self.settings = self.settings_dialog.settings

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
                                            groups_hist_widget=self.pgGroups_Hist_PlotWidget,
                                            level_info_container=self.wdgInt_Level_Info_Container,
                                            level_info_text=self.txtLevelInfoInt,
                                            int_level_line=self.lineInt_Level)
        # Connect all GUI buttons with outside class functions
        i_c = self.int_controller
        self.btnApplyBin.clicked.connect(i_c.gui_apply_bin)
        self.btnApplyBinAll.clicked.connect(i_c.gui_apply_bin_all)
        self.btnResolve.clicked.connect(i_c.gui_resolve)
        self.btnResolve_Selected.clicked.connect(i_c.gui_resolve_selected)
        self.btnResolveAll.clicked.connect(i_c.gui_resolve_all)
        self.chbInt_Show_ROI.stateChanged.connect(i_c.roi_chb_changed)
        self.chbInt_Show_Hist.stateChanged.connect(i_c.hist_chb_changed)
        self.chbInt_Show_Level_Info.stateChanged.connect(i_c.level_info_chb_changed)
        self.chbInt_Show_Groups.stateChanged.connect(i_c.plot_all)
        self.actionInt_Trim_Traces.triggered.connect(i_c.gui_trim_traces)
        self.actionInt_Reset_ROI_Current.triggered.connect(i_c.gui_reset_roi_current)
        self.actionInt_Reset_ROI_Selected.triggered.connect(i_c.gui_reset_roi_selected)
        self.actionInt_Reset_ROI_All.triggered.connect(i_c.gui_reset_roi_all)
        # self.actionTime_Resolve_Current.triggered.connect(i_c.time_resolve_current)
        # self.actionTime_Resolve_Selected.triggered.connect(i_c.time_resolve_selected)
        # self.actionTime_Resolve_All.triggered.connect(i_c.time_resolve_all)
        self.chbInt_Exp_Trace.stateChanged.connect(i_c.exp_trace_chb_changed)

        self.lifetime_controller = \
            LifetimeController(self, lifetime_hist_widget=self.pgLifetime_Hist_PlotWidget,
                               residual_widget=self.pgLieftime_Residuals_PlotWidget)
        l_c = self.lifetime_controller
        self.btnPrevLevel.clicked.connect(l_c.gui_prev_lev)
        self.btnNextLevel.clicked.connect(l_c.gui_next_lev)
        self.btnWholeTrace.clicked.connect(l_c.gui_whole_trace)
        self.chbLifetime_Show_Groups.stateChanged.connect(l_c.plot_all)
        self.chbShow_Residuals.stateChanged.connect(l_c.gui_show_hide_residuals)
        self.chbLifetime_Use_ROI.stateChanged.connect(l_c.gui_use_roi_changed)
        self.btnLifetime_Apply_ROI.clicked.connect(l_c.gui_apply_roi_current)
        self.btnLifetime_Apply_ROI_Selected.clicked.connect(l_c.gui_apply_roi_selected)
        self.btnLifetime_Apply_ROI_All.clicked.connect(l_c.gui_apply_roi_all)
        self.btnJumpToGroups.clicked.connect(l_c.gui_jump_to_groups)
        self.btnLoadIRF.clicked.connect(l_c.gui_load_irf)
        self.btnFitParameters.clicked.connect(l_c.gui_fit_param)
        self.btnFitCurrent.clicked.connect(l_c.gui_fit_current)
        self.btnFit.clicked.connect(l_c.gui_fit_levels)
        self.btnFitSelected.clicked.connect(l_c.gui_fit_selected)
        self.btnFitAll.clicked.connect(l_c.gui_fit_all)

        self.grouping_controller = \
            GroupingController(self, bic_plot_widget=self.pgGroups_BIC_PlotWidget)
        g_c = self.grouping_controller
        self.btnGroupCurrent.clicked.connect(g_c.gui_group_current)
        self.btnGroupSelected.clicked.connect(g_c.gui_group_selected)
        self.btnGroupAll.clicked.connect(g_c.gui_group_all)
        self.btnApplyGroupsCurrent.clicked.connect(g_c.gui_apply_groups_current)
        self.btnApplyGroupsSelected.clicked.connect(g_c.gui_apply_groups_selected)
        self.btnApplyGroupsAll.clicked.connect(g_c.gui_apply_groups_all)

        self.pgSpectra_Image_View = pg.ImageView(view=pg.PlotItem())
        self.laySpectra.addWidget(self.pgSpectra_Image_View)
        self.pgSpectra_Image_View.show()
        self.spectra_controller = \
            SpectraController(self, spectra_image_view=self.pgSpectra_Image_View)

        self.raster_scan_controller = \
            RasterScanController(self, raster_scan_image_view=self.pgRaster_Scan_Image_View,
                                 list_text=self.txtRaster_Scan_List)

        self.antibunch_controller = AntibunchingController(self, corr_widget=self.pgAntibunching_PlotWidget)
        a_c = self.antibunch_controller
        self.btnLoadIRFCorr.clicked.connect(a_c.gui_load_irf)
        self.btnCorrCurrent.clicked.connect(a_c.gui_correlate_current)

        self.btnSubBackground.clicked.connect(self.spectra_controller.gui_sub_bkg)

        self.actionOpen_h5.triggered.connect(self.act_open_h5)
        self.actionSave_Selected.triggered.connect(self.act_save_selected)
        self.actionSave_Analysis.triggered.connect(self.act_save_analysis)
        self.actionSelect_All.triggered.connect(self.act_select_all)
        self.actionInvert_Selection.triggered.connect(self.act_invert_selection)
        self.actionDeselect_All.triggered.connect(self.act_deselect_all)
        self.actionTrim_Dead_Traces.triggered.connect(self.act_trim)
        self.actionSwitch_All.triggered.connect(self.act_switch_all)
        self.actionSwitch_Selected.triggered.connect(self.act_switch_selected)
        self.actionSet_Startpoint.triggered.connect(self.act_set_startpoint)
        self.actionConvert_pt3.triggered.connect(self.convert_pt3_dialog)
        self.actionRange_Selection.triggered.connect(self.range_selection)
        self.actionSettings.triggered.connect(self.act_open_settings_dialog)
        self.actionDetect_Remove_Bursts_Current.triggered.connect(self.act_detect_remove_bursts_current)
        self.actionDetect_Remove_Bursts_Selected.triggered.connect(self.act_detect_remove_bursts_selected)
        self.actionDetect_Remove_Bursts_All.triggered.connect(self.act_detect_remove_bursts_all)
        self.actionRemove_Bursts_Current.triggered.connect(self.act_remove_bursts_current)
        self.actionRemove_Bursts_Selected.triggered.connect(self.act_remove_bursts_selected)
        self.actionRemove_Bursts_All.triggered.connect(self.act_remove_bursts_all)
        self.actionRestore_Bursts_Current.triggered.connect(self.act_restore_bursts_current)
        self.actionRestore_Bursts_Selected.triggered.connect(self.act_restore_bursts_selected)
        self.actionRestore_Bursts_All.triggered.connect(self.act_restore_bursts_all)

        self.chbGroup_Use_ROI.stateChanged.connect(self.gui_group_use_roi)
        self.btnEx_Current.clicked.connect(self.gui_export_current)
        self.btnEx_Selected.clicked.connect(self.gui_export_selected)
        self.btnEx_All.clicked.connect(self.gui_export_all)
        self.chbEx_Plot_Intensity.clicked.connect(self.gui_plot_intensity_clicked)
        self.chbEx_Plot_Lifetimes.clicked.connect(self.gui_plot_lifetime_clicked)
        self.chbEx_DF_Traces.stateChanged.connect(self.set_export_options)
        self.chbEx_DF_Levels.stateChanged.connect(self.set_export_options)
        self.chbEx_DF_Grouped_Levels.stateChanged.connect(self.set_export_options)
        self.btnSelectAllExport.clicked.connect(self.select_all_export_options)
        self.btnSelectAllExport_Plots.clicked.connect(self.select_all_plots_export_options)
        self.btnSelectAllExport_DataFrames.clicked.connect(
            self.select_all_dataframes_export_options)

        self.lblGrouping_ROI.setVisible(False)

        self.cmbEx_DataFrame_Format.addItems(DATAFRAME_FORMATS)

        # Create and connect model for dataset tree
        self.treemodel = DatasetTreeModel()
        self.treeViewParticles.setModel(self.treemodel)
        # Connect the tree selection to data display
        self.treeViewParticles.selectionModel().currentChanged.connect(self.display_data)
        self.treeViewParticles.clicked.connect(self.tree_view_clicked)
        # self.treeViewParticles.keyPressEvent().connect(self.tree_view_key_press)
        self._root_was_checked = False

        self.comboSelectCard.currentIndexChanged.connect(self.card_selected)

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
        self.irf_loaded = False

        # self._current_level = None

        self.tabWidget.currentChanged.connect(self.tab_change)

        self.current_dataset = None
        self.current_particle = None

        self.reset_gui()
        self.repaint()

        self.lock = None

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

    # def resizeEvent(self, a0: QResizeEvent):
    # if self.tabSpectra.size().height() <= self.tabSpectra.size().width():
    #     self.pgSpectra.resize(self.tabSpectra.size().height(),
    #                           self.tabSpectra.size().height() - self.btnSubBackground.size().height() - 40)
    # else:
    #     self.pgSpectra.resize(self.tabSpectra.size().width(),
    #                           self.tabSpectra.size().width() - 40)
    # pass

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
        self.gui_export(mode='current')

    def gui_export_selected(self):
        self.gui_export(mode='selected')

    def gui_export_all(self):
        self.gui_export(mode='all')

    def act_detect_remove_bursts_current(self):
        self.detect_remove_bursts(mode='current')

    def act_detect_remove_bursts_selected(self):
        self.detect_remove_bursts(mode='selected')

    def act_detect_remove_bursts_all(self):
        self.detect_remove_bursts(mode='all')

    def act_remove_bursts_current(self):
        self.remove_bursts(mode='current', confirm=False)

    def act_remove_bursts_selected(self):
        self.remove_bursts(mode='selected', confirm=False)

    def act_remove_bursts_all(self):
        self.remove_bursts(mode='all', confirm=False)

    def act_restore_bursts_current(self):
        self.restore_bursts(mode='current')

    def act_restore_bursts_selected(self):
        self.restore_bursts(mode='selected')

    def act_restore_bursts_all(self):
        self.restore_bursts(mode='all')

    def set_bin_size(self, bin_size: int):
        self.spbBinSize.setValue(bin_size)

    def act_open_settings_dialog(self):
        self.settings_dialog.exec()

    def gui_group_use_roi(self):
        if self.data_loaded:
            use_roi = self.chbGroup_Use_ROI.isChecked()
            for particle in self.current_dataset.particles:
                particle.ahca.use_roi_for_grouping = use_roi

    def act_open_h5(self):
        """ Allows the user to point to a h5 file and then starts a thread that reads and loads the file. """

        logger.info("Performing Open H5 Action")
        last_opened_file = fm.path(name='last_opened.txt', file_type=fm.Type.ResourcesRoot)
        if os.path.exists(last_opened_file) and os.path.isfile(last_opened_file):
            with open(last_opened_file, 'r') as file:
                last_opened_path = file.read()
            if not os.path.isdir(last_opened_path):
                last_opened_path = ''
        file_path = QFileDialog.getOpenFileName(self, 'Open HDF5 file', last_opened_path,
                                                "HDF5 files (*.h5)")
        did_open = False
        loading_analysis = False
        if os.path.exists(file_path[0][:-2] + 'smsa') and \
                os.path.isfile(file_path[0][:-2] + 'smsa'):
            msg_box = QMessageBox(parent=self)
            msg_box.setWindowTitle("Load analysis?")
            msg_box.setText("Analysis file found. Would you like to load it?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.exec()
            if msg_box.result() == QMessageBox.Yes:
                load_analysis_worker = LoadAnalysisWorker(main_window=self,
                                                          file_path=file_path[0][:-2] + 'smsa')
                load_analysis_worker.signals.status_message.connect(self.status_message)
                load_analysis_worker.signals.start_progress.connect(self.start_progress)
                load_analysis_worker.signals.end_progress.connect(self.end_progress)
                load_analysis_worker.signals.error.connect(self.error_handler)
                load_analysis_worker.signals. \
                    openfile_finished.connect(self.open_file_thread_complete)
                load_analysis_worker.signals.save_file_version_outdated. \
                    connect(self.open_save_file_version_outdated)
                load_analysis_worker.signals.show_residual_widget. \
                    connect(self.lifetime_controller.show_residuals_widget)
                self.threadpool.start(load_analysis_worker)
                loading_analysis = True
                did_open = True
        if file_path != ('', '') and not loading_analysis:
            self.status_message(message="Opening file...")
            # logger.info("About to create ProcessThread object")
            of_process_thread = ProcessThread(num_processes=1)
            # logger.info("About to connect signals")
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

            # logger.info("About to create OpenFile object")
            of_obj = OpenFile(file_path=file_path)  # , progress_tracker=of_progress_tracker)
            of_process_thread.add_tasks_from_methods(of_obj, 'open_h5')
            # logger.info("About to start Process Thread")
            self.threadpool.start(of_process_thread)
            # logger.info("Started Process Thread")
            self.active_threads.append(of_process_thread)
            did_open = True
        if did_open:
            with open(last_opened_file, 'w') as file:
                file.write(os.path.split(file_path[0])[0])

    def act_save_selected(self):
        """" Saves selected particles into a new HDF5 file."""

        msg = QMessageBox(self)
        msg.setWindowTitle("Still in development")
        msg.setText("This functionality is still in development")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        return

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

        if self.current_dataset.name == fname:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle('Save Error')
            msg.setText('Can''t add particles to currently opened file.')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            return

        self.current_dataset.save_particles(fname, selected_nums)

    def act_save_analysis(self):
        if self.current_dataset is not None:
            save_analysis_worker = SaveAnalysisWorker(main_window=self,
                                                      dataset=self.current_dataset)
            save_analysis_worker.signals.status_message.connect(self.status_message)
            save_analysis_worker.signals.start_progress.connect(self.start_progress)
            save_analysis_worker.signals.end_progress.connect(self.end_progress)
            save_analysis_worker.signals.error.connect(self.error_handler)
            self.threadpool.start(save_analysis_worker)
            # save_analysis.save_analysis(self, self.current_dataset)

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

    def set_startpoint(self, irf_data=None, start=None):
        if start is None:
            start = self.lifetime_controller.startpoint
        try:
            # self.tree2dataset().makehistograms(remove_zeros=False, startpoint=start, channel=True)
            dataset = self.current_dataset
            dataset.makehistograms(remove_zeros=False, startpoint=start, channel=True)
        except Exception as exc:
            print(exc)
        if self.lifetime_controller.irf_loaded and irf_data:
            self.lifetime_controller.change_irf_start(start, irf_data)
        if self.lifetime_controller.startpoint is None:
            self.lifetime_controller.startpoint = start
        self.display_data()
        logger.info('Set startpoint')

    """#######################################
    ############ Internal Methods ############
    #######################################"""

    def add_dataset(self, dataset_node):
        self.dataset_node = dataset_node
        self.dataset_index = self.treemodel.addChild(dataset_node)
        self.current_dataset = dataset_node.dataobj

    def add_node(self, particle_node, num):
        index = self.treemodel.addChild(particle_node, self.dataset_index)  # , progress_sig)
        if num == 0:
            self.treeViewParticles.expand(self.dataset_index)
            self.treeViewParticles.setCurrentIndex(index)
            self.current_particle = particle_node.dataobj

        self.part_nodes.append(particle_node)
        self.part_index.append(index)

    def add_all_nodes(self, all_nodes):
        for node, num in all_nodes:
            if num == -1:
                assert type(node.dataobj) is smsh5.H5dataset, "First node must be for H5Dataset"
                self.add_dataset(node)
                self.treeViewParticles.expand(self.dataset_index)
            # index = self.treemodel.addChild(node, self.datasetindex)  # , progress_sig)
            else:
                assert type(node.dataobj) is smsh5.Particle, "Node must be for Particle"
                self.add_node(node, num)

    def tree_view_clicked(self, model_index):
        if type(self.treemodel.data(model_index, Qt.UserRole)) is smsh5.Particle:
            self.set_export_options()
            self.grouping_controller.check_rois_and_set_label()
            self.lifetime_controller.update_apply_roi_button_colors()
        if self.treemodel.data(model_index, Qt.UserRole) is self.dataset_node.dataobj:
            root_node_checked = self.dataset_node.checked()
            if all([node.checked() for node in self.part_nodes]) != root_node_checked:
                for part_node in self.part_nodes:
                    part_node.setChecked(root_node_checked)
                self._root_was_checked = root_node_checked
            if root_node_checked:
                self.lblNum_Selected.setText(str(len(self.part_nodes)))
            else:
                self.lblNum_Selected.setText('0')
            self.treeViewParticles.viewport().repaint()
        else:
            checked_list = [node.checked() for node in self.part_nodes]
            all_checked = all(checked_list)
            self.dataset_node.setChecked(all_checked)
            num_checked = sum(checked_list)
            self.lblNum_Selected.setText(str(num_checked))
            self.treeViewParticles.viewport().repaint()

    def tree_view_key_press(self, event):
        pass
        # print('here')

    def act_select_all(self, *args, **kwargs):
        if self.data_loaded:
            for node in self.part_nodes:
                node.setChecked(True)
            self.lblNum_Selected.setText(str(len(self.part_nodes)))

    def act_invert_selection(self, *args, **kwargs):
        if self.data_loaded:
            for node in self.part_nodes:
                node.setChecked(not node.checked())
            num_checked = sum([node.checked() for node in self.part_nodes])
            self.lblNum_Selected.setText(str(num_checked))
            self.treeViewParticles.viewport().repaint()

    def act_deselect_all(self, *args, **kwargs):
        if self.data_loaded:
            for node in self.part_nodes:
                node.setChecked(False)
            self.lblNum_Selected.setText('0')

    def tab_change(self, active_tab_index: int):
        if self.data_loaded and hasattr(self, 'current_particle'):
            if self.tabWidget.currentIndex() in [0, 1, 2, 3, 4, 5]:
                self.display_data()

    def update_int_gui(self):
        cur_part = self.current_particle

        if cur_part.has_levels:
            self.chbInt_Disp_Resolved.show()
        else:
            self.chbInt_Disp_Resolved.hide()

        if cur_part.has_burst:
            self.chbInt_Disp_Photon_Bursts.show()
            if cur_part.cpts.bursts_deleted is not None:
                self.chbInt_Disp_Photon_Bursts.setChecked(True)
            else:
                self.chbInt_Disp_Photon_Bursts.setChecked(False)
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

    def card_selected(self) -> None:
        self.display_data(combocard=True)

    def display_data(self, current=None, prev=None, combocard=False) -> None:
        """ Displays the intensity trace and the histogram of the current particle.

            Directly called by the tree signal currentChanged, thus the two arguments.

        Parameters
        ----------
        current : QtCore.QModelIndex
            The index of the current selected particle as defined by QtCore.QModelIndex.
        prev : QtCore.QModelIndex
            The index of the previous selected particle as defined by QtCore.QModelIndex.
        combocard : bool
            True if called due to selecting other TCSPC card.
        """

        # self.current_level = None

        # self.current_ind = current
        # self.pre_ind = prev
        self.treeViewParticles.viewport().repaint()
        if current is not None:
            if hasattr(self, 'current_particle'):
                self.current_particle = self.treemodel.get_particle(current)
            # self.current_level = None  # Reset current level when particle changes.
        if hasattr(self, 'current_particle') and type(self.current_particle) is smsh5.Particle:
            # Select primary or secondary particle based on selected tcspc card
            if self.comboSelectCard.currentIndex() == 1 and self.current_particle.sec_part is not None:
                assert not self.current_particle.is_secondary_part
                self.current_particle = self.current_particle.sec_part
            elif self.comboSelectCard.currentIndex() == 0 and self.current_particle.is_secondary_part:
                self.current_particle = self.current_particle.prim_part

            cur_tab_name = self.tabWidget.currentWidget().objectName()

            self.txtDescription.setText(self.current_particle.description)

            # If not called due to a change in selected card, update the card selector with available choices
            if not combocard:
                if not self.current_particle.is_secondary_part:
                    card1 = self.current_particle.tcspc_card
                    if self.current_particle.sec_part is not None:
                        card2 = self.current_particle.sec_part.tcspc_card
                    else:
                        card2 = None
                else:
                    card1 = self.current_particle.prim_part.tcspc_card
                    card2 = self.current_particle.tcspc_card
                if self.comboSelectCard.count() == 0:
                    self.comboSelectCard.insertItem(0, card1)
                    self.comboSelectCard.insertItem(1, card2)
                else:
                    self.comboSelectCard.setItemText(0, card1)
                    if self.comboSelectCard.count() == 1:
                        self.comboSelectCard.insertItem(1, card2)
                    else:
                        self.comboSelectCard.setItemText(1, card2)
            assert self.comboSelectCard.count() <= 2

            if cur_tab_name in ['tabIntensity', 'tabGrouping', 'tabLifetime']:
                if cur_tab_name == 'tabIntensity':
                    self.update_int_gui()
                self.int_controller.set_bin(self.current_particle.bin_size)
                self.int_controller.plot_trace()
                self.int_controller.update_level_info()
                if cur_tab_name != 'tabLifetime':
                    self.int_controller.plot_hist()
                else:
                    self.lifetime_controller.plot_decay(remove_empty=False)
                    self.lifetime_controller.plot_convd()
                    self.lifetime_controller.plot_residuals()
                    self.lifetime_controller.update_results()
                    self.lifetime_controller.update_apply_roi_button_colors()

                if self.current_particle.has_groups:
                    self.int_controller.plot_group_bounds()
                    if cur_tab_name == 'tabGrouping':
                        self.grouping_controller.plot_group_bic()
                else:
                    self.grouping_controller.clear_bic()

            elif cur_tab_name == 'tabSpectra' and self.current_particle.has_spectra:
                self.spectra_controller.plot_spectra()

            elif cur_tab_name == 'tabRaster_Scan' and self.current_particle.has_raster_scan:
                self.raster_scan_controller.plot_raster_scan()

            elif cur_tab_name == 'tabExport':
                self.set_export_options()

            # Set Enables
            set_apply_groups = False
            if self.current_particle.has_levels:
                self.int_controller.plot_levels()
                set_group = True
                if self.current_particle.has_groups:
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

        assert type(progress_value) is int, \
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
            new_show_value = int(self.current_progress // 1)
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
        The identifier could be the number of the particle of the datasetnode value.

        Parameters
        ----------
        identifier
            The integer number or a datasetnode object of the particle in question.
        Returns
        -------

        """
        if type(identifier) is int:
            return self.dataset_index.child(identifier, 0).data(Qt.UserRole)
        if type(identifier) is DatasetTreeNode:
            return identifier.dataobj

    def tree2dataset(self) -> smsh5.H5dataset:
        """ Returns the H5dataset object of the file loaded.

        Returns
        -------
        smsh5.H5dataset
        """
        # return self.treemodel.data(self.treemodel.index(0, 0), Qt.UserRole)
        return self.dataset_index.data(Qt.UserRole)

    def set_data_loaded(self):
        self.data_loaded = True

    def open_save_file_version_outdated(self):
        msg_box = QMessageBox(parent=self)
        msg_box.setWindowTitle("Save File Outdated")
        msg_box.setText("The save file is outdated. Please reload *.h5 file instead.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def open_file_thread_complete(self, thread: ProcessThread = None, irf=False) -> None:
        """ Is called as soon as all of the threads have finished. """

        if self.data_loaded and not irf:
            self.current_dataset = self.tree2dataset()
            self.treeViewParticles.expandAll()
            self.treeViewParticles.setCurrentIndex(self.part_index[0])
            self.current_particle = self.tree2particle(0)
            any_spectra = any([part.has_spectra for part in self.current_dataset.particles])
            if any_spectra:
                self.current_dataset.has_spectra = True

            if not any([p.has_levels for p in self.current_dataset.particles]):
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

            if self.data_loaded:
                self.actionSave_Analysis.setEnabled(True)
                self.actionSelect_All.setEnabled(True)
                self.actionInvert_Selection.setEnabled(True)
                self.actionDeselect_All.setEnabled(True)
                self.actionRange_Selection.setEnabled(True)
                self.menuIntensity.setEnabled(True)
                self.menuLifetime.setEnabled(True)
                self.chbEx_Use_ROI.setEnabled(True)
                self.chbInt_Show_ROI.setEnabled(True)
                self.chbGroup_Use_ROI.setEnabled(True)
                self.chbEx_Trace.setEnabled(True)
                self.chbEx_Hist.setEnabled(True)
                self.chbEx_Plot_Intensity.setEnabled(True)
                self.rdbInt_Only.setEnabled(True)
                self.chbEx_Plot_Lifetimes.setEnabled(True)
                self.rdbHist_Only.setEnabled(True)
                self.actionRange_Selection.setEnabled(True)
                self.set_export_options()

                self.reset_gui()

                self.chbInt_Show_ROI.setCheckState(1)
                self.display_data()

            logger.info('File opened')

    def set_export_options(self):
        particles = self.get_checked_particles()
        particles.append(self.current_particle)
        # if len(particles) == 0:
        #     particles = [self.current_particle]

        all_have_levels = all([p.has_levels for p in particles])
        all_have_groups = all([p.has_groups for p in particles])
        all_have_lifetimes = all([p.has_fit_a_lifetime for p in particles])
        all_have_raster_scans = all([p.has_raster_scan for p in particles])
        all_have_spectra = all([p.has_spectra for p in particles])

        self.chbEx_Levels.setEnabled(all_have_levels)
        self.chbEx_DF_Levels.setEnabled(all_have_levels)
        if self.chbEx_DF_Levels.isChecked() and all_have_lifetimes:
            self.chbEx_DF_Levels_Lifetimes.setEnabled(True)
        else:
            self.chbEx_DF_Levels_Lifetimes.setEnabled(False)

        self.chbEx_Grouped_Levels.setEnabled(all_have_groups)
        self.chbEx_DF_Grouped_Levels.setEnabled(all_have_groups)
        if self.chbEx_DF_Grouped_Levels.isChecked() and all_have_groups:
            self.chbEx_DF_Grouped_Levels_Lifetimes.setEnabled(True)
        else:
            self.chbEx_DF_Grouped_Levels_Lifetimes.setEnabled(False)

        self.chbEx_Grouping_Info.setEnabled(all_have_groups)
        self.chbEx_Grouping_Results.setEnabled(all_have_groups)
        self.chbEx_DF_Grouping_Info.setEnabled(all_have_groups)

        # Hists always enabled
        self.chbEx_Lifetimes.setEnabled(all_have_lifetimes)

        self.chbEx_Spectra_2D.setEnabled(all_have_spectra)
        self.chbEx_Spectra_Fitting.setEnabled(False)  # Add when spectra analysis added
        self.chbEx_Spectra_Traces.setEnabled(False)  # Add when spectra analysis added

        # Int plot always enalbed
        self.rdbInt_Only.setEnabled(True)
        if not (all_have_groups or all_have_levels):
            self.rdbInt_Only.setChecked(True)
        self.rdbWith_Levels.setEnabled(all_have_levels)
        self.rdbAnd_Groups.setEnabled(all_have_groups)

        # Always able to export traces
        self.chbEx_DF_Traces.setEnabled(True)

        # self.chbEx_Hist.setEnabled(all_have_lifetimes)  # Shouldn't this be true always?

        self.chbEx_Plot_Group_BIC.setEnabled(all_have_groups)

        self.chbEx_Plot_Lifetimes.setEnabled(all_have_lifetimes)
        self.rdbWith_Fit.setEnabled(all_have_lifetimes)
        self.rdbAnd_Residuals.setEnabled(all_have_lifetimes)
        self.chbEx_Plot_Lifetimes_Only_Groups.setEnabled(all_have_lifetimes)

        self.chbEx_Plot_Spectra.setEnabled(all_have_spectra)

        self.chbEx_Raster_Scan_2D.setEnabled(all_have_raster_scans)
        self.chbEx_Plot_Raster_Scans.setEnabled(all_have_raster_scans)

    def select_all_export_options(self):
        self.chbEx_Trace.setChecked(self.chbEx_Trace.isEnabled())
        self.chbEx_Levels.setChecked(self.chbEx_Levels.isEnabled())
        self.chbEx_Grouped_Levels.setChecked(self.chbEx_Grouped_Levels.isEnabled())
        self.chbEx_Grouping_Info.setChecked(self.chbEx_Grouping_Info.isEnabled())
        self.chbEx_Grouping_Results.setChecked(self.chbEx_Grouping_Results.isEnabled())
        self.chbEx_DF_Grouping_Info.setChecked(self.chbEx_DF_Grouping_Info.isEnabled())
        self.chbEx_Hist.setChecked(self.chbEx_Hist.isEnabled())
        self.chbEx_Lifetimes.setChecked(self.chbEx_Lifetimes.isEnabled())
        self.chbEx_Spectra_2D.setChecked(self.chbEx_Spectra_2D.isEnabled())
        # self.chbEx_Spectra_Fitting.setChecked(self.chbEx_Spectra_Fitting.isEnabled())
        # self.chbEx_Sptecra_Traces.setChecked(self.chbEx_Sptecra_Traces.isEnabled())
        self.chbEx_Plot_Intensity.setChecked(self.chbEx_Plot_Intensity.isEnabled())
        self.rdbInt_Only.setChecked(self.rdbInt_Only.isEnabled())
        self.rdbWith_Levels.setChecked(self.rdbWith_Levels.isEnabled())
        self.rdbAnd_Groups.setChecked(self.rdbAnd_Groups.isEnabled())
        self.chbEx_Plot_Group_BIC.setChecked(self.chbEx_Plot_Group_BIC.isEnabled())
        self.chbEx_Plot_Lifetimes.setChecked(self.chbEx_Plot_Lifetimes.isEnabled())
        self.rdbHist_Only.setChecked(self.rdbHist_Only.isEnabled())
        self.rdbWith_Fit.setChecked(self.rdbWith_Fit.isEnabled())
        self.rdbAnd_Residuals.setChecked(self.rdbAnd_Residuals.isEnabled())
        self.chbEx_Plot_Spectra.setChecked(self.chbEx_Plot_Spectra.isEnabled())
        self.chbEx_Raster_Scan_2D.setChecked(self.chbEx_Raster_Scan_2D.isEnabled())
        self.chbEx_Plot_Raster_Scans.setChecked(self.chbEx_Plot_Raster_Scans.isEnabled())

        # Not sure if there is only duplication of below
        # self.chbEx_DF_Levels.setChecked(self.chbEx_DF_Levels.isEnabled())
        # self.chbEx_DF_Levels_Lifetimes.setChecked(self.chbEx_DF_Levels_Lifetimes.isEnabled())
        # self.chbEx_DF_Grouped_Levels.setChecked(self.chbEx_DF_Grouped_Levels.isEnabled())
        # self.chbEx_DF_Grouped_Levels_Lifetimes.setChecked(
        #     self.chbEx_DF_Grouped_Levels_Lifetimes.isEnabled())
        # self.chbEx_DF_Grouping_Info.setChecked(self.chbEx_DF_Grouping_Info.isEnabled())

        self.chbEx_DF_Traces.setChecked(self.chbEx_DF_Traces.isEnabled())
        self.chbEx_DF_Levels.setChecked(self.chbEx_DF_Levels.isEnabled())
        self.chbEx_DF_Levels_Lifetimes.setChecked(self.chbEx_DF_Levels_Lifetimes.isEnabled())
        self.chbEx_DF_Grouped_Levels.setChecked(self.chbEx_DF_Grouped_Levels.isEnabled())
        self.chbEx_DF_Grouped_Levels_Lifetimes.setChecked(
            self.chbEx_DF_Grouped_Levels_Lifetimes.isEnabled())
        self.chbEx_DF_Grouping_Info.setChecked(self.chbEx_DF_Grouping_Info.isEnabled())

    def select_all_plots_export_options(self):
        self.chbEx_Plot_Intensity.setChecked(self.chbEx_Plot_Intensity.isEnabled())
        self.rdbInt_Only.setChecked(self.rdbInt_Only.isEnabled())
        self.rdbWith_Levels.setChecked(self.rdbWith_Levels.isEnabled())
        self.rdbAnd_Groups.setChecked(self.rdbAnd_Groups.isEnabled())
        self.chbEx_Plot_Group_BIC.setChecked(self.chbEx_Plot_Group_BIC.isEnabled())
        self.chbEx_Plot_Lifetimes.setChecked(self.chbEx_Plot_Lifetimes.isEnabled())
        self.rdbHist_Only.setChecked(self.rdbHist_Only.isEnabled())
        self.rdbWith_Fit.setChecked(self.rdbWith_Fit.isEnabled())
        self.rdbAnd_Residuals.setChecked(self.rdbAnd_Residuals.isEnabled())
        self.chbEx_Plot_Spectra.setChecked(self.chbEx_Plot_Spectra.isEnabled())
        self.chbEx_Plot_Raster_Scans.setChecked(self.chbEx_Plot_Raster_Scans.isEnabled())

    def select_all_dataframes_export_options(self):
        self.chbEx_DF_Traces.setChecked(self.chbEx_DF_Traces.isEnabled())
        self.chbEx_DF_Levels.setChecked(self.chbEx_DF_Levels.isEnabled())
        self.chbEx_DF_Levels_Lifetimes.setChecked(self.chbEx_DF_Levels_Lifetimes.isEnabled())
        self.chbEx_DF_Grouped_Levels.setChecked(self.chbEx_DF_Grouped_Levels.isEnabled())
        self.chbEx_DF_Grouped_Levels_Lifetimes.setChecked(
            self.chbEx_DF_Grouped_Levels_Lifetimes.isEnabled())
        self.chbEx_DF_Grouping_Info.setChecked(self.chbEx_DF_Grouping_Info.isEnabled())

    @pyqtSlot(Exception)
    def open_file_error(self, err: Exception):
        # logger.error(err)
        pass

    def detect_remove_bursts(self, mode: str = None) -> None:
        if mode == 'current':
            particles = [self.current_particle]
        elif mode == 'selected':
            particles = self.get_checked_particles()
        else:
            particles = self.current_dataset.particles

        for part in particles:
            part.cpts.check_burst()
        self.remove_bursts(mode=mode)

    def remove_bursts(self, mode: str = None, confirm: bool = True) -> None:
        if mode == 'current':
            particles = [self.current_particle]
        elif mode == 'selected':
            particles = self.get_checked_particles()
        else:
            particles = self.current_dataset.particles

        has_burst = [particle.has_burst for particle in particles]
        if sum(has_burst):
            if confirm:
                msgbx = TimedMessageBox(30, parent=self)
                msgbx.setIcon(QMessageBox.Question)
                msgbx.setText("Would you like to remove the photon bursts?")
                msgbx.set_timeout_text(
                    message_pretime="(Removing photon bursts in ",
                    message_posttime=" seconds)")
                msgbx.setWindowTitle("Photon bursts detected")
                msgbx.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
                msgbx.setDefaultButton(QMessageBox.Yes)
                msgbx.show()
                msgbx_result, _ = msgbx.exec()
            if not confirm or msgbx_result == QMessageBox.Yes:
                for particle in particles:
                    if particle.has_burst:
                        particle.cpts.remove_bursts()
                        particle.makelevelhists()
                    if particle.has_groups:
                        particle.remove_and_reset_grouping()

            self.display_data()

    def restore_bursts(self, mode: str = None) -> None:
        if mode == 'current':
            particles = [self.current_particle]
        elif mode == 'selected':
            particles = self.get_checked_particles()
        else:
            particles = self.current_dataset.particles

        for part in particles:
            if part.cpts.bursts_deleted is not None:
                part.cpts.restore_bursts()
                part.makelevelhists()
                if part.has_groups:
                    part.remove_and_reset_grouping()

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
        for ind in range(self.treemodel.rowCount(self.dataset_index)):
            if self.part_nodes[ind].checked():
                checked.append((ind, self.part_nodes[ind]))
                # checked_nums.append(ind)
                # checked_particles.append(self.part_nodes[ind])
        return checked

    def get_checked_nums(self):
        checked_nums = list()
        for ind in range(self.treemodel.rowCount(self.dataset_index)):
            if self.part_nodes[ind].checked():
                checked_nums.append(ind + 1)
        return checked_nums

    def get_checked_particles(self):
        checked_particles = list()
        for ind in range(self.treemodel.rowCount(self.dataset_index)):
            if self.part_nodes[ind].checked():
                checked_particles.append(self.tree2particle(ind))
        return checked_particles

    def set_particle_check_state(self, particle_number: int, set_checked: bool):
        self.part_nodes[particle_number].setChecked(set_checked)

    def set_level_resolved(self):
        self.current_dataset.level_resolved = True
        # print(self.level_resolved)

    def gui_plot_intensity_clicked(self, new_value):
        self.frmPlot_Int_Selection.setEnabled(new_value)

    def gui_plot_lifetime_clicked(self, new_value):
        self.frmPlot_Lifetime_Selection.setEnabled(new_value)

    def gui_export(self, mode: str = None):
        self.lock = Lock()
        f_dir = QFileDialog.getExistingDirectory(self)
        export_worker = ExportWorker(mainwindow=self, mode=mode, lock=self.lock, f_dir=f_dir)
        sigs = export_worker.signals
        sigs.start_progress.connect(self.start_progress)
        sigs.progress.connect(self.update_progress)
        sigs.end_progress.connect(self.end_progress)
        sigs.status_message.connect(self.status_message)
        sigs.error.connect(self.error_handler)

        sigs.plot_trace_lock.connect(self.int_controller.plot_trace)
        sigs.plot_trace_export_lock.connect(self.int_controller.plot_trace)
        sigs.plot_levels_lock.connect(self.int_controller.plot_levels)
        sigs.plot_levels_export_lock.connect(self.int_controller.plot_levels)
        sigs.plot_group_bounds_export_lock.connect(self.int_controller.plot_group_bounds)
        sigs.plot_grouping_bic_export_lock.connect(self.grouping_controller.plot_group_bic)
        sigs.plot_decay_lock.connect(self.lifetime_controller.plot_decay)
        sigs.plot_decay_export_lock.connect(self.lifetime_controller.plot_decay)
        sigs.plot_convd_lock.connect(self.lifetime_controller.plot_convd)
        sigs.plot_convd_export_lock.connect(self.lifetime_controller.plot_convd)
        sigs.plot_decay_convd_export_lock.connect(self.lifetime_controller.plot_decay_and_convd)
        sigs.plot_decay_convd_residuals_export_lock.connect(
            self.lifetime_controller.plot_decay_convd_and_hist)
        sigs.show_residual_widget_lock.connect(self.lifetime_controller.show_residuals_widget)
        sigs.plot_residuals_export_lock.connect(self.lifetime_controller.plot_residuals)
        sigs.plot_spectra_export_lock.connect(self.spectra_controller.plot_spectra)
        sigs.plot_raster_scan_export_lock.connect(self.raster_scan_controller.plot_raster_scan)

        self.threadpool.start(export_worker)

    def convert_pt3_dialog(self):
        convert_pt3 = ConvertPt3Dialog(mainwindow=self)
        convert_pt3.exec()

    def range_selection(self):
        range_selection_dialog = RangeSelectionDialog(main_window=self)
        if range_selection_dialog.exec_():
            selection_indexes = range_selection_dialog.get_selection(max_range=len(self.part_nodes))
            mode_only, mode_add, mode_remove, _ = range_selection_dialog.get_mode()
            if max(selection_indexes) > len(self.part_nodes):
                msg = QMessageBox(self)
                msg.setWindowTitle("Range Selection")
                msg.setText("Selection out of bounds!")
                msg.setIcon(QMessageBox.Warning)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec()
            else:
                for i, node in enumerate(self.part_nodes):
                    is_checked = node.checked()
                    if mode_only:
                        if i + 1 in selection_indexes:
                            is_checked = True
                        else:
                            is_checked = False
                    elif mode_add:
                        if i + 1 in selection_indexes:
                            is_checked = True
                    elif mode_remove:
                        if i + 1 in selection_indexes:
                            is_checked = False
                    else:
                        if i + 1 not in selection_indexes:
                            is_checked = True
                        else:
                            is_checked = False

                    if is_checked != node.checked():
                        node.setChecked(is_checked)
                num_checked = sum([node.checked() for node in self.part_nodes])
                self.lblNum_Selected.setText(str(num_checked))

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
        enable_levels = False
        if new_state:
            enable_levels = self.current_dataset.has_levels
        self.actionTrim_Dead_Traces.setEnabled(enable_levels)
        self.chbGroup_Auto_Apply.setEnabled(enable_levels)

        # Lifetime
        self.tabLifetime.setEnabled(new_state)
        self.btnFitParameters.setEnabled(new_state)
        self.btnLoadIRF.setEnabled(new_state)
        if new_state:
            enable_fitting = self.lifetime_controller.irf_loaded
        else:
            enable_fitting = new_state
        self.chbHasIRF.setChecked(self.lifetime_controller.irf_loaded)
        self.btnFitCurrent.setEnabled(enable_fitting)
        self.btnFit.setEnabled(enable_fitting)
        self.btnFitAll.setEnabled(enable_fitting)
        self.btnFitSelected.setEnabled(enable_fitting)
        self.btnNextLevel.setEnabled(enable_levels)
        self.btnPrevLevel.setEnabled(enable_levels)
        # print(enable_levels)

        # Spectral
        if self.current_dataset and self.current_dataset.has_spectra:
            self.tabSpectra.setEnabled(True)
            self.btnSubBackground.setEnabled(new_state)
        else:
            self.tabSpectra.setEnabled(False)

    def error_handler(self, e: Exception):
        raise e


def display_on():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
    logger.info('Execution State set to Always On')


def display_reset():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    logger.info('Execution State Reset')
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

    # Create version file for distribution. Or use the command bellow:
    # create-version-file version.yml --outfile versionfile.txt --version SMS_VERSION
    if '--dev' in sys.argv:
        try:
            # noinspection PyUnresolvedReferences
            import pyinstaller_versionfile

            pyinstaller_versionfile.create_versionfile_from_input_file(
                output_file="versionfile.txt",
                input_file="version.yml",
                version=SMS_VERSION)
        except ImportError as e:
            pass

    if '--debug' not in sys.argv:
        freeze_support()
        Process(target=main).start()
    else:
        main()
