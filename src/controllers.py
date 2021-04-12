
from __future__ import annotations
from typing import Union, List, TYPE_CHECKING
from copy import copy

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QWidget, QFrame, QInputDialog, QFileDialog
import time
from multiprocessing import Queue

if TYPE_CHECKING:
    from main import MainWindow

from my_logger import setup_logger
from smsh5 import H5dataset, Particle, ParticleAllHists
from tcspcfit import FittingParameters, FittingDialog
from threads import WorkerFitLifetimes, WorkerGrouping, WorkerResolveLevels, \
    ProcessThread, ProcessTask, ProcessTaskResult
from thread_tasks import OpenFile

logger = setup_logger(__name__)


class IntController(QObject):

    def __init__(self, mainwindow: MainWindow,
                 int_widget: pg.PlotWidget,
                 int_hist_container: QWidget,
                 int_hist_line: QFrame,
                 int_hist_widget: pg.PlotWidget,
                 lifetime_widget: pg.PlotWidget,
                 groups_int_widget: pg.PlotWidget,
                 groups_hist_widget: pg.PlotWidget):
        super().__init__()
        self.mainwindow = mainwindow
        self.resolve_mode = None
        self.results_gathered = False

        self.int_widget = int_widget
        self.int_plot = int_widget.getPlotItem()
        self.setup_widget(self.int_widget)
        self.setup_plot(self.int_plot)

        self.int_hist_container = int_hist_container
        self.show_int_hist = self.mainwindow.chbInt_Show_Hist.isChecked()
        self.int_hist_line = int_hist_line
        self.int_hist_widget = int_hist_widget
        self.int_hist_plot = int_hist_widget.getPlotItem()
        self.setup_widget(self.int_hist_widget)
        self.setup_plot(self.int_hist_plot, is_int_hist=True)

        self.lifetime_widget = lifetime_widget
        self.lifetime_plot = lifetime_widget.getPlotItem()
        self.setup_widget(self.lifetime_widget)
        self.setup_plot(self.lifetime_plot)

        self.group_int_widget = groups_int_widget
        self.groups_int_plot = groups_int_widget.getPlotItem()
        self.setup_widget(self.group_int_widget)
        self.setup_plot(self.groups_int_plot)

        self.groups_hist_widget = groups_hist_widget
        self.groups_hist_plot = groups_hist_widget.getPlotItem()
        self.setup_widget(self.groups_hist_widget)
        self.setup_plot(self.groups_hist_plot, is_group_hist=True)

        # Setup axes and limits
        # self.groups_hist_plot.getAxis('bottom').setLabel('Relative Frequency')

        self.confidence_index = {
            0: 99,
            1: 95,
            2: 90,
            3: 69}

    def setup_plot(self, plot_item: pg.PlotItem,
                   is_int_hist: bool = False,
                   is_group_hist: bool = False):

        # Set axis label bold and size
        axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)

        left_axis = plot_item.getAxis('left')
        bottom_axis = plot_item.getAxis('bottom')

        left_axis.setPen(axis_line_pen)
        bottom_axis.setPen(axis_line_pen)

        font = left_axis.label.font()
        # font.setBold(True)
        font.setPointSize(10)

        left_axis.label.setFont(font)
        bottom_axis.label.setFont(font)

        if is_int_hist or is_group_hist:
            # Setup axes and limits
            left_axis.setStyle(showValues=False)
            bottom_axis.setStyle(showValues=False)
            bottom_axis.setLabel('Relative Frequency')
            if is_int_hist:
                plot_item.setYLink(self.int_plot.getViewBox())
            else:
                plot_item.setYLink(self.groups_int_plot.getViewBox())
            plot_item.vb.setLimits(xMin=0, xMax=1, yMin=0)
        else:
            left_axis.setLabel('Intensity', 'counts/100ms')
            bottom_axis.setLabel('Time', 's')
            plot_item.vb.setLimits(xMin=0, yMin=0)

    @staticmethod
    def setup_widget(plot_widget: pg.PlotWidget):

        # Set widget background and antialiasing
        plot_widget.setBackground(background=None)
        plot_widget.setAntialiasing(True)

    def hide_unhide_hist(self):

        self.show_int_hist = self.mainwindow.chbInt_Show_Hist.isChecked()

        if self.show_int_hist:
            self.int_hist_container.show()
            self.int_hist_line.show()
            self.plot_hist()
        else:
            self.int_hist_container.hide()
            self.int_hist_line.hide()

    def gui_apply_bin(self):
        """ Changes the bin size of the data of the current particle and then displays the new trace. """
        try:
            self.mainwindow.currentparticle.binints(self.get_bin())
        except Exception as err:
            logger.error('Error Occured:')
        else:
            self.mainwindow.display_data()
            self.mainwindow.repaint()
            logger.info('Single trace binned')

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
            logger.info('Error Occured: ' + str(err))
        else:
            self.plot_trace()
            self.mainwindow.repaint()
            logger.info('All traces binned')

    def ask_end_time(self):
        """ Prompts the user to supply an end time."""

        end_time_s, ok = QInputDialog.getDouble(self.mainwindow, 'End Time',
                                                'Provide end time in seconds', 0, 1, 10000, 3)
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

    def plot_trace(self, particle: Particle = None, for_export: bool = False) -> None:
        """ Used to display the trace from the absolute arrival time data of the current particle. """

        try:
            # self.currentparticle = self.treemodel.data(self.current_ind, Qt.UserRole)
            if particle is None:
                particle = self.mainwindow.currentparticle
            trace = particle.binnedtrace.intdata
            times = particle.binnedtrace.inttimes / 1E3
        except AttributeError:
            logger.error('No trace!')
        else:
            plot_pen = QPen()
            plot_pen.setCosmetic(True)
            if for_export:
                cur_tab_name = 'tabIntensity'
            else:
                cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()

            if cur_tab_name != 'tabSpectra':
                if cur_tab_name == 'tabIntensity':
                    plot_item = self.int_plot
                    plot_pen.setWidthF(1.5)
                    plot_pen.setColor(QColor('green'))
                elif cur_tab_name == 'tabLifetime':
                    plot_item = self.lifetime_plot
                    plot_pen.setWidthF(1.1)
                    plot_pen.setColor(QColor('green'))
                elif cur_tab_name == 'tabGrouping':
                    plot_item = self.groups_int_plot
                    plot_pen.setWidthF(1.1)
                    plot_pen.setColor(QColor(0, 0, 0, 50))

                plot_pen.setJoinStyle(Qt.RoundJoin)

                plot_item.clear()
                unit = 'counts/' + str(self.get_bin()) + 'ms'
                plot_item.getAxis('left').setLabel(text='Intensity', units=unit)
                plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=times[-1])
                plot_item.plot(x=times, y=trace, pen=plot_pen, symbol=None)

    def plot_levels(self, particle: Particle = None, for_export: bool = False):
        """ Used to plot the resolved intensity levels of the current particle. """
        if particle is None:
            particle = self.mainwindow.currentparticle
        if not particle.has_levels:
            return
        try:
            level_ints, times = particle.levels2data()
            level_ints = level_ints * self.get_bin() / 1E3
        except AttributeError:
            logger.error('No levels!')
        # else:
        plot_pen = QPen()

        if for_export:
            cur_tab_name = 'tabIntensity'
        else:
            cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()

        if cur_tab_name == 'tabIntensity':
            plot_item = self.int_plot
            # pen_width = 1.5
            plot_pen.setWidthF(1.5)
            plot_pen.setColor(QColor('black'))
        elif cur_tab_name == 'tabLifetime':
            plot_item = self.lifetime_plot
            # pen_width = 1.1
            plot_pen.setWidthF(1.1)
            plot_pen.setColor(QColor('black'))
        elif cur_tab_name == 'tabGrouping':
            plot_item = self.groups_int_plot
            plot_pen.setWidthF(1)
            plot_pen.setColor(QColor(0, 0, 0, 100))
        else:
            return

        plot_pen.brush()
        plot_pen.setJoinStyle(Qt.RoundJoin)
        plot_pen.setCosmetic(True)

        plot_item.plot(x=times, y=level_ints, pen=plot_pen, symbol=None)

        if self.mainwindow.current_level is not None:
            current_ints, current_times = particle.current2data(
                self.mainwindow.current_level)
            current_ints = current_ints * self.get_bin() / 1E3
            # print(current_ints, current_times)

            if not (current_ints[0] == np.inf or current_ints[1] == np.inf):
                plot_pen.setColor(QColor('red'))
                plot_pen.setWidthF(3)
                plot_item.plot(x=current_times, y=current_ints, pen=plot_pen, symbol=None)
            else:
                logger.info('Infinity in level')

    def plot_hist(self, particle: Particle = None, for_export: bool = False):
        if particle is None:
            particle = self.mainwindow.currentparticle
        try:
            int_data = particle.binnedtrace.intdata
        except AttributeError:
            logger.error('No trace!')
        else:
            plot_pen = QPen()
            plot_pen.setColor(QColor(0, 0, 0, 0))

            if for_export:
                cur_tab_name = 'tabIntensity'
            else:
                cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()

            if cur_tab_name == 'tabIntensity':
                if self.show_int_hist or for_export:
                    plot_item = self.int_hist_plot
                else:
                    return
            elif cur_tab_name == 'tabGrouping':
                plot_item = self.groups_hist_plot
            else:
                return

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

                level_ints *= self.mainwindow.currentparticle.bin_size / 1000
                dwell_times = [level.dwell_time_s for level in
                               self.mainwindow.currentparticle.levels]
                level_freq, level_hist_bins = np.histogram(np.negative(level_ints), bins=bin_edges,
                                                           weights=dwell_times, density=True)
                level_freq /= np.max(level_freq)
                level_hist = pg.PlotCurveItem(x=level_hist_bins, y=level_freq, stepMode=True,
                                              pen=plot_pen, fillLevel=0, brush=(0, 0, 0, 255))

                level_hist.rotate(-90)
                plot_item.addItem(level_hist)


    # def export_particle_plot(self, particle:Particle, width:int = 800):
    #     plt =

    def plot_group_bounds(self, particle: Particle = None, for_export: bool = False):
        if particle is None:
            particle = self.mainwindow.currentparticle

        if for_export:
            cur_tab_name = 'tabIntensity'
        else:
            cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()

        if cur_tab_name == 'tabIntensity' or cur_tab_name == 'tabGrouping':
            if not particle.has_groups \
                    or particle.ahca.best_step.single_level\
                    or particle.ahca.selected_step.num_groups < 2:
                return
            try:
                groups = particle.groups
                group_bounds = particle.groups_bounds
            except AttributeError:
                logger.error('No groups!')

            if cur_tab_name == 'tabIntensity':
                if self.mainwindow.chbInt_Show_Groups.isChecked() or for_export:
                    int_plot = self.int_plot
                else:
                    return
            elif cur_tab_name == 'tabGrouping':
                int_plot = self.groups_int_plot

            int_conv = particle.bin_size / 1000

            for i, bound in enumerate(group_bounds):
                if i % 2:
                    bound = (bound[0] * int_conv, bound[1] * int_conv)
                    int_plot.addItem(
                        pg.LinearRegionItem(values=bound, orientation='horizontal', movable=False,
                                            pen=QPen().setWidthF(0)))

            line_pen = QPen()
            line_pen.setWidthF(1)
            line_pen.setStyle(Qt.DashLine)
            line_pen.brush()
            # plot_pen.setJoinStyle(Qt.RoundJoin)
            line_pen.setColor(QColor(0, 0, 0, 150))
            line_pen.setCosmetic(True)
            line_times = [0, particle.dwell_time]
            for group in groups:
                g_ints = [group.int * int_conv, group.int * int_conv]
                int_plot.plot(x=line_times, y=g_ints, pen=line_pen, symbol=None)

    def plot_all(self):
        self.plot_trace()
        self.plot_levels()
        self.plot_hist()
        self.plot_group_bounds()

    def export_image(self, particle: Particle,
                     file_path: str,
                     only_levels: bool = True,
                     with_groups: bool = True):
        int_height = self.int_widget.geometry().height()
        hist_height = self.int_hist_widget.geometry().height()

        self.plot_trace(particle=particle, for_export=True)
        self.plot_levels(particle=particle, for_export=True)
        exporter = pg.exporters.ImageExporter(self.int_plot)
        # TODO: Here

    def start_resolve_thread(self, mode: str = 'current', thread_finished=None,
                             end_time_s=None) -> None:
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

        mw = self.mainwindow
        if thread_finished is None:
            if mw.data_loaded:
                thread_finished = self.resolve_thread_complete
            else:
                thread_finished = mw.open_file_thread_complete

        _, conf = self.get_gui_confidence()
        data = mw.tree2dataset()
        currentparticle = mw.currentparticle
        # print(currentparticle)

        self.resolve_mode = mode
        if mode == 'current':
            status_message = "Resolving current particle levels..."
            cpt_objs = [currentparticle.cpts]
        elif mode == 'selected':
            status_message = "Resolving selected particle levels..."
            checked_parts = mw.get_checked_particles()
            cpt_objs = [part.cpts for part in checked_parts]
        elif mode == 'all':
            status_message = "Resolving all particle levels..."
            cpt_objs = [part.cpts for part in data.particles]
        else:
            logger.error(msg="Provided mode not valid")
            raise TypeError

        r_process_thread = ProcessThread()
        r_process_thread.add_tasks_from_methods(objects=cpt_objs,
                                                method_name='run_cpa',
                                                args=(conf, True))

        r_process_thread.signals.start_progress.connect(mw.start_progress)
        r_process_thread.signals.status_update.connect(mw.status_message)
        r_process_thread.signals.step_progress.connect(mw.update_progress)
        r_process_thread.signals.end_progress.connect(mw.end_progress)
        r_process_thread.signals.error.connect(self.error)
        r_process_thread.signals.results.connect(self.gather_replace_results)
        r_process_thread.signals.finished.connect(thread_finished)
        r_process_thread.worker_signals.reset_gui.connect(mw.reset_gui)
        r_process_thread.worker_signals.level_resolved.connect(mw.set_level_resolved)
        r_process_thread.status_message = status_message

        mw.threadpool.start(r_process_thread)
        mw.active_threads.append(r_process_thread)


    def gather_replace_results(self, results: Union[List[ProcessTaskResult], ProcessTaskResult]):
        particles = self.mainwindow.currentparticle.dataset.particles
        part_uuids = [part.uuid for part in particles]
        if type(results) is not list:
            results = [results]
        result_part_uuids = [result.new_task_obj.uuid for result in results]
        try:
            for num, result in enumerate(results):
                result_part_ind = part_uuids.index(result_part_uuids[num])
                target_particle = self.mainwindow.tree2particle(result_part_ind).cpts._particle
                result.new_task_obj._particle = target_particle
                result.new_task_obj._cpa._particle = target_particle
                target_particle.cpts = result.new_task_obj
            self.results_gathered = True
        except ValueError as e:
            logger.error(e)

    def resolve_thread_complete(self, thread: ProcessThread):
        count = 0

        while self.results_gathered is False:
            time.sleep(1)
            count += 1
            if count >= 5:
                logger.error(msg="Results gathering timeout")
                raise RuntimeError

        if self.mainwindow.currentparticle.has_levels:  # tree2dataset().cpa_has_run:
            self.mainwindow.tabGrouping.setEnabled(True)
        if self.mainwindow.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
            self.mainwindow.display_data()
        self.mainwindow.check_remove_bursts(mode=self.resolve_mode)
        self.mainwindow.chbEx_Levels.setEnabled(True)
        self.mainwindow.set_startpoint()
        self.mainwindow.level_resolved = True
        self.mainwindow.reset_gui()
        self.mainwindow.status_message("Done")
        logger.info('Resolving levels complete')

        self.results_gathered = False

    def get_gui_confidence(self):
        """ Return current GUI value for confidence percentage. """

        return [self.mainwindow.cmbConfIndex.currentIndex(),
                self.confidence_index[self.mainwindow.cmbConfIndex.currentIndex()]]


    def error(self, e):
        logger.error(e)


class LifetimeController(QObject):

    def __init__(self,
                 mainwindow: MainWindow,
                 lifetime_hist_widget: pg.PlotWidget,
                 residual_widget: pg.PlotWidget):
        super().__init__()
        self.mainwindow = mainwindow

        self.lifetime_hist_widget = lifetime_hist_widget
        self.life_hist_plot = lifetime_hist_widget.getPlotItem()
        self.setup_widget(self.lifetime_hist_widget)

        self.residual_widget = residual_widget
        self.residual_plot = residual_widget.getPlotItem()
        self.setup_widget(self.residual_widget)

        self.setup_plot(self.life_hist_plot)
        self.setup_plot(self.residual_plot)

        self.fitparamdialog = FittingDialog(self.mainwindow, self)
        self.fitparam = FittingParameters(self)
        self.irf_loaded = False

        self.first = 0
        self.startpoint = None
        self.tmin = 0

    def setup_plot(self, plot: pg.PlotItem):
        # Set axis label bold and size
        axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)
        plot.getAxis('left').setPen(axis_line_pen)
        plot.getAxis('bottom').setPen(axis_line_pen)
        plot.getAxis('left').label.font().setBold(True)
        plot.getAxis('bottom').label.font().setBold(True)
        plot.getAxis('left').label.font().setPointSize(16)
        plot.getAxis('bottom').label.font().setPointSize(16)

        # Setup axes and limits
        plot.getAxis('left').setLabel('Num. of occur.', 'counts/bin')
        plot.getAxis('bottom').setLabel('Decay time', 'ns')
        plot.getViewBox().setLimits(xMin=0, yMin=0)

    @staticmethod
    def setup_widget(plot_widget: pg.PlotWidget):
        # Set widget background and antialiasing
        plot_widget.setBackground(background=None)
        plot_widget.setAntialiasing(True)

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

        file_path = QFileDialog.getOpenFileName(self.mainwindow, 'Open HDF5 file', '',
                                                "HDF5 files (*.h5)")
        if file_path != ('', ''):  # fname will equal ('', '') if the user canceled.
            mw = self.mainwindow
            mw.status_message(message="Opening IRF file...")
            of_process_thread = ProcessThread(num_processes=1)
            of_process_thread.worker_signals.add_datasetindex.connect(mw.add_dataset)
            of_process_thread.worker_signals.add_particlenode.connect(mw.add_node)
            of_process_thread.worker_signals.add_all_particlenodes.connect(mw.add_all_nodes)
            of_process_thread.worker_signals.bin_size.connect(mw.set_bin_size)
            of_process_thread.worker_signals.data_loaded.connect(mw.set_data_loaded)
            of_process_thread.worker_signals.add_irf.connect(self.add_irf)
            of_process_thread.signals.status_update.connect(mw.status_message)
            of_process_thread.signals.start_progress.connect(mw.start_progress)
            of_process_thread.signals.set_progress.connect(mw.set_progress)
            of_process_thread.signals.step_progress.connect(mw.update_progress)
            of_process_thread.signals.add_progress.connect(mw.update_progress)
            of_process_thread.signals.end_progress.connect(mw.end_progress)
            of_process_thread.signals.error.connect(mw.error_handler)
            of_process_thread.signals.finished.connect(mw.reset_gui)

            of_obj = OpenFile(file_path=file_path, is_irf=True, tmin=self.tmin)
            of_process_thread.add_tasks_from_methods(of_obj, 'open_irf')
            mw.threadpool.start(of_process_thread)
            mw.active_threads.append(of_process_thread)


            # of_worker = WorkerOpenFile(fname, irf=True, tmin=self.tmin)
            # of_worker.signals.openfile_finished.connect(self.mainwindow.open_file_thread_complete)
            # of_worker.signals.start_progress.connect(self.mainwindow.start_progress)
            # of_worker.signals.progress.connect(self.mainwindow.update_progress)
            # of_worker.signals.auto_progress.connect(self.mainwindow.update_progress)
            # of_worker.signals.start_progress.connect(self.mainwindow.start_progress)
            # of_worker.signals.status_message.connect(self.mainwindow.status_message)
            # of_worker.signals.add_datasetindex.connect(self.mainwindow.add_dataset)
            # of_worker.signals.add_particlenode.connect(self.mainwindow.add_node)
            # of_worker.signals.reset_tree.connect(
            #     lambda: self.mainwindow.treemodel.modelReset.emit())
            # of_worker.signals.data_loaded.connect(self.mainwindow.set_data_loaded)
            # of_worker.signals.bin_size.connect(self.mainwindow.spbBinSize.setValue)
            # of_worker.signals.add_irf.connect(self.add_irf)
            #
            # self.mainwindow.threadpool.start(of_worker)

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
            logger.error("No decay")
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
                    logger.error('No Decay!')
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
            plot_item = self.life_hist_plot
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
                logger.error('No Decay!')
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
            # plot_item = self.pgLifetime.getPlotItem()
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
            self.life_hist_plot.plot(x=t, y=convd, pen=plot_pen, symbol=None)
            unit = 'ns with ' + str(currentparticle.channelwidth) + 'ns bins'
            self.life_hist_plot.getAxis('bottom').setLabel('Decay time', unit)
            self.life_hist_plot.getViewBox().setLimits(xMin=0, yMin=0, xMax=t[-1])

    def start_fitting_thread(self, mode: str = 'current') -> None:
        """
        Creates a worker to resolve levels.ckibacxxx

        Depending on the ``current_selected_all`` parameter the worker will be
        given the necessary parameter to fit the current, selected or all particles.

        Parameters
        ----------
        mode : {'current', 'selected', 'all'}
            Possible values are 'current' (default), 'selected', and 'all'.
        """

        assert mode in ['current', 'selected', 'all'], \
            "'resolve_all' and 'resolve_selected' can not both be given as parameters."

        mw = self.mainwindow
        if mode == 'current':
            status_message = "Fitting Levels for Current Particle..."
            particles = [mw.currentparticle]
        elif mode == 'selected':
            status_message = "Fitting Levels for Selected Particles..."
            particles = mw.get_checked_particles()
        elif mode == 'all':
            status_message = "Fitting Levels for All Particles..."
            particles = mw.tree2dataset().particles

        f_p = self.fitparam
        channelwidth = particles[0].channelwidth
        if f_p.start is None:
            start = None
        else:
            start = int(f_p.start / channelwidth)
        if f_p.end is None:
            end = None
        else:
            end = int(f_p.end / channelwidth)

        part_hists = list()
        for part in particles:
            part_hists.append(ParticleAllHists(particle=part))

        clean_fit_param = copy(self.fitparam)
        clean_fit_param.parent = None
        clean_fit_param.fpd = None

        f_process_thread = ProcessThread()
        f_process_thread.add_tasks_from_methods(objects=part_hists,
                                                method_name='fit_part_and_levels',
                                                args=(channelwidth, start, end, clean_fit_param))
        f_process_thread.signals.start_progress.connect(mw.start_progress)
        f_process_thread.signals.status_update.connect(mw.status_message)
        f_process_thread.signals.step_progress.connect(mw.update_progress)
        f_process_thread.signals.end_progress.connect(mw.end_progress)
        f_process_thread.signals.error.connect(self.error)
        f_process_thread.signals.results.connect(self.gather_replace_results)  # Todo
        f_process_thread.signals.finished.connect(self.fitting_thread_complete)
        f_process_thread.worker_signals.reset_gui.connect(mw.reset_gui)
        f_process_thread.status_message = status_message

        mw.threadpool.start(f_process_thread)
        mw.active_threads.append(f_process_thread)

    def gather_replace_results(self, results: Union[List[ProcessTaskResult], ProcessTaskResult]):
        particles = self.mainwindow.currentparticle.dataset.particles
        part_uuids = [part.uuid for part in particles]
        if type(results) is not list:
            results = [results]
        result_part_uuids = [result.new_task_obj.part_uuid for result in results]
        try:
            for num, result in enumerate(results):
                result_part_ind = part_uuids.index(result_part_uuids[num])
                target_particle = self.mainwindow.tree2particle(result_part_ind)

                target_hist = target_particle.histogram
                target_microtimes = target_hist.microtimes

                result.new_task_obj.part_hist.particle = target_particle
                result.new_task_obj.part_hist.microtimes = target_microtimes

                target_particle.histogram = result.new_task_obj.part_hist

                for num, res_hist in enumerate(result.new_task_obj.level_hists):
                    target_level = target_particle.levels[num]
                    target_level_microtimes = target_level.microtimes

                    res_hist.particle = target_particle
                    res_hist.microtimes = target_level_microtimes
                    res_hist.level = target_level

                    target_level.histogram = res_hist
        except ValueError as e:
            logger.error(e)

    def fitting_thread_complete(self, mode):
        if self.mainwindow.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
            self.mainwindow.display_data()
        self.mainwindow.chbEx_Lifetimes.setEnabled(False)
        self.mainwindow.chbEx_Lifetimes.setEnabled(True)
        self.mainwindow.chbEx_Hist.setEnabled(True)
        self.mainwindow.status_message("Done")
        # print(self.mainwindow.chbEx_Lifetimes.isChecked())
        logger.info('Fitting levels complete')

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

    def error(self, e):
        logger.error(e)


class GroupingController(QObject):

    def __init__(self, mainwidow: MainWindow, bic_plot_widget: pg.PlotWidget):
        super().__init__()
        self.mainwindow = mainwidow

        # self.groups_hist_widget = groups_hist_widget
        # self.groups_hist_plot = groups_hist_widget.addPlot()
        # self.groups_hist_widget.setBackground(background=None)

        self.bic_plot_widget = bic_plot_widget
        self.bic_scatter_plot = self.bic_plot_widget.getPlotItem()
        self.bic_plot_widget.setBackground(background=None)

        # Set axis label bold and size
        axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)

        self.bic_scatter_plot.getAxis('left').setPen(axis_line_pen)
        self.bic_scatter_plot.getAxis('bottom').setPen(axis_line_pen)
        self.bic_scatter_plot.getAxis('left').label.font().setBold(True)
        self.bic_scatter_plot.getAxis('bottom').label.font().setBold(True)
        self.bic_scatter_plot.getAxis('left').label.font().setPointSize(12)
        self.bic_scatter_plot.getAxis('bottom').label.font().setPointSize(12)

        self.bic_scatter_plot.getAxis('left').setLabel('BIC')
        self.bic_scatter_plot.getAxis('bottom').setLabel('Number of Groups')
        self.bic_scatter_plot.getViewBox().setLimits(xMin=0)

        self.all_bic_plots = None
        self.all_last_solutions = None

    def clear_bic(self):
        self.bic_scatter_plot.clear()

    def solution_clicked(self, plot, points):
        last_solution = self.all_last_solutions[self.mainwindow.currentparticle.dataset_ind]
        if last_solution != points[0]:
            curr_part = self.mainwindow.currentparticle
            point_num_groups = int(points[0].pos()[0])
            new_ind = curr_part.ahca.steps_num_groups.index(point_num_groups)
            curr_part.ahca.set_selected_step(new_ind)
            if last_solution:
                last_solution.setPen(pg.mkPen(width=1, color='k'))
            for p in points:
                p.setPen('c', width=2)
            last_solution = points[0]
            self.all_last_solutions[self.mainwindow.currentparticle.dataset_ind] = last_solution

            self.mainwindow.display_data()

    def plot_group_bic(self):
        cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()
        if cur_tab_name == 'tabGrouping':
            cur_part = self.mainwindow.currentparticle
            if cur_part.ahca.best_step.single_level:
                self.bic_plot_widget.getPlotItem().clear()
                return
            try:
                grouping_bics = cur_part.grouping_bics.copy()
                grouping_selected_ind = cur_part.grouping_selected_ind
                best_grouping_ind = cur_part.best_grouping_ind
                grouping_num_groups = cur_part.grouping_num_groups.copy()

            except AttributeError:
                logger.error('No groups!')

            if self.all_bic_plots is None and self.all_last_solutions is None:
                num_parts = self.mainwindow.tree2dataset().num_parts
                self.all_bic_plots = [None] * num_parts
                self.all_last_solutions = [None] * num_parts

            scat_plot_item = self.all_bic_plots[cur_part.dataset_ind]
            if scat_plot_item is None:
                spot_other_pen = pg.mkPen(width=1, color='k')
                spot_selected_pen = pg.mkPen(width=2, color='c')
                spot_other_brush = pg.mkBrush(color='k')
                spot_best_brush = pg.mkBrush(color='r')

                scat_plot_item = pg.ScatterPlotItem()
                bic_spots = []
                for i, g_bic in enumerate(grouping_bics):
                    if i == best_grouping_ind:
                        spot_brush = spot_best_brush
                    else:
                        spot_brush = spot_other_brush

                    if i == grouping_selected_ind:
                        spot_pen = spot_selected_pen
                    else:
                        spot_pen = spot_other_pen

                    bic_spots.append({'pos': (grouping_num_groups[i], g_bic),
                                      'size': 10,
                                      'pen': spot_pen,
                                      'brush': spot_brush})

                scat_plot_item.addPoints(bic_spots)

                self.all_bic_plots[cur_part.dataset_ind] = scat_plot_item
                best_solution = scat_plot_item.points()[best_grouping_ind]
                self.all_last_solutions[cur_part.dataset_ind] = best_solution
                scat_plot_item.sigClicked.connect(self.solution_clicked)

            self.bic_plot_widget.getPlotItem().clear()
            self.bic_scatter_plot.addItem(scat_plot_item)

    def gui_group_current(self):
        self.start_grouping_thread(mode='current')

    def gui_group_selected(self):
        self.start_grouping_thread(mode='selected')

    def gui_group_all(self):
        self.start_grouping_thread(mode='all')

    def gui_apply_groups_current(self):
        self.apply_groups()

    def gui_apply_groups_selected(self):
        self.apply_groups('selected')

    def gui_apply_groups_all(self):
        self.apply_groups('all')

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

        mw = self.mainwindow

        if mode == 'current':
            grouping_objs = [mw.currentparticle.ahca]
            status_message = "Grouping levels for current particle..."
        elif mode == 'selected':
            checked_particles = mw.get_checked_particles()
            grouping_objs = [particle.ahca for particle in checked_particles]
            status_message = "Grouping levels for selected particle..."
        elif mode == 'all':
            all_particles = mw.currentparticle.dataset.particles
            grouping_objs = [particle.ahca for particle in all_particles]
            status_message = "Grouping levels for all particle..."

        g_process_thread = ProcessThread()
        g_process_thread.add_tasks_from_methods(objects=grouping_objs, method_name='run_grouping')

        g_process_thread.signals.status_update.connect(mw.status_message)
        g_process_thread.signals.start_progress.connect(mw.start_progress)
        g_process_thread.signals.step_progress.connect(mw.update_progress)
        g_process_thread.signals.end_progress.connect(mw.end_progress)
        g_process_thread.signals.error.connect(self.error)
        g_process_thread.signals.results.connect(self.gather_replace_results)
        g_process_thread.signals.finished.connect(self.grouping_thread_complete)
        g_process_thread.worker_signals.reset_gui.connect(mw.reset_gui)
        g_process_thread.status_message = status_message

        self.mainwindow.threadpool.start(g_process_thread)

    def gather_replace_results(self, results: Union[List[ProcessTaskResult], ProcessTaskResult]):
        particles = self.mainwindow.currentparticle.dataset.particles
        part_uuids = [part.uuid for part in particles]
        if type(results) is not list:
            results = [results]
        result_part_uuids = [result.new_task_obj.uuid for result in results]
        try:
            for num, result in enumerate(results):
                result_part_ind = part_uuids.index(result_part_uuids[num])
                new_part = self.mainwindow.tree2particle(result_part_ind)
                result_ahca = result.new_task_obj
                result_ahca.particle = new_part
                result_ahca.best_step._particle = new_part
                for step in result_ahca.steps:
                    step._paricle = new_part
                new_part.ahca = result_ahca

            # self.results_gathered = True
        except ValueError as e:
            logger.error(e)

    def grouping_thread_complete(self, mode):
        if self.mainwindow.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
            self.mainwindow.display_data()
        self.mainwindow.status_message("Done")
        self.mainwindow.levels_grouped = True
        self.mainwindow.chbEx_Grouped_Levels.setEnabled(True)
        self.mainwindow.gbxExport_Groups.setEnabled(True)
        self.mainwindow.chbEx_Grouping_Info.setEnabled(True)
        self.mainwindow.chbEx_Grouping_Results.setEnabled(True)
        self.mainwindow.reset_gui()
        logger.info('Grouping levels complete')

    def apply_groups(self, mode: str = 'current'):
        if mode == 'current':
            particles = [self.mainwindow.currentparticle]
        elif mode == 'selected':
            particles = self.mainwindow.get_checked_particles()
        else:
            particles = self.mainwindow.currentparticle.dataset.particles

        bool_use = not all([part.using_group_levels for part in particles])
        for particle in particles:
            particle.using_group_levels = bool_use

        self.mainwindow.int_controller.plot_all()

    def error(self, e: Exception):
        logger.error(e)


class SpectraController(QObject):

    def __init__(self, mainwindow: MainWindow, spectra_widget: pg.ImageView):
        super().__init__()

        self.mainwindow = mainwindow
        self.spectra_widget = spectra_widget

        self.spectra_widget.setPredefinedGradient('plasma')

        # blue, red = Color('blue'), Color('red')
        # colours = blue.range_to(red, 256)
        # c_array = np.array([np.array(colour.get_rgb())*255 for colour in colours])
        # self._look_up_table = c_array.astype(np.uint8)

        # self.spectra_plot = spectra_widget.addPlot()
        # self.spectra_plot_item = pg.ImageItem()
        # self.spectra_plot.addItem(self.spectra_plot_item)

        # axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)
        # self.spectra_image_item.getAxis('left').setPen(axis_line_pen)
        # self.spectra_image_item.getAxis('bottom').setPen(axis_line_pen)
        #
        # # Set axis label bold and size
        # font = self.spectra_image_item.getAxis('left').label.font()
        # font.setBold(True)
        #
        # self.spectra_image_item.getAxis('left').label.setFont(font)
        # self.spectra_image_item.getAxis('bottom').label.setFont(font)
        #
        # self.spectra_image_item.getAxis('left').setLabel('X Range', 'um')
        # self.spectra_image_item.getAxis('bottom').setLabel('Y Range', '<span>&#181;</span>m')
        # self.spectra_image_item.getViewBox().setAspectLocked(lock=True, ratio=1)
        # self.spectra_image_item.getViewBox().setLimits(xMin=0, yMin=0)
        # self.spectra_plot.setContentsMargins(5, 5, 5, 5)

    def gui_sub_bkg(self):
        """ Used to subtract the background TODO: Explain the sub_background """

        print("gui_sub_bkg")

    def plot_spectra(self):
        # curr_part = self.mainwindow.currentparticle
        spectra_data = np.flip(self.mainwindow.currentparticle.spectra.data[:])
        spectra_data = spectra_data.transpose()
        data_shape = spectra_data.shape
        current_ratio = data_shape[1]/data_shape[0]
        self.spectra_widget.getView().setAspectLocked(False, current_ratio)

        self.spectra_widget.setImage(spectra_data)
        print('here')
        # self.spectra_widget.getImageItem().setLookupTable(self._look_up_table)

def resolve_levels(start_progress_sig: pyqtSignal, progress_sig: pyqtSignal,
                   status_sig: pyqtSignal, reset_gui_sig: pyqtSignal,
                   level_resolved_sig: pyqtSignal,
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
                logger.info(f'Busy Resolving Particle {num + 1}')
                part.cpts.run_cpa(confidence=conf, run_levels=True, end_time_s=end_time_s)
                progress_sig.emit()
            status_sig.emit('Done')
        except Exception as exc:
            raise RuntimeError("Couldn't resolve levels.") from exc

    level_resolved_sig.emit()
    data.makehistograms(progress=False)
    reset_gui_sig.emit()


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
            logger.info(f'Busy Grouping Particle {num + 1}')
            part.ahca.run_grouping()
            progress_sig.emit()
        status_sig.emit('Done')
    except Exception as exc:
        raise RuntimeError("Couldn't group levels.") from exc

    # grou.emit()
    # data.makehistograms(progress=False)
    # reset_gui_sig.emit()


# def fit_lifetimes(start_progress_sig: pyqtSignal, progress_sig: pyqtSignal,
#                   status_sig: pyqtSignal, reset_gui_sig: pyqtSignal,
#                   data, particles, fitparam, mode: str,
#                   resolve_selected=None) -> None:  # parallel: bool = False
#     """
#     TODO: edit the docstring
#     Resolves the levels in particles by finding the change points in the
#     abstimes data of a Particle instance.
#
#     Parameters
#     ----------
#     start_progress_sig : pyqtSignal
#         Used to call method to set up progress bar on G
#     progress_sig : pyqtSignal
#         Used to call method to increment progress bar on G
#     status_sig : pyqtSignal
#         Used to call method to show status bar message on G
#     mode : {'current', 'selected', 'all'}
#         Determines the mode that the levels need to be resolved on. Options are 'current', 'selected' or 'all'
#     resolve_selected : list[smsh5.Partilce]
#         A list of Particle instances in smsh5, that isn't the current one, to be resolved.
#     """
#
#     print(mode)
#     assert mode in ['current', 'selected', 'all'], \
#         "'resolve_all' and 'resolve_selected' can not both be given as parameters."
#
#     channelwidth = particles.channelwidth
#     if fitparam.start is None:
#         start = None
#     else:
#         start = int(fitparam.start / channelwidth)
#     if fitparam.end is None:
#         end = None
#     else:
#         end = int(fitparam.end / channelwidth)
#
#     if mode == 'current':  # Fit all levels in current particle
#         status_sig.emit('Fitting Levels for Selected Particles...')
#         start_progress_sig.emit(len(particles.levels))
#
#         for level in particles.levels:
#             try:
#                 if not level.histogram.fit(fitparam.numexp, fitparam.tau, fitparam.amp,
#                                            fitparam.shift / channelwidth, fitparam.decaybg,
#                                            fitparam.irfbg,
#                                            start, end, fitparam.addopt,
#                                            fitparam.irf, fitparam.shiftfix):
#                     pass  # fit unsuccessful
#                 progress_sig.emit()
#             except AttributeError:
#                 print("No decay")
#         particles.numexp = fitparam.numexp
#         status_sig.emit("Ready...")
#
#     elif mode == 'all':  # Fit all levels in all particles
#         status_sig.emit('Fitting All Particle Levels...')
#         start_progress_sig.emit(data.num_parts)
#
#         for particle in data.particles:
#             fit_part_and_levels(channelwidth, end, fitparam, particle, progress_sig, start)
#         status_sig.emit("Ready...")
#
#     elif mode == 'selected':  # Fit all levels in selected particles
#         assert resolve_selected is not None, \
#             'No selected particles provided.'
#         status_sig.emit('Resolving Selected Particle Levels...')
#         start_progress_sig.emit(len(resolve_selected))
#         for particle in resolve_selected:
#             fit_part_and_levels(channelwidth, end, fitparam, particle, progress_sig, start)
#         status_sig.emit('Ready...')
#
#     reset_gui_sig.emit()


# def fit_part_and_levels(channelwidth, end, fitparam, particle, progress_sig, start):
#     if not particle.histogram.fit(fitparam.numexp, fitparam.tau, fitparam.amp,
#                                   fitparam.shift / channelwidth, fitparam.decaybg,
#                                   fitparam.irfbg,
#                                   start, end, fitparam.addopt,
#                                   fitparam.irf, fitparam.shiftfix):
#         pass  # fit unsuccessful
#     particle.numexp = fitparam.numexp
#     progress_sig.emit()
#     if not particle.has_levels:
#         return
#     for level in particle.levels:
#         try:
#             if not level.histogram.fit(fitparam.numexp, fitparam.tau, fitparam.amp,
#                                        fitparam.shift / channelwidth, fitparam.decaybg,
#                                        fitparam.irfbg,
#                                        start, end, fitparam.addopt,
#                                        fitparam.irf, fitparam.shiftfix):
#                 pass  # fit unsuccessful
#         except AttributeError:
#             print("No decay")


# def get_plot(ui_pg_layout_widget: pg.GraphicsLayoutWidget) -> pg.PlotItem:
#     return ui_pg_layout_widget.addPlot()


# def setup_plot(plot: pg.PlotItem):
#     # plot.setBackground(background=None)
#     # plot_item = plot.getPlotItem()
#
#     axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)
#     plot.getAxis('left').setPen(axis_line_pen)
#     plot.getAxis('bottom').setPen(axis_line_pen)
#     # Set axis label bold and size
#     font = plot.getAxis('left').label.font()
#     font.setBold(True)
#     # if plot == self.pgLifetime_Int:
#     #     font.setPointSize(8)
#     # elif plot == self.pgGroups_Int or plot == self.pgGroups_Hist:
#     #     font.setPointSize(10)
#     # else:
#     #     font.setPointSize(12)
#     plot.getAxis('left').label.setFont(font)
#     plot.getAxis('bottom').label.setFont(font)
#
#     # plot.setAntialiasing(True)
