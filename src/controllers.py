from __future__ import annotations
import os
from typing import Union, List, Tuple, TYPE_CHECKING
from copy import copy
import tempfile
from io import BytesIO

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPen, QColor, QPalette, QFont
from PyQt5.QtWidgets import QWidget, QFrame, QInputDialog, QFileDialog, QTextBrowser, QCheckBox
import time
import pickle
from threading import Lock
# from multiprocessing.synchronize import Lock
from time import sleep

if TYPE_CHECKING:
    from main import MainWindow

from my_logger import setup_logger
from smsh5 import H5dataset, Particle, ParticleAllHists, RasterScan
from tcspcfit import FittingParameters, FittingDialog
from threads import ProcessThread, ProcessTaskResult, WorkerBinAll
from thread_tasks import OpenFile, BinAll, bin_all
import matplotlib.pyplot as plt

EXPORT_WIDTH = 1500
EXPORT_MPL_WIDTH = 10
EXPORT_MPL_HEIGHT = 4.5
EXPORT_MPL_DPI = 300

logger = setup_logger(__name__)


def export_plot_item(plot_item: pg.PlotItem, path: str, text: str = None):
    left_autofill = plot_item.getAxis('left').autoFillBackground()
    bottom_autofill = plot_item.getAxis('bottom').autoFillBackground()
    vb_autofill = plot_item.vb.autoFillBackground()
    # left_label_font = plot_item.getAxis('left').label.font()
    # bottom_label_font = plot_item.getAxis('bottom').label.font()
    # left_axis_font = plot_item.getAxis('left').font()
    # bottom_axis_font = plot_item.getAxis('bottom').font()

    if text is not None:
        f_size = plot_item.height() * 7 / 182
        font = QFont()
        font.setPixelSize(f_size)
        text_item = pg.TextItem(text=text, color='k', anchor=(1, 0))
        text_item.setFont(font)
        plot_item.addItem(text_item, ignoreBounds=True)
        text_item.setPos(plot_item.vb.width(), 0)
        text_item.setParentItem(plot_item.vb)

    plot_item.getAxis('left').setAutoFillBackground(True)
    plot_item.getAxis('bottom').setAutoFillBackground(True)
    plot_item.vb.setAutoFillBackground(True)

    # new_label_point_size = plot_item.height() * 10.0/486.0
    # plot_item.getAxis('left').label.font().setPointSizeF(new_label_point_size)
    # plot_item.getAxis('bottom').label.font().setPointSizeF(new_label_point_size)
    # new_axis_point_size = plot_item.height() * 8.25/486
    # plot_item.getAxis('left').font().setPointSizeF(new_axis_point_size)
    # plot_item.getAxis('bottom').font().setPointSizeF(new_axis_point_size)

    ex = ImageExporter(plot_item.scene())
    ex.parameters()['width'] = EXPORT_WIDTH
    ex.export(path)

    plot_item.getAxis('left').setAutoFillBackground(left_autofill)
    plot_item.getAxis('bottom').setAutoFillBackground(bottom_autofill)
    plot_item.vb.setAutoFillBackground(vb_autofill)
    # plot_item.getAxis('left').label.setFont(left_label_font)
    # plot_item.getAxis('bottom').label.setFont(bottom_label_font)
    # plot_item.getAxis('left').setFont(left_axis_font)
    # plot_item.getAxis('bottom').setFont(bottom_axis_font)


class IntController(QObject):

    def __init__(self, mainwindow: MainWindow,
                 int_widget: pg.PlotWidget,
                 int_hist_container: QWidget,
                 int_hist_line: QFrame,
                 int_hist_widget: pg.PlotWidget,
                 lifetime_widget: pg.PlotWidget,
                 groups_int_widget: pg.PlotWidget,
                 groups_hist_widget: pg.PlotWidget,
                 level_info_container: QWidget,
                 level_info_text: QTextBrowser,
                 int_level_line: QFrame):
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

        self.int_level_info_container = level_info_container
        self.level_info_text = level_info_text
        self.int_level_line = int_level_line
        self.show_level_info = self.mainwindow.chbInt_Show_Level_Info.isChecked()
        self.hide_show_chb(chb_obj=self.mainwindow.chbInt_Show_Level_Info, show=False)
        mw_bg_colour = self.mainwindow.palette().color(QPalette.Background)
        level_info_palette = self.level_info_text.viewport().palette()
        level_info_palette.setColor(QPalette.Base, mw_bg_colour)
        self.level_info_text.viewport().setPalette(level_info_palette)

        self.int_plot.vb.scene().sigMouseClicked.connect(self.any_int_plot_double_click)
        self.groups_int_plot.vb.scene().sigMouseClicked.connect(self.any_int_plot_double_click)
        self.lifetime_plot.vb.scene().sigMouseClicked.connect(self.any_int_plot_double_click)

        self.temp_fig = None
        self.temp_ax = None
        self.temp_bins = None

        # Setup axes and limits
        # self.groups_hist_plot.getAxis('bottom').setLabel('Relative Frequency')

        self.confidence_index = {
            0: 99,
            1: 95,
            2: 90,
            3: 69}

    def setup_plot(self, plot_item: pg.PlotItem,
                   is_int_hist: bool = False,
                   is_group_hist: bool = False,
                   is_lifetime: bool = False):

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
            if not is_lifetime:
                bottom_axis.setLabel('Time', 's')
            plot_item.vb.setLimits(xMin=0, yMin=0)

    @staticmethod
    def setup_widget(plot_widget: pg.PlotWidget):

        # Set widget background and antialiasing
        plot_widget.setBackground(background=None)
        plot_widget.setAntialiasing(True)

    def hide_show_chb(self, chb_obj: QCheckBox, show: bool):
        chb_obj.blockSignals(True)
        chb_obj.setChecked(show)
        chb_obj.blockSignals(False)
        if chb_obj is self.mainwindow.chbInt_Show_Level_Info:
            if show:
                self.int_level_info_container.show()
                self.int_level_line.show()
            else:
                self.int_level_info_container.hide()
                self.int_level_line.hide()
        elif chb_obj is self.mainwindow.chbInt_Show_Hist:
            if show:
                self.int_hist_container.show()
                self.int_hist_line.show()
            else:
                self.int_hist_container.hide()
                self.int_hist_line.hide()

    def hist_chb_changed(self):

        self.show_int_hist = self.mainwindow.chbInt_Show_Hist.isChecked()

        if self.show_int_hist:
            if self.show_level_info:
                self.hide_show_chb(chb_obj=self.mainwindow.chbInt_Show_Level_Info, show=False)
                self.show_level_info = False
            self.int_hist_container.show()
            self.int_hist_line.show()
            self.plot_hist()
        else:
            self.int_hist_container.hide()
            self.int_hist_line.hide()

    def level_info_chb_changed(self):

        self.show_level_info = self.mainwindow.chbInt_Show_Level_Info.isChecked()

        if self.show_level_info:
            if self.show_int_hist:
                self.hide_show_chb(chb_obj=self.mainwindow.chbInt_Show_Hist, show=False)
                self.show_int_hist = False
            self.int_level_info_container.show()
            self.int_level_line.show()
            self.update_level_info()
        else:
            self.int_level_info_container.hide()
            self.int_level_line.hide()

    def gui_apply_bin(self):
        """ Changes the bin size of the data of the current particle and then displays the new trace. """
        try:
            self.mainwindow.current_particle.binints(self.get_bin())
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

        self.start_binall_thread(self.get_bin())

    def start_binall_thread(self, bin_size) -> None:
        """

        Parameters
        ----------
        bin_size
        """

        mw = self.mainwindow
        try:
            dataset = mw.current_dataset
            mw.start_progress(dataset.num_parts)
            mw.status_message("Binning all particles...")
            for part in dataset.particles:
                part.binints(bin_size)
                mw.update_progress()
        except Exception as err:
            mw.status_message("An error has occurred...")
            mw.end_progress()
            logger.error('Error Occured:')
        else:
            mw.display_data()
            mw.repaint()
            mw.status_message("Done")
            mw.end_progress()
            logger.info('All traces binned')

    def binall_thread_complete(self):

        self.mainwindow.status_message('Done')
        self.plot_trace()
        logger.info('Binnig all levels complete')

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

    def plot_trace(self, particle: Particle = None,
                   for_export: bool = False,
                   export_path: str = None,
                   lock: bool = False) -> None:
        """ Used to display the trace from the absolute arrival time data of the current particle. """

        if type(export_path) is bool:
            lock = export_path
            export_path = None
        try:
            # self.currentparticle = self.treemodel.data(self.current_ind, Qt.UserRole)
            if particle is None:
                particle = self.mainwindow.current_particle
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

                unit = 'counts/' + str(self.get_bin()) + 'ms'
                if not for_export:
                    plot_pen.setJoinStyle(Qt.RoundJoin)

                    plot_item.clear()
                    plot_item.getAxis('left').setLabel(text='Intensity', units=unit)
                    plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=times[-1])
                    plot_item.plot(x=times, y=trace, pen=plot_pen, symbol=None)
                else:
                    if self.temp_fig is None:
                        self.temp_fig = plt.figure()
                        self.temp_fig.set_size_inches(EXPORT_MPL_WIDTH, EXPORT_MPL_HEIGHT)
                    else:
                        self.temp_fig.clf()
                    gs = self.temp_fig.add_gridspec(nrows=1, ncols=5, wspace=0, left=0.07,
                                                    right=0.98)
                    int_ax = self.temp_fig.add_subplot(gs[0, :-1])
                    hist_ax = self.temp_fig.add_subplot(gs[0, -1])
                    self.temp_ax = {'int_ax': int_ax, 'hist_ax': hist_ax}
                    hist_ax.tick_params(direction='in', labelleft=False, labelbottom=False)
                    hist_ax.spines['top'].set_visible(False)
                    hist_ax.spines['right'].set_visible(False)
                    int_ax.plot(times, trace)
                    int_ax.set(xlabel='time (s)',
                               ylabel=f'intensity {unit}',
                               xlim=[0, times[-1]],
                               ylim=[0, max(trace)])
                    int_ax.spines['top'].set_visible(False)
                    self.temp_fig.suptitle(f"{particle.name} Intensity Trace")
                    self.plot_hist(particle=particle,
                                   for_export=for_export,
                                   export_path=export_path,
                                   for_levels=False)
        if lock:
            self.mainwindow.lock.release()

    def plot_levels(self, particle: Particle = None,
                    for_export: bool = False,
                    export_path: str = None,
                    lock: bool = False):
        """ Used to plot the resolved intensity levels of the current particle. """
        if type(export_path) is bool:
            lock = export_path
            export_path = None
        if particle is None:
            particle = self.mainwindow.current_particle
        if not particle.has_levels:
            return
        try:
            level_ints, times = particle.levels2data()
            level_ints = level_ints * self.get_bin() / 1E3
        except AttributeError:
            logger.error('No levels!')

        if not for_export:
            plot_pen = QPen()
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

            # plot_pen.brush()
            plot_pen.setJoinStyle(Qt.RoundJoin)
            plot_pen.setCosmetic(True)

            plot_item.plot(x=times, y=level_ints, pen=plot_pen, symbol=None)
        else:
            self.temp_ax['int_ax'].plot(times, level_ints, linewidth=0.7)
            self.temp_fig.suptitle(f"{particle.name} Intensity Trace with Levels")
            self.plot_hist(particle=particle,
                           for_export=True,
                           export_path=export_path,
                           for_levels=True)

        if not for_export and (cur_tab_name == 'tabLifetime' or cur_tab_name == 'tabIntensity'):
            current_level = particle.level_selected
            if current_level is not None:
                if current_level <= particle.num_levels - 1:
                    current_ints, current_times = particle.current2data(current_level)
                else:
                    current_group = current_level - particle.num_levels
                    current_ints, current_times = particle.current_group2data(current_group)
                current_ints = current_ints * self.get_bin() / 1E3

                if not (current_ints[0] == np.inf or current_ints[1] == np.inf):
                    level_plot_pen = QPen()
                    level_plot_pen.setCosmetic(True)
                    level_plot_pen.setJoinStyle(Qt.RoundJoin)
                    level_plot_pen.setColor(QColor('red'))
                    level_plot_pen.setWidthF(3)
                    plot_item.plot(x=current_times, y=current_ints, pen=level_plot_pen, symbol=None)
                else:
                    logger.info('Infinity in level')
        if lock:
            self.mainwindow.lock.release()

    def plot_hist(self, particle: Particle = None,
                  for_export: bool = False,
                  export_path: str = None,
                  for_levels: bool = False,
                  for_groups: bool = False):
        if particle is None:
            particle = self.mainwindow.current_particle
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

            if not for_export:
                plot_item.clear()

                bin_edges = np.histogram_bin_edges(np.negative(int_data), bins=100)
                freq, hist_bins = np.histogram(np.negative(int_data), bins=bin_edges, density=True)
                freq /= np.max(freq)
                int_hist = pg.PlotCurveItem(x=hist_bins, y=freq, pen=plot_pen,
                                            stepMode=True, fillLevel=0, brush=(0, 0, 0, 50))
                int_hist.rotate(-90)
                plot_item.addItem(int_hist)
            elif not (for_levels or for_groups):
                hist_ax = self.temp_ax['hist_ax']
                _, bins, _ = hist_ax.hist(int_data,
                                          bins=50,
                                          orientation='horizontal',
                                          density=True,
                                          edgecolor='k',
                                          range=self.temp_ax['int_ax'].get_ylim(),
                                          label='Trace')
                self.temp_bins = bins
                hist_ax.set_ylim(self.temp_ax['int_ax'].get_ylim())

            if particle.has_levels:
                level_ints = particle.level_ints
                level_ints *= particle.bin_size / 1000
                dwell_times = [level.dwell_time_s for level in
                               particle.levels]
                if not for_export:
                    level_freq, level_hist_bins = np.histogram(np.negative(level_ints),
                                                               bins=bin_edges,
                                                               weights=dwell_times,
                                                               density=True)
                    level_freq /= np.max(level_freq)
                    level_hist = pg.PlotCurveItem(x=level_hist_bins, y=level_freq, stepMode=True,
                                                  pen=plot_pen, fillLevel=0, brush=(0, 0, 0, 255))

                    level_hist.rotate(-90)
                    plot_item.addItem(level_hist)
                elif for_levels and particle.has_levels:
                    hist_ax = self.temp_ax['hist_ax']
                    hist_ax.hist(level_ints,
                                 bins=50,
                                 weights=dwell_times,
                                 orientation='horizontal',
                                 density=True,
                                 # rwidth=0.5,
                                 edgecolor='k',
                                 linewidth=0.5,
                                 alpha=0.4,
                                 range=self.temp_ax['int_ax'].get_ylim(),
                                 label='Resolved')
                elif for_groups and particle.has_groups:
                    group_ints = np.array(particle.groups_ints)
                    group_ints *= particle.bin_size / 1000
                    group_dwell_times = [group.dwell_time_s for group in particle.groups]
                    hist_ax = self.temp_ax['hist_ax']
                    hist_ax.hist(group_ints,
                                 bins=50,
                                 weights=group_dwell_times,
                                 orientation='horizontal',
                                 density=True,
                                 # rwidth=0.3,
                                 # color='k',
                                 fill=False,
                                 hatch='///',
                                 edgecolor='k',
                                 linewidth=0.5,
                                 # alpha=0.4,
                                 range=self.temp_ax['int_ax'].get_ylim(),
                                 label='Grouped')

        if for_export and export_path is not None:
            if not (os.path.exists(export_path) and os.path.isdir(export_path)):
                raise AssertionError("Provided path not valid")
            if not (for_levels or for_groups):
                full_path = os.path.join(export_path, particle.name + ' trace.png')
            elif for_levels:
                full_path = os.path.join(export_path, particle.name + ' trace (levels).png')
            else:
                full_path = os.path.join(export_path,
                                         particle.name + ' trace (levels and groups).png')
            hist_ax.legend(prop={'size': 6}, frameon=False)
            self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
            sleep(1)

    def update_level_info(self, particle: Particle = None):
        if particle is None:
            particle = self.mainwindow.current_particle

        cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()
        if cur_tab_name == 'tabIntensity' and self.show_level_info:
            info = ''
            if particle.level_selected is None:
                info = info + "Whole Trace"
                info = info + f"\n{'*' * len(info)}"
                info = info + f"\nTotal Dwell Time (s) = {particle.dwell_time: .3g}"
                info = info + f"\n# of Photons = {particle.num_photons}"
                if particle.has_levels:
                    info = info + f"\n# of Levels = {particle.num_levels}"
                if particle.has_groups:
                    info = info + f"\n# of Groups = {particle.num_groups}"
                if particle.has_levels:
                    info = info + f"\nHas Photon Bursts = {particle.has_burst}"
            elif particle.has_levels:
                is_group_level = False
                if particle.level_selected <= particle.num_levels - 1:
                    level = particle.levels[particle.level_selected]
                    info = info + f"Level {particle.level_selected + 1}"
                else:
                    level = particle.groups[particle.level_selected - particle.num_levels]
                    is_group_level = True
                    info = info + f"Group {particle.level_selected - particle.num_levels + 1}"
                info = info + f"\n{'*' * len(info)}"
                info = info + f"\nIntensity (counts/s) = {level.int_p_s: .3g}"
                info = info + f"\nDwell Time (s) = {level.dwell_time_s: .3g}"
                info = info + f"\n# of Photons = {level.num_photons}"
            self.level_info_text.setText(info)

    def plot_group_bounds(self, particle: Particle = None,
                          for_export: bool = False,
                          export_path: str = None,
                          lock: bool = False):
        if type(export_path) is bool:
            lock = export_path
            export_path = None
        if particle is None:
            particle = self.mainwindow.current_particle

        if for_export:
            cur_tab_name = 'tabIntensity'
        else:
            cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()

        if cur_tab_name == 'tabIntensity' \
                or cur_tab_name == 'tabGrouping' \
                or cur_tab_name == 'tabLifetime':
            if not particle.has_groups \
                    or particle.ahca.best_step.single_level \
                    or particle.ahca.selected_step.num_groups < 2:
                if lock:
                    self.mainwindow.lock.release()
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
            elif cur_tab_name == 'tabLifetime':
                if self.mainwindow.chbLifetime_Show_Groups.isChecked():
                    int_plot = self.lifetime_plot
                else:
                    return

            int_conv = particle.bin_size / 1000

            if for_export:
                int_ax = self.temp_ax['int_ax']
            for i, bound in enumerate(group_bounds):
                if i % 2:
                    bound = (bound[0] * int_conv, bound[1] * int_conv)
                    if not for_export:
                        int_plot.addItem(
                            pg.LinearRegionItem(values=bound,
                                                orientation='horizontal',
                                                movable=False,
                                                pen=QPen().setWidthF(0)))
                    else:
                        ymin, ymax = bound
                        int_ax.axhspan(ymin=ymin, ymax=ymax, color='k', alpha=0.15, linestyle='')

            if not for_export:
                line_pen = QPen()
                line_pen.setWidthF(1)
                line_pen.setStyle(Qt.DashLine)
                line_pen.brush()
                # plot_pen.setJoinStyle(Qt.RoundJoin)
                line_pen.setColor(QColor(0, 0, 0, 150))
                line_pen.setCosmetic(True)
                line_times = [0, particle.dwell_time]
            for group in groups:
                g_int = group.int_p_s * int_conv
                if not for_export:
                    g_ints = [g_int] * 2
                    int_plot.plot(x=line_times, y=g_ints, pen=line_pen, symbol=None)
                else:
                    int_ax.axhline(g_int, linestyle='--', linewidth=0.5, color='k')

            if for_export and export_path is not None:
                if not (os.path.exists(export_path) and os.path.isdir(export_path)):
                    raise AssertionError("Provided path not valid")
                full_path = os.path.join(export_path,
                                         particle.name + ' trace (levels and groups).png')
                self.temp_fig.suptitle(f"{particle.name} Intensity Trace with Levels and Groups")
                self.plot_hist(particle=particle,
                               for_export=for_export,
                               export_path=export_path,
                               for_groups=True)
                # self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
                # export_plot_item(plot_item=int_plot, path=full_path)
        if lock:
            self.mainwindow.lock.release()

    def plot_all(self):
        self.plot_trace()
        self.plot_levels()
        self.plot_hist()
        self.plot_group_bounds()

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
        currentparticle = mw.current_particle
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

        all_sums = self.mainwindow.current_dataset.all_sums
        r_process_thread = ProcessThread()
        r_process_thread.add_tasks_from_methods(objects=cpt_objs,
                                                method_name='run_cpa',
                                                args=(all_sums, conf, True))

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
        particles = self.mainwindow.current_dataset.particles
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
                target_particle
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

        if self.mainwindow.current_particle.has_levels:  # tree2dataset().cpa_has_run:
            self.mainwindow.tabGrouping.setEnabled(True)
        self.mainwindow.current_dataset.has_levels = True
        if self.mainwindow.treeViewParticles.currentIndex().data(Qt.UserRole) is not None:
            self.mainwindow.display_data()
        self.mainwindow.check_remove_bursts(mode=self.resolve_mode)
        # self.mainwindow.chbEx_Levels.setEnabled(True)
        # self.mainwindow.rdbWith_Levels.setEnabled(True)
        self.mainwindow.set_startpoint()
        self.mainwindow.reset_gui()
        self.mainwindow.status_message("Done")
        logger.info('Resolving levels complete')

        self.results_gathered = False

    def get_gui_confidence(self):
        """ Return current GUI value for confidence percentage. """

        return [self.mainwindow.cmbConfIndex.currentIndex(),
                self.confidence_index[self.mainwindow.cmbConfIndex.currentIndex()]]

    def any_int_plot_double_click(self, event: MouseClickEvent):
        if event.double():
            event.accept()
            cp = self.mainwindow.current_particle
            if cp.has_levels:
                use_groups = False
                if event.currentItem is self.int_plot.vb:
                    use_groups = self.mainwindow.chbInt_Show_Groups.isChecked()
                elif event.currentItem is self.groups_int_plot.vb:
                    use_groups = True
                elif event.currentItem is self.lifetime_plot.vb:
                    use_groups = self.mainwindow.chbLifetime_Show_Groups.isChecked()

                if cp.has_groups and use_groups:
                    clicked_int = event.currentItem.mapSceneToView(event.scenePos()).y()
                    clicked_int = clicked_int * (1000 / self.mainwindow.spbBinSize.value())
                    clicked_group = None
                    group_bounds = cp.groups_bounds
                    group_bounds.reverse()
                    for i, (group_low, group_high) in enumerate(group_bounds):
                        if group_low <= clicked_int <= group_high:
                            clicked_group = i
                            break
                    if clicked_group is not None:
                        cp.level_selected = clicked_group + cp.num_levels
                        self.mainwindow.display_data()
                else:
                    try:
                        clicked_time = event.currentItem.mapSceneToView(event.scenePos()).x()
                    except AttributeError as err:
                        if err.args[0] == '\'AxisItem\' object has no attribute \'mapSceneToView\'':
                            cp.level_selected = None
                        else:
                            logger.error(err)
                            raise err
                    else:
                        level_times = [lvl.times_s for lvl in cp.levels]
                        clicked_level = None
                        for i, (start, end) in enumerate(level_times):
                            if start <= clicked_time <= end:
                                clicked_level = i
                                break
                        if clicked_level is not None:
                            cp.level_selected = clicked_level
                    finally:
                        self.mainwindow.display_data()

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
        self.residual_widget.hide()

        self.setup_plot(self.life_hist_plot)
        self.setup_plot(self.residual_plot, is_residuals=True)

        self.fitparamdialog = FittingDialog(self.mainwindow, self)
        self.fitparam = FittingParameters(self)
        self.irf_loaded = False

        self.first = 0
        self.startpoint = None
        self.tmin = 0

        self.temp_fig = None
        self.temp_ax = None

    def setup_plot(self, plot: pg.PlotItem, is_residuals: bool = False):
        # Set axis label bold and size
        axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)
        plot.getAxis('left').setPen(axis_line_pen)
        plot.getAxis('bottom').setPen(axis_line_pen)
        plot.getAxis('left').label.font().setBold(True)
        plot.getAxis('bottom').label.font().setBold(True)
        plot.getAxis('left').label.font().setPointSize(16)
        plot.getAxis('bottom').label.font().setPointSize(16)

        # Setup axes and limits
        if not is_residuals:
            plot.getAxis('left').setLabel('Num. of occur.', 'counts/bin')
            plot.getAxis('bottom').setLabel('Decay time', 'ns')
            plot.getViewBox().setLimits(xMin=0, yMin=0)
        else:
            plot.getAxis('left').setLabel('Weighted residual')
            plot.getAxis('bottom').setLabel('Time (ns)')
            plot.getViewBox().setLimits(xMin=0)

    @staticmethod
    def setup_widget(plot_widget: pg.PlotWidget):
        # Set widget background and antialiasing
        plot_widget.setBackground(background=None)
        plot_widget.setAntialiasing(True)

    def gui_prev_lev(self):
        """ Moves to the previous resolves level and displays its decay curve. """

        cp = self.mainwindow.current_particle
        changed = False
        if cp.level_selected is not None:
            if cp.level_selected == 0:
                cp.level_selected = None
                changed = True
            else:
                cp.level_selected -= 1
                changed = True

        if changed:
            self.mainwindow.display_data()

    def gui_next_lev(self):
        """ Moves to the next resolves level and displays its decay curve. """

        cp = self.mainwindow.current_particle
        changed = False
        if cp.level_selected is None:
            cp.level_selected = 0
            changed = True
        elif cp.has_groups:
            if cp.level_selected < cp.num_levels + cp.num_groups - 1:
                cp.level_selected += 1
                changed = True
        elif cp.level_selected < cp.num_levels - 1:
            cp.level_selected += 1
            changed = True

        if changed:
            self.mainwindow.display_data()

    def gui_whole_trace(self):
        "Unselects selected level and shows whole trace's decay curve"

        # self.mainwindow.current_level = None
        self.mainwindow.current_particle.level_selected = None
        self.mainwindow.display_data()

    def gui_jump_to_groups(self):
        cp = self.mainwindow.current_particle
        if cp.has_groups:
            cp.level_selected = cp.num_levels
            self.mainwindow.display_data()

    def gui_show_hide_residuals(self):

        show = self.mainwindow.chbShow_Residuals.isChecked()

        if show:
            self.residual_widget.show()
        else:
            self.residual_widget.hide()

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

    def add_irf(self, decay, t, irfdata):

        self.fitparam.irf = decay
        self.fitparam.irft = t
        # self.fitparam.irfdata = irfdata
        self.irf_loaded = True
        self.mainwindow.set_startpoint(irf_data=irfdata)
        self.mainwindow.reset_gui
        self.mainwindow.dataset_node.dataobj.irf = decay
        self.mainwindow.dataset_node.dataobj.irf_t = t
        self.mainwindow.dataset_node.dataobj.has_irf = True
        self.fitparamdialog.updateplot()

    def gui_fit_param(self):
        """ Opens a dialog to choose the setting with which the decay curve will be fitted. """

        if self.fitparamdialog.exec():
            self.fitparam.getfromdialog()

    def gui_fit_current(self):
        """ Fits the currently selected level's decay curve using the provided settings. """

        cp = self.mainwindow.current_particle
        selected_level = cp.level_selected
        if selected_level is None:
            histogram = cp.histogram
        else:
            # level = self.mainwindow.current_level
            if selected_level <= cp.num_levels - 1:
                histogram = cp.levels[selected_level].histogram
            else:
                selected_group = selected_level - cp.num_levels
                histogram = cp.groups[selected_group].histogram
        try:
            channelwidth = self.mainwindow.current_particle.channelwidth
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
            else:
                cp.has_fit_a_lifetime = True
        except AttributeError:
            logger.error("No decay")
        else:
            # self.mainwindow.display_data()
            self.fitting_thread_complete('current')

    def gui_fit_selected(self):
        """ Fits the all the levels decay curves in the all the selected particles using the provided settings. """

        self.start_fitting_thread(mode='selected')

    def gui_fit_all(self):
        """ Fits the all the levels decay curves in the all the particles using the provided settings. """

        self.start_fitting_thread(mode='all')

    def gui_fit_levels(self):
        """ Fits the all the levels decay curves for the current particle. """

        self.start_fitting_thread()

    def plot_all(self):
        self.mainwindow.display_data()

    def update_results(self, select_ind: int = None, particle: Particle = None,
                       for_export: bool = False, str_return: bool = False) -> Union[str, None]:

        if select_ind is None:
            select_ind = self.mainwindow.current_particle.level_selected
        elif select_ind <= -1:
            select_ind = None
        if particle is None:
            particle = self.mainwindow.current_particle
        is_group = False
        is_level = False

        fit_name = f"{particle.name}"
        if select_ind is None:
            histogram = particle.histogram
            fit_name = fit_name + ", Whole Trace"
        elif select_ind <= particle.num_levels - 1:
            histogram = particle.levels[select_ind].histogram
            fit_name = fit_name + f", Level #{select_ind + 1}"
            is_level = True
        else:
            group_ind = select_ind - particle.num_levels
            histogram = particle.groups[group_ind].histogram
            is_group = True
            fit_name = fit_name + f", Group #{group_ind + 1}"
        if not histogram.fitted:
            self.mainwindow.textBrowser.setText('')
            return

        info = ''
        if not for_export:
            info = fit_name + f"\n{len(fit_name) * '*'}\n"

        tau = histogram.tau
        amp = histogram.amp
        try:
            info = info + 'Tau = ' + ' '.join('{:#.3g} ns'.format(F) for F in tau)
            info = info + '\nAmp = ' + ' '.join('{:#.3g} '.format(F) for F in amp)
        except TypeError:  # only one component
            info = info + 'Tau = {:#.3g} ns'.format(tau)
            info = info + '\nAmp = {:#.3g}'.format(amp)

        info = info + f'\n\nShift = {histogram.shift: .3g} ns'
        if not for_export:
            info = info + f'\nDecay BG = {histogram.bg: .3g}'
            info = info + f'\nIRF BG = {histogram.irfbg: .3g}'
        info = info + f'\nChi-Sq = {histogram.chisq: .3g}'
        if not for_export:
            info = info + f'\n(0.8 <- 1 -> 1.3)'

        if is_group:
            group = particle.groups[group_ind]
            info = info + f'\n\nTotal Dwell Time (s) = {group.dwell_time_s: .3g}'
            info = info + f'\n# of photons = {group.num_photons}'
        elif is_level:
            level = particle.levels[select_ind]
            info = info + f'\n\nDwell Time (s) {level.dwell_time_s: .3g}'
            info = info + f'\n# of photons = {level.num_photons}'
        else:
            info = info + f'\n\nDwell Times (s) = {particle.dwell_time: .3g}'
            info = info + f'\n# of photons = {particle.num_photons}'

        if not for_export:
            self.mainwindow.textBrowser.setText(info)

        if str_return:
            return info

    def plot_decay_and_convd(self, particle: Particle,
                             export_path: str,
                             has_groups: bool,
                             only_groups: bool = False,
                             lock: bool = False):
        if not only_groups:
            for i in range(particle.num_levels):
                self.plot_decay(select_ind=i,
                                particle=particle,
                                remove_empty=False,
                                for_export=True,
                                export_path=None)
                self.plot_convd(select_ind=i,
                                particle=particle,
                                remove_empty=False,
                                for_export=True,
                                export_path=export_path)
        if has_groups:
            for i in range(particle.num_groups):
                i_g = i + particle.num_levels
                self.plot_decay(select_ind=i_g,
                                particle=particle,
                                remove_empty=False,
                                for_export=True,
                                export_path=None)
                self.plot_convd(select_ind=i_g,
                                particle=particle,
                                remove_empty=False,
                                for_export=True,
                                export_path=export_path)
        if lock:
            self.mainwindow.lock.release()

    def plot_decay_convd_and_hist(self, particle: Particle,
                                  export_path: str,
                                  has_groups: bool,
                                  only_groups: bool = False,
                                  lock: bool = False):
        if not only_groups:
            for i in range(particle.num_levels):
                self.plot_decay(select_ind=i,
                                particle=particle,
                                remove_empty=False,
                                for_export=True,
                                export_path=None)
                self.plot_convd(select_ind=i,
                                particle=particle,
                                remove_empty=False,
                                for_export=True,
                                export_path=None)
                self.plot_residuals(select_ind=i,
                                    particle=particle,
                                    for_export=True,
                                    export_path=export_path)
        if has_groups:
            for i in range(particle.num_groups):
                i_g = i + particle.num_levels
                self.plot_decay(select_ind=i_g,
                                particle=particle,
                                remove_empty=False,
                                for_export=True,
                                export_path=None)
                self.plot_convd(select_ind=i_g,
                                particle=particle,
                                remove_empty=False,
                                for_export=True,
                                export_path=None)
                self.plot_residuals(select_ind=i_g,
                                    particle=particle,
                                    for_export=True,
                                    export_path=export_path)
        if lock:
            self.mainwindow.lock.release()

    def plot_decay(self, select_ind: int = None,
                   particle: Particle = None,
                   remove_empty: bool = False,
                   for_export: bool = False,
                   export_path: str = None,
                   lock: bool = False) -> None:
        """ Used to display the histogram of the decay data of the current particle. """

        if type(export_path) is bool:
            lock = export_path
            export_path = None
        if select_ind is None:
            select_ind = self.mainwindow.current_particle.level_selected
        elif select_ind <= -1:
            select_ind = None
        # print(currentlevel)
        if particle is None:
            particle = self.mainwindow.current_particle

        group_ind = None
        if select_ind is None:
            if particle.histogram.fitted:
                decay = particle.histogram.fit_decay
                t = particle.histogram.convd_t
            else:
                try:
                    decay = particle.histogram.decay
                    t = particle.histogram.t
                except AttributeError:
                    logger.error('No Decay!')
                    return
        elif select_ind < particle.num_levels:
            if particle.cpts.levels[select_ind].histogram.fitted:
                decay = particle.levels[select_ind].histogram.fit_decay
                t = particle.cpts.levels[select_ind].histogram.convd_t
            else:
                try:
                    decay = particle.cpts.levels[select_ind].histogram.decay
                    t = particle.cpts.levels[select_ind].histogram.t
                except ValueError:
                    return
        else:
            group_ind = select_ind - particle.num_levels
            if particle.groups[group_ind].histogram.fitted:
                decay = particle.groups[group_ind].histogram.fit_decay
                t = particle.groups[group_ind].histogram.convd_t
            else:
                try:
                    decay = particle.groups[group_ind].histogram.decay
                    t = particle.groups[group_ind].histogram.t
                except ValueError:
                    return

        try:
            decay.size
        except AttributeError as e:
            print(e)
        if decay.size == 0:
            return  # some levels have no photons

        cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()
        if cur_tab_name == 'tabLifetime' or for_export:

            if remove_empty:
                self.first = (decay > 4).argmax(axis=0)
                t = t[self.first:-1] - t[self.first]
                decay = decay[self.first:-1]
            else:
                self.first = 0
            unit = f'ns with {particle.channelwidth: .3g} ns bins'
            max_t = particle.histogram.t[-1]

            if not for_export:
                life_hist_plot = self.life_hist_plot
                life_hist_plot.clear()
                plot_pen = QPen()
                plot_pen.setWidthF(2)
                plot_pen.setJoinStyle(Qt.RoundJoin)
                plot_pen.setColor(QColor('blue'))
                plot_pen.setCosmetic(True)
                try:
                    life_hist_plot.plot(x=t, y=decay, pen=plot_pen, symbol=None)
                except Exception as e:
                    logger.error(e)
                    shortest = min([len(t), len(decay)])
                    t = t[:shortest]
                    decay = decay[:shortest]
                    life_hist_plot.plot(x=t, y=decay, pen=plot_pen, symbol=None)

                life_hist_plot.getAxis('bottom').setLabel('Decay time', unit)
                life_hist_plot.getViewBox().setLimits(xMin=0, yMin=0, xMax=max_t)
                life_hist_plot.getViewBox().setRange(xRange=[0, max_t])
                self.fitparamdialog.updateplot()
            else:
                if self.temp_fig is None:
                    self.temp_fig = plt.figure()
                else:
                    self.temp_fig.clf()
                if self.mainwindow.rdbAnd_Residuals.isChecked():
                    self.temp_fig.set_size_inches(EXPORT_MPL_WIDTH, 1.5 * EXPORT_MPL_HEIGHT)
                    gs = self.temp_fig.add_gridspec(5, 1, hspace=0, left=0.1, right=0.95)
                    decay_ax = self.temp_fig.add_subplot(gs[0:-1, 0])
                    decay_ax.tick_params(direction='in', labelbottom=False)
                    residual_ax = self.temp_fig.add_subplot(gs[-1, 0])
                    residual_ax.spines['right'].set_visible(False)
                    self.temp_ax = {'decay_ax': decay_ax, 'residual_ax': residual_ax}
                else:
                    self.temp_fig.set_size_inches(EXPORT_MPL_WIDTH, EXPORT_MPL_HEIGHT)
                    gs = self.temp_fig.add_gridspec(1, 1, left=0.05, right=0.95)
                    decay_ax = self.temp_fig.add_subplot(gs[0, 0])
                    self.temp_ax = {'decay_ax': decay_ax}

                decay_ax.spines['top'].set_visible(False)
                decay_ax.spines['right'].set_visible(False)
                try:
                    decay_ax.semilogy(t, decay)
                except Exception as e:
                    shortest = min([len(t), len(decay)])
                    t = t[:shortest]
                    decay = decay[:shortest]
                    decay_ax.semilogy(t, decay)

                min_pos_decay = decay[np.where(decay > 0, decay, np.inf).argmin()]
                max_decay = max(decay)
                if min_pos_decay == max(decay):
                    max_decay = min_pos_decay * 2
                decay_ax.set(xlabel=f'decay time ({unit})',
                             ylabel='counts',
                             xlim=[t[0], max_t],
                             ylim=[min_pos_decay, max_decay])

            if for_export and export_path is not None:
                if not (os.path.exists(export_path) and os.path.isdir(export_path)):
                    raise AssertionError("Provided path not valid")
                if select_ind is None:
                    type_str = ' hist (whole trace).png'
                    title_str = f"{particle.name} Decay Trace"
                elif group_ind is None:
                    type_str = f' hist (level {select_ind + 1}).png'
                    title_str = f"{particle.name}, Level {select_ind + 1} Decay Trace"
                else:
                    type_str = f' hist (group {group_ind + 1}).png'
                    title_str = f"{particle.name}, Group {group_ind + 1} Decay Trace"
                self.temp_fig.suptitle(title_str)
                full_path = os.path.join(export_path, particle.name + type_str)
                self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
                sleep(1)
                # export_plot_item(plot_item=life_hist_plot, path=full_path)
        if lock:
            self.mainwindow.lock.release()

    def plot_convd(self, select_ind: int = None,
                   particle: Particle = None,
                   remove_empty: bool = False,
                   for_export: bool = False,
                   export_path: str = None,
                   lock: bool = False) -> None:
        """ Used to display the histogram of the decay data of the current particle. """

        if type(export_path) is bool:
            lock = export_path
            export_path = None
        if select_ind is None:
            select_ind = self.mainwindow.current_particle.level_selected
        elif select_ind <= -1:
            select_ind = None
        if particle is None:
            particle = self.mainwindow.current_particle

        group_ind = None
        if select_ind is None:
            try:
                convd = particle.histogram.convd
                t = particle.histogram.convd_t

            except AttributeError:
                logger.error('No Decay!')
                return
        elif select_ind <= particle.num_levels - 1:
            try:
                convd = particle.levels[select_ind].histogram.convd
                t = particle.levels[select_ind].histogram.convd_t
            except ValueError:
                return
        else:
            try:
                group_ind = select_ind - particle.num_levels
                convd = particle.groups[group_ind].histogram.convd
                t = particle.groups[group_ind].histogram.convd_t
            except ValueError:
                return

        if convd is None or t is None:
            return

        # convd = convd / convd.max()

        cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()
        if cur_tab_name == 'tabLifetime' or for_export:
            if not for_export:
                plot_pen = QPen()
                plot_pen.setWidthF(1)
                plot_pen.setJoinStyle(Qt.RoundJoin)
                plot_pen.setColor(QColor('red'))
                plot_pen.setCosmetic(True)

                self.life_hist_plot.plot(x=t, y=convd, pen=plot_pen, symbol=None)
                unit = f'ns with {particle.channelwidth: .3g} ns bins'
                self.life_hist_plot.getAxis('bottom').setLabel('Decay time', unit)
                self.life_hist_plot.getViewBox().setXRange(min=t[0], max=t[-1], padding=0)
                self.life_hist_plot.getViewBox().setLimits(xMin=0, yMin=0, xMax=t[-1])
            else:
                decay_ax = self.temp_ax['decay_ax']
                decay_ax.semilogy(t, convd)
                _, max_y = decay_ax.get_ylim()
                min_y = min(convd)
                if min_y <= 0:
                    min_y = 1E-1
                if not min_y < max_y:
                    max_y = min_y + 10
                decay_ax.set_ylim(min_y, max_y)

            if for_export and export_path is not None:
                # plot_item = self.life_hist_plot
                if select_ind is None:
                    type_str = f'{particle.name} hist-fitted (whole trace).png'
                    title_str = f'{particle.name} Decay Trace and Fit'
                elif group_ind is None:
                    type_str = f'{particle.name} hist-fitted (level {select_ind + 1}).png'
                    title_str = f'{particle.name}, Level {select_ind + 1} Decay Trace and Fit'
                else:
                    type_str = f'{particle.name} hist-fitted (group {group_ind + 1}).png'
                    title_str = f'{particle.name}, Group {group_ind + 1} Decay Trace and Fit'
                full_path = os.path.join(export_path, type_str)
                text_select_ind = select_ind
                if text_select_ind is None:
                    text_select_ind = -1
                text_str = self.update_results(select_ind=text_select_ind, particle=particle,
                                               for_export=True, str_return=True)
                decay_ax.text(0.8, 0.9, text_str, fontsize=6, transform=decay_ax.transAxes)
                self.temp_fig.suptitle(title_str)
                if EXPORT_MPL_DPI > 50:
                    export_dpi = 50
                else:
                    export_dpi = EXPORT_MPL_DPI
                self.temp_fig.savefig(full_path, dpi=export_dpi)
                sleep(1)

                # export_plot_item(plot_item=plot_item, path=full_path, text=text_str)
        if lock:
            self.mainwindow.lock.release()

    def plot_residuals(self, select_ind: int = None,
                       particle: Particle = None,
                       for_export: bool = False,
                       export_path: str = None,
                       lock: bool = False) -> None:
        """ Used to display the histogram of the decay data of the current particle. """

        if type(export_path) is bool:
            lock = export_path
            export_path = None
        if select_ind is None:
            select_ind = self.mainwindow.current_particle.level_selected
        elif select_ind <= -1:
            select_ind = None
        if particle is None:
            particle = self.mainwindow.current_particle

        group_ind = None
        if select_ind is None:
            try:
                residuals = particle.histogram.residuals
                t = particle.histogram.convd_t
            except AttributeError:
                logger.error('No Decay!')
                return
        elif select_ind <= particle.num_levels - 1:
            try:
                residuals = particle.levels[select_ind].histogram.residuals
                t = particle.levels[select_ind].histogram.convd_t
            except ValueError:
                return
        else:
            try:
                group_ind = select_ind - particle.num_levels
                residuals = particle.groups[group_ind].histogram.residuals
                t = particle.groups[group_ind].histogram.convd_t
            except ValueError:
                return

        cur_tab_name = self.mainwindow.tabWidget.currentWidget().objectName()
        if cur_tab_name == 'tabLifetime' or for_export:

            unit = f'ns with {particle.channelwidth: .3g} ns bins'
            if not for_export:
                if residuals is None or t is None:
                    self.residual_plot.clear()
                    return
                self.residual_plot.clear()
                scat_plot = pg.ScatterPlotItem(x=t, y=residuals, symbol='o', size=3, pen="#0000CC",
                                               brush="#0000CC")
                self.residual_plot.addItem(scat_plot)
                self.residual_plot.getAxis('bottom').setLabel('Decay time', unit)
                self.residual_plot.getViewBox().setXRange(min=t[0], max=t[-1], padding=0)
                self.residual_plot.getViewBox().setYRange(min=residuals.min(),
                                                          max=residuals.max(), padding=0)
                self.residual_plot.getViewBox().setLimits(xMin=0, xMax=t[-1])
            else:
                residual_ax = self.temp_ax['residual_ax']
                residual_ax.scatter(t, residuals, s=1)
                min_x, max_x = self.temp_ax['decay_ax'].get_xlim()
                residual_ax.set(xlim=[min_x, max_x],
                                xlabel=f'decay time ({unit})')

            if for_export and export_path is not None:
                if select_ind is None:
                    type_str = ' residuals (whole trace).png'
                    title_str = f'{particle.name} Decay Trace, Fit and Residuals'
                elif group_ind is None:
                    type_str = f' residuals (level {select_ind + 1} with residuals).png'
                    title_str = f'{particle.name},' \
                                f' Level {select_ind + 1} Decay Trace, Fit and Residuals'
                else:
                    type_str = f' residuals (group {group_ind + 1} with residuals).png'
                    title_str = f'{particle.name}, Group {group_ind + 1}' \
                                f' Decay Trace, Fit and Residuals'
                text_select_ind = select_ind
                if text_select_ind is None:
                    text_select_ind = -1
                text_str = self.update_results(select_ind=text_select_ind, particle=particle,
                                               for_export=True, str_return=True)
                decay_ax = self.temp_ax['decay_ax']
                decay_ax.text(0.9, 0.9, text_str, fontsize=6, transform=decay_ax.transAxes)
                full_path = os.path.join(export_path, particle.name + type_str)
                self.temp_fig.suptitle(title_str)
                self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
                sleep(1)
        if lock:
            self.mainwindow.lock.release()

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
            particles = [mw.current_particle]
        elif mode == 'selected':
            status_message = "Fitting Levels for Selected Particles..."
            particles = mw.get_checked_particles()
        elif mode == 'all':
            status_message = "Fitting Levels for All Particles..."
            particles = mw.current_dataset.particles

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
        particles = self.mainwindow.current_dataset.particles
        part_uuids = [part.uuid for part in particles]
        if type(results) is not list:
            results = [results]
        result_part_uuids = [result.new_task_obj.part_uuid for result in results]
        try:
            for num, result in enumerate(results):
                any_successful_fit = None
                result_part_ind = part_uuids.index(result_part_uuids[num])
                target_particle = self.mainwindow.tree2particle(result_part_ind)

                target_hist = target_particle.histogram
                target_microtimes = target_hist.microtimes

                result.new_task_obj.part_hist._particle = target_particle
                result.new_task_obj.part_hist.microtimes = target_microtimes

                target_particle.histogram = result.new_task_obj.part_hist
                any_successful_fit = [result.new_task_obj.part_hist.fitted]

                for i, res_hist in enumerate(result.new_task_obj.level_hists):
                    target_level = target_particle.cpts.levels[i]
                    target_level_microtimes = target_level.microtimes

                    res_hist._particle = target_particle
                    res_hist.microtimes = target_level_microtimes
                    res_hist.level = target_level

                    target_level.histogram = res_hist
                    any_successful_fit.append(res_hist.fitted)

                for i, res_group_hist in enumerate(result.new_task_obj.group_hists):
                    target_group_lvls_inds = target_particle.groups[i].lvls_inds
                    target_g_lvls_microtimes = np.array([])
                    for lvls_ind in target_group_lvls_inds:
                        m_times = target_particle.cpts.levels[lvls_ind].microtimes
                        target_g_lvls_microtimes = np.append(target_g_lvls_microtimes, m_times)

                    res_group_hist._particle = target_particle
                    res_group_hist.microtimes = target_g_lvls_microtimes
                    res_group_hist.level = target_group_lvls_inds

                    target_particle.groups[i].histogram = res_group_hist
                    any_successful_fit.append(res_group_hist.fitted)

                if any(any_successful_fit):
                    target_particle.has_fit_a_lifetime = True

        except ValueError as e:
            logger.error(e)

    def fitting_thread_complete(self, mode: str = None):
        if self.mainwindow.current_particle is not None:
            self.mainwindow.display_data()
        # self.mainwindow.chbEx_Lifetimes.setEnabled(False)
        # self.mainwindow.chbEx_Lifetimes.setEnabled(True)
        # self.mainwindow.chbEx_Hist.setEnabled(True)
        # self.mainwindow.rdbWith_Fit.setEnabled(True)
        # self.mainwindow.rdbAnd_Residuals.setEnabled(True)
        self.mainwindow.chbShow_Residuals.setChecked(True)
        if not mode == 'current':
            self.mainwindow.status_message("Done")
        self.mainwindow.current_dataset.has_lifetimes = True
        logger.info('Fitting levels complete')

    def change_irf_start(self, start, irf_data):
        # dataset = self.fitparam.irfdata
        dataset = irf_data

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

    def show_residuals_widget(self, show: bool = True, lock: bool = None):
        if show:
            self.residual_widget.show()
        else:
            self.residual_widget.hide()
        self.mainwindow.chbShow_Residuals.setChecked(show)
        if lock:
            self.mainwindow.lock.release()

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

        self.temp_dir = None
        self.temp_fig = None
        self.temp_ax = None

    def clear_bic(self):
        self.bic_scatter_plot.clear()

    def solution_clicked(self, plot, points):
        curr_part = self.mainwindow.current_particle
        last_solution = self.all_last_solutions[curr_part.dataset_ind]
        if last_solution != points[0]:
            curr_part = self.mainwindow.current_particle
            point_num_groups = int(points[0].pos()[0])
            new_ind = curr_part.ahca.steps_num_groups.index(point_num_groups)
            curr_part.ahca.set_selected_step(new_ind)
            curr_part.using_group_levels = False
            curr_part.level_selected = None
            if last_solution:
                last_solution.setPen(pg.mkPen(width=1, color='k'))
            for p in points:
                p.setPen('r', width=2)
            last_solution = points[0]
            self.all_last_solutions[curr_part.dataset_ind] = last_solution

            if curr_part.using_group_levels:
                curr_part.using_group_levels = False
            self.mainwindow.display_data()

    def plot_group_bic(self, particle: Particle = None,
                       for_export: bool = False,
                       export_path: str = None,
                       lock: bool = False):

        if type(export_path) is bool:
            lock = export_path
            export_path = None

        if particle is None:
            particle = self.mainwindow.current_particle
        if particle.ahca.best_step.single_level:
            self.bic_plot_widget.getPlotItem().clear()
            return
        try:
            grouping_bics = particle.grouping_bics.copy()
            grouping_selected_ind = particle.grouping_selected_ind
            best_grouping_ind = particle.best_grouping_ind
            grouping_num_groups = particle.grouping_num_groups.copy()
        except AttributeError:
            logger.error('No groups!')

        cur_tab_name = 'tabGrouping'
        if cur_tab_name == 'tabGrouping':  # or for_export:
            if self.all_bic_plots is None and self.all_last_solutions is None:
                num_parts = self.mainwindow.tree2dataset().num_parts
                self.all_bic_plots = [None] * num_parts
                self.all_last_solutions = [None] * num_parts

            scat_plot_item = self.all_bic_plots[particle.dataset_ind]
            if scat_plot_item is None:
                spot_other_pen = pg.mkPen(width=1, color='k')
                spot_selected_pen = pg.mkPen(width=2, color='r')
                spot_other_brush = pg.mkBrush(color='k')
                spot_best_brush = pg.mkBrush(color='g')

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

                self.all_bic_plots[particle.dataset_ind] = scat_plot_item
                best_solution = scat_plot_item.points()[best_grouping_ind]
                self.all_last_solutions[particle.dataset_ind] = best_solution
                scat_plot_item.sigClicked.connect(self.solution_clicked)

            if not for_export:
                self.bic_scatter_plot.clear()
                # self.bic_plot_widget.getPlotItem().clear()
                self.bic_scatter_plot.addItem(scat_plot_item)
            elif particle.has_groups:
                # grouping_bics = particle.grouping_bics.copy()
                # grouping_selected_ind = particle.grouping_selected_ind
                # best_grouping_ind = particle.best_grouping_ind
                # grouping_num_groups = particle.grouping_num_groups.copy()
                if self.temp_fig is None:
                    self.temp_fig, self.temp_ax = plt.subplots()
                else:
                    # self.temp_fig.clear()
                    self.temp_ax.cla()
                self.temp_fig.set_size_inches(EXPORT_MPL_WIDTH, EXPORT_MPL_HEIGHT)
                self.temp_ax.set(xlabel="Solution's Number of Groups",
                                 ylabel='BIC')
                # self.temp_ax.tick_params(labeltop=False, labelright=False)
                self.temp_ax.spines['right'].set_visible(False)
                self.temp_ax.spines['top'].set_visible(False)
                self.temp_ax.tick_params(axis='x', top=False)
                self.temp_ax.tick_params(axis='y', right=False)

                norm_points = [[grouping_num_groups[i], bic] for i, bic in enumerate(grouping_bics)\
                               if (i != best_grouping_ind
                                   or i != grouping_selected_ind
                                   or grouping_num_groups[i] == 1)]
                norm_points = np.array(norm_points)

                marker_size = 70
                marker_line_width = 2
                norm_color = 'lightgrey'
                norm_line_color = 'k'
                best_color = 'lightgreen'
                selected_line_color = 'r'

                self.temp_ax.scatter(norm_points[:, 0], norm_points[:, 1],
                                     s=marker_size,
                                     color=norm_color,
                                     linewidths=marker_line_width,
                                     edgecolors=norm_line_color,
                                     label='Solutions')

                if grouping_selected_ind == best_grouping_ind:
                    num_groups = grouping_num_groups[best_grouping_ind]
                    bic_value = grouping_bics[best_grouping_ind]
                    self.temp_ax.scatter(num_groups, bic_value,
                                         s=marker_size,
                                         color=best_color,
                                         linewidths=marker_line_width,
                                         edgecolors=selected_line_color,
                                         label='Best Solution Selected')
                else:
                    selected_num_groups = grouping_num_groups[grouping_selected_ind]
                    selected_bic_value = grouping_bics[grouping_selected_ind]
                    self.temp_ax.scatter(selected_num_groups, selected_bic_value,
                                         s=marker_size,
                                         color=norm_color,
                                         linewidths=marker_line_width,
                                         edgecolors=selected_line_color,
                                         label='Solution Selected')

                    best_num_groups = grouping_num_groups[best_grouping_ind]
                    best_bic_value = grouping_bics[best_grouping_ind]
                    self.temp_ax.scatter(best_num_groups, best_bic_value,
                                         s=marker_size,
                                         color=best_color,
                                         linewidths=marker_line_width,
                                         edgecolors=norm_line_color,
                                         label='Best Solution')

                self.temp_ax.set_title(f'{particle.name} Grouping Steps')
                self.temp_ax.legend(frameon=False, loc='lower right')

            if for_export and export_path is not None:
                if not (os.path.exists(export_path) and os.path.isdir(export_path)):
                    raise AssertionError("Provided path not valid")
                full_path = os.path.join(export_path, particle.name + ' BIC.png')
                self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
                sleep(1)
                # export_plot_item(plot_item=self.bic_scatter_plot, path=full_path)
        if lock:
            self.mainwindow.lock.release()

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
            grouping_objs = [mw.current_particle.ahca]
            status_message = "Grouping levels for current particle..."
        elif mode == 'selected':
            checked_particles = mw.get_checked_particles()
            grouping_objs = [particle.ahca for particle in checked_particles]
            status_message = "Grouping levels for selected particle..."
        elif mode == 'all':
            all_particles = mw.current_dataset.particles
            grouping_objs = [particle.ahca for particle in all_particles]
            status_message = "Grouping levels for all particle..."

        # g_process_thread = ProcessThread(num_processes=1, task_buffer_size=1)

        self.temp_dir = tempfile.TemporaryDirectory(prefix="Full_SMS_Grouping")
        g_process_thread = ProcessThread(temp_dir=self.temp_dir)
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
        particles = self.mainwindow.current_dataset.particles
        part_uuids = [part.uuid for part in particles]
        if type(results) is not list:
            results = [results]
        result_part_uuids = [result.new_task_obj.uuid for result in results]
        try:
            for num, result in enumerate(results):
                result_part_ind = part_uuids.index(result_part_uuids[num])
                new_part = self.mainwindow.current_dataset.particles[result_part_ind]
                new_part.level_selected = None

                result_ahca = result.new_task_obj
                result_ahca._particle = new_part
                result_ahca.best_step._particle = new_part
                for step in result_ahca.steps:
                    step._particle = new_part
                    for group_attr_name in ['_ahc_groups', 'groups', '_seed_groups']:
                        if hasattr(step, group_attr_name):
                            group_attr = getattr(step, group_attr_name)
                            if group_attr is not None:
                                for group in group_attr:
                                    for ahc_lvl in group.lvls:
                                        ahc_hist = ahc_lvl.histogram
                                        if hasattr(ahc_hist, '_particle'):
                                            ahc_hist._particle = new_part

                new_part.ahca = result_ahca
                if new_part.has_groups:
                    new_part.makegrouphists()
                    new_part.makegrouplevelhists()
                    # new_part.using_group_levels = True
                    # new_part.makelevelhists()
                    # new_part.using_group_levels = False

            # self.results_gathered = True
        except ValueError as e:
            logger.error(e)

    def grouping_thread_complete(self, mode):
        results = list()
        for result_file in os.listdir(self.temp_dir.name):
            with open(os.path.join(self.temp_dir.name, result_file), 'rb') as f:
                results.append(pickle.load(f))
        self.temp_dir.cleanup()
        self.temp_dir = None
        self.gather_replace_results(results=results)
        self.mainwindow.current_dataset.has_groups = True
        if self.mainwindow.current_particle is not None:
            self.mainwindow.display_data()
        self.mainwindow.status_message("Done")
        self.mainwindow.reset_gui()
        logger.info('Grouping levels complete')

    def apply_groups(self, mode: str = 'current'):
        if mode == 'current':
            particles = [self.mainwindow.current_particle]
        elif mode == 'selected':
            particles = self.mainwindow.get_checked_particles()
        else:
            particles = self.mainwindow.current_dataset.particles

        bool_use = not all([part.using_group_levels for part in particles])
        for particle in particles:
            particle.using_group_levels = bool_use
            particle.level_selected = None

        self.mainwindow.int_controller.plot_all()

    def error(self, e: Exception):
        logger.error(e)
        print(e)


class SpectraController(QObject):

    def __init__(self, mainwindow: MainWindow, spectra_image_view: pg.ImageView):
        super().__init__()

        self.mainwindow = mainwindow
        self.spectra_image_view = spectra_image_view
        self.spectra_image_view.setPredefinedGradient('plasma')
        self.spectra_image_view.view.getAxis('left').setLabel("Wavelength (nm)")
        self.spectra_image_view.view.getAxis('bottom').setLabel("Time (s)")

        self.temp_fig = None
        self.temp_ax = None

        # self.spectra_imv = self.spectra_widget
        # self.spectra_widget.view = pg.PlotItem()

        # self.spectra_imv = self.spectra_widget.addItem(pg.ImageItem())
        # self.spectra_imv = pg.ImageView(view=self.spectra_widget.plotItem, parent=self.spectra_widget)
        # self.spectra_imv.show()

        # blue, red = Color('blue'), Color('red')
        # colours = blue.range_to(red, 256)
        # c_array = np.array([np.array(colour.get_rgb())*255 for colour in colours])
        # self._look_up_table = c_array.astype(np.uint8)

        # self.spectra_plot = spectra_widget.addPlot()
        # self.spectra_plot_item = self.spectra_widget.plotItem
        # self.spectra_image_item = pg.ImageItem()
        # self.spectra_widget.addItem(self.spectra_image_item)
        #
        # axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)
        # self.spectra_plot_item.getAxis('left').setPen(axis_line_pen)
        # self.spectra_plot_item.getAxis('bottom').setPen(axis_line_pen)
        #
        # # Set axis label bold and size
        # font = self.spectra_plot_item.getAxis('left').label.font()
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

    def plot_spectra(self, particle: Particle = None,
                     for_export: bool = False,
                     export_path: str = None,
                     lock: bool = False):
        if type(export_path) is bool:
            lock = export_path
            export_path = None
        if particle is None:
            particle = self.mainwindow.current_particle
        # spectra_data = np.flip(particle.spectra.data[:])
        spectra_data = particle.spectra.data[:]
        # spectra_data = spectra_data.transpose()
        if not for_export:
            data_shape = spectra_data.shape
            current_ratio = data_shape[1] / data_shape[0]
            self.spectra_image_view.getView().setAspectLocked(False, current_ratio)

            self.spectra_image_view.setImage(spectra_data)

            wl = particle.spectra.wavelengths
            y_ticks_wavelength = np.linspace(wl.max(), wl.min(), 15)
            y_ticks_pixel = np.linspace(0, 512, 15)
            y_ticks = [[(y_ticks_pixel[i], f"{y_ticks_wavelength[i]: .1f}") for i in range(15)]]
            self.spectra_image_view.view.getAxis('left').setTicks(y_ticks)

            t_series = particle.spectra.series_times
            mod_selector = len(t_series) // 30 + 1
            x_ticks_t = list()
            for i in range(len(t_series)):
                if not (mod_selector + i) % mod_selector:
                    x_ticks_t.append(t_series[i])
            x_ticks_value = np.linspace(0, spectra_data.shape[0], len(x_ticks_t))
            x_ticks = [[(x_ticks_value[i], f"{x_ticks_t[i]:.1f}") for i in range(len(x_ticks_t))]]
            self.spectra_image_view.view.getAxis('bottom').setTicks(x_ticks)
        else:
            if self.temp_fig is None:
                self.temp_fig = plt.figure()
            else:
                self.temp_fig.clf()
            gs = self.temp_fig.add_gridspec(1, 1, left=0.1, right=0.99)
            self.temp_ax = self.temp_fig.add_subplot(gs[0, 0])
            self.temp_fig.set_size_inches(EXPORT_MPL_WIDTH, EXPORT_MPL_HEIGHT)
            spectra_data = np.flip(spectra_data, axis=1)
            spectra_data = spectra_data.transpose()
            avg_int = np.mean(spectra_data[1:5, :])
            spectra_data -= avg_int
            times = particle.spectra.series_times
            wavelengths = np.flip(particle.spectra.wavelengths)
            c = self.temp_ax.pcolormesh(times, wavelengths, spectra_data,
                                        shading='auto',
                                        cmap='inferno')
            c_bar = self.temp_fig.colorbar(c, ax=self.temp_ax)
            c_bar.set_label('Intensity (counts/s)')
            self.temp_ax.set(xlabel='times (s)',
                             ylabel='wavelength (nm)',
                             title=f'{particle.name} Spectral Trace')


        # self.spectra_image_view.view.getAxis('left').setTicks([particle.spectra.wavelengths.tolist()])

        if for_export and export_path is not None:
            full_path = os.path.join(export_path, f"{particle.name} spectra.png")
            self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
            sleep(1)
            # ex = ImageExporter(self.spectra_image_view.getView())
            # ex.parameters()['width'] = EXPORT_WIDTH
            # ex.export(full_path)

        # print('here')
        # self.spectra_widget.getImageItem().setLookupTable(self._look_up_table)

        if lock:
            self.mainwindow.lock.release()


class MyCrosshairOverlay(pg.CrosshairROI):
    def __init__(self, pos=None, size=None, **kargs):
        self._shape = None
        pg.ROI.__init__(self, pos, size, **kargs)
        self.sigRegionChanged.connect(self.invalidate)
        self.aspectLocked = True


class RasterScanController(QObject):

    def __init__(self, main_window: MainWindow, raster_scan_image_view: pg.ImageView,
                 list_text: QTextBrowser):
        super().__init__()
        self.main_window = main_window
        self.raster_scan_image_view = raster_scan_image_view
        self.raster_scan_image_view.setPredefinedGradient('plasma')
        self.list_text = list_text
        self._crosshair_item = None
        self._text_item = None

        self.temp_fig = None
        self.temp_ax = None

    @staticmethod
    def create_crosshair_item(pos: Tuple[int, int]) -> MyCrosshairOverlay:
        pen = QPen(Qt.green, 0.1)
        pen.setWidthF(0.5)
        crosshair_item = MyCrosshairOverlay(pos=pos, size=2, pen=pen, movable=False)
        return crosshair_item

    @staticmethod
    def create_text_item(text: str, pos: Tuple[int, int]) -> pg.TextItem:
        text_item = pg.TextItem(text=text)
        # Offset
        text_item.setPos(*pos)
        text_item.setColor(QColor(Qt.green))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        text_item.setFont(font)
        text_item.setAnchor(anchor=(1, 0))
        return text_item

    def plot_raster_scan(self, particle: Particle = None,
                         raster_scan: RasterScan = None,
                         for_export: bool = False,
                         export_path: str = None,
                         lock: bool = False):
        if type(export_path) is bool:
            lock = export_path
            export_path = None
        dataset = self.main_window.current_dataset
        if particle is None and raster_scan is None:
            particle = self.main_window.current_particle
            raster_scan = particle.raster_scan
        elif particle is not None:
            if raster_scan is None:
                raster_scan = particle.raster_scan
        else:
            particle = dataset.particles[raster_scan.particle_indexes[0]]
        raster_scan_data = raster_scan.dataset[:]
        # data_shape = raster_scan_data.shape
        # current_ratio = data_shape[1]/data_shape[0]
        # self.raster_scan_image_view.getView().setAspectLocked(False, current_ratio)

        if for_export and export_path is not None:
            if self.temp_fig is None:
                self.temp_fig = plt.figure()
            else:
                self.temp_fig.clear()
                # self.temp_ax.clear()
            self.temp_fig.set_size_inches(8, 8)
            gs = self.temp_fig.add_gridspec(1, 1, left=0.1, right=0.97, bottom=0.05, top=0.97)
            self.temp_ax = self.temp_fig.add_subplot(gs[0, 0])

            left = raster_scan.x_start
            right = left + raster_scan.range
            bottom = raster_scan.y_start
            top = bottom + raster_scan.range
            raster_scan_data = np.flip(raster_scan_data, axis=1)
            raster_scan_data = raster_scan_data.transpose()
            c = self.temp_ax.imshow(raster_scan_data,
                                    cmap='inferno',
                                    aspect='equal',
                                    extent=(left, right, bottom, top))
            if len(raster_scan.particle_indexes) > 1:
                first_part = raster_scan.particle_indexes[0] + 1
                last_part = raster_scan.particle_indexes[-1] + 1
                title = f"Raster Scan for Particles {first_part}-{last_part}"
            else:
                title = f"Raster Scan for Particle {raster_scan.particle_indexes[0] + 1}"
            self.temp_ax.set(xlabel='x axis (um)',
                             ylabel='y axis (um)',
                             title=title)
            c_bar = self.temp_fig.colorbar(c, ax=self.temp_ax)
            c_bar.set_label('intensity (counts/s)')
            hw = 0.25
            for ind in raster_scan.particle_indexes:
                part_pos = dataset.particles[ind].raster_scan_coordinates
                h_line = ([part_pos[0]-hw, part_pos[0]+hw], [part_pos[1], part_pos[1]])
                v_line = ([part_pos[0], part_pos[0]], [part_pos[1]-hw, part_pos[1]+hw])
                self.temp_ax.plot(*h_line, linewidth=3, color='lightgreen')
                self.temp_ax.plot(*v_line, linewidth=3, color='lightgreen')
                self.temp_ax.text(part_pos[0]-hw, part_pos[1]-hw, str(ind+1),
                                  fontsize=14,
                                  color='lightgreen',
                                  fontweight='heavy')
            full_path = os.path.join(export_path, title + '.png')
            self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)

            # if self._crosshair_item is not None:
            #     self.raster_scan_image_view.getView().removeItem(self._crosshair_item)
            # if self._text_item is not None:
            #     self.raster_scan_image_view.getView().removeItem(self._text_item)
            # all_crosshair_items = list()
            # all_text_items = list()
            # for part_index in raster_scan.particle_indexes:
            #     this_particle = dataset.particles[part_index]
            #     raw_coords = this_particle.raster_scan_coordinates
            #     coords = (raw_coords[0] - raster_scan.x_start) / um_per_pixel, \
            #              (raw_coords[1] - raster_scan.y_start) / um_per_pixel
            #     crosshair_item = self.create_crosshair_item(pos=coords)
            #     all_crosshair_items.append(crosshair_item)
            #     text_item = self.create_text_item(text=str(this_particle.dataset_ind + 1),
            #                                       pos=coords)
            #     all_text_items.append(text_item)
            #     self.raster_scan_image_view.getView().addItem(crosshair_item)
            #     self.raster_scan_image_view.getView().addItem(text_item)
            #
            #     # for text_item in all_text_items:
            #     #     text_item.setScale(0.1)
            #
            # self.raster_scan_image_view.autoRange()
            # self.raster_scan_image_view.autoLevels()
            # # self.raster_scan_image_view.getView()
            # sleep(1)
            # full_path = os.path.join(export_path,
            #                          f"Raster Scan {raster_scan.dataset_index + 1}.png")
            # ex = ImageExporter(self.raster_scan_image_view.getView())
            # ex.parameters()['width'] = EXPORT_WIDTH
            # image = ex.export(toBytes=True)
            # image.save(full_path)
            #
            # for crosshair_item in all_crosshair_items:
            #     self.raster_scan_image_view.getView().removeItem(crosshair_item)
            # for text_item in all_text_items:
            #     self.raster_scan_image_view.getView().removeItem(text_item)
            # if self._crosshair_item is not None:
            #     self.raster_scan_image_view.getView().addItem(self._crosshair_item)
            # if self._text_item is not None:
            #     self.raster_scan_image_view.getView().addItem(self._text_item)

        else:
            self.raster_scan_image_view.setImage(raster_scan_data)

            um_per_pixel = raster_scan.range / particle.raster_scan.pixel_per_line

            raw_coords = particle.raster_scan_coordinates
            coords = (raw_coords[0] - raster_scan.x_start) / um_per_pixel, \
                     (raw_coords[1] - raster_scan.y_start) / um_per_pixel

            if self._crosshair_item is None:
                self._crosshair_item = self.create_crosshair_item(pos=coords)
                self.raster_scan_image_view.getView().addItem(self._crosshair_item)
            else:
                self._crosshair_item.setPos(pos=coords)

            # part_text = particle.name.split(' ')[1]
            # Offset text
            text_coords = (coords[0] - 1, coords[1] + 1)
            if self._text_item is None:
                self._text_item = self.create_text_item(text=str(particle.dataset_ind + 1),
                                                        pos=text_coords)
                self.raster_scan_image_view.getView().addItem(self._text_item)
            else:
                self._text_item.setText(text=str(particle.dataset_ind + 1))
                self._text_item.setPos(*text_coords)

            self.list_text.clear()
            dataset = self.main_window.current_dataset
            all_text = f"<h3>Raster Scan {raster_scan.dataset_index + 1}</h3><p>"
            rs_part_coord = [dataset.particles[part_ind].raster_scan_coordinates
                             for part_ind in raster_scan.particle_indexes]
            all_text = all_text + f"<p>Range (um) = {raster_scan.range}<br></br>" \
                                  f"Pixels per line = {raster_scan.pixel_per_line}<br></br>" \
                                  f"Int time (ms/um) = {raster_scan.integration_time}<br></br>" \
                                  f"X Start (um) = {raster_scan.x_start: .1f}<br></br>" \
                                  f"Y Start (um) = {raster_scan.y_start: .1f}</p><p>"
            for num, part_index in enumerate(raster_scan.particle_indexes):
                if num != 0:
                    all_text = all_text + "<br></br>"
                if particle is dataset.particles[part_index]:
                    all_text = all_text + \
                               f"<strong>{num + 1}) {particle.name}</strong>: "
                else:
                    all_text = all_text + f"{num + 1}) {dataset.particles[part_index].name}: "
                all_text = all_text + f"x={rs_part_coord[num][0]: .1f}, " \
                                      f"y={rs_part_coord[num][1]: .1f}"
            self.list_text.setText(all_text)

        if lock:
            self.main_window.lock.release()

# def resolve_levels(start_progress_sig: pyqtSignal, progress_sig: pyqtSignal,
#                    status_sig: pyqtSignal,
#                    reset_gui_sig: pyqtSignal,
#                    level_resolved_sig: pyqtSignal,
#                    conf: Union[int, float], data: H5dataset,
#                    currentparticle: Particle,
#                    mode: str,
#                    resolve_selected=None,
#                    end_time_s=None) -> None:
#     """
#     TODO: edit the docstring
#     Resolves the levels in particles by finding the change points in the
#     abstimes data of a Particle instance.
#
#     Parameters
#     ----------
#     end_time_s
#     currentparticle : Particle
#     conf
#     level_resolved_sig
#     reset_gui_sig
#     data : H5dataset
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
#     # print(mode)
#     assert mode in ['current', 'selected', 'all'], \
#         "'resolve_all' and 'resolve_selected' can not both be given as parameters."
#
#     if mode == 'current':  # Then resolve current
#         currentparticle.cpts.run_cpa(confidence=conf / 100, run_levels=True, end_time_s=end_time_s)
#
#     else:
#         if mode == 'all':  # Then resolve all
#             status_text = 'Resolving All Particle Levels...'
#             parts = data.particles
#
#         elif mode == 'selected':  # Then resolve selected
#             assert resolve_selected is not None, \
#                 'No selected particles provided.'
#             status_text = 'Resolving Selected Particle Levels...'
#             parts = resolve_selected
#
#         try:
#             status_sig.emit(status_text)
#             start_progress_sig.emit(len(parts))
#             for num, part in enumerate(parts):
#                 logger.info(f'Busy Resolving Particle {num + 1}')
#                 part.cpts.run_cpa(confidence=conf, run_levels=True, end_time_s=end_time_s)
#                 progress_sig.emit()
#             status_sig.emit('Done')
#         except Exception as exc:
#             raise RuntimeError("Couldn't resolve levels.") from exc
#
#     level_resolved_sig.emit()
#     data.makehistograms(progress=False)
#     reset_gui_sig.emit()


# def group_levels(start_progress_sig: pyqtSignal,
#                  progress_sig: pyqtSignal,
#                  status_sig: pyqtSignal,
#                  reset_gui_sig: pyqtSignal,
#                  data: H5dataset,
#                  mode: str,
#                  currentparticle: Particle = None,
#                  group_selected=None) -> None:
#     """
#     TODO: edit the docstring
#     Resolves the levels in particles by finding the change points in the
#     abstimes data of a Particle instance.
#
#     Parameters
#     ----------
#     currentparticle : Particle
#     conf
#     level_resolved_sig
#     reset_gui_sig
#     data : H5dataset
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
#     # print(mode)
#     assert mode in ['current', 'selected', 'all'], \
#         "'resolve_all' and 'resolve_selected' can not both be given as parameters."
#
#     if mode == 'current':
#         status_text = 'Grouping Current Particle Levels...'
#         parts = [currentparticle]
#     elif mode == 'all':  # Then resolve all
#         status_text = 'Grouping All Particle Levels...'
#         parts = data.particles
#
#     elif mode == 'selected':  # Then resolve selected
#         assert group_selected is not None, \
#             'No selected particles provided.'
#         status_text = 'Grouping Selected Particle Levels...'
#         parts = group_selected
#
#     try:
#         status_sig.emit(status_text)
#         start_progress_sig.emit(len(parts))
#         for num, part in enumerate(parts):
#             logger.info(f'Busy Grouping Particle {num + 1}')
#             part.ahca.run_grouping()
#             progress_sig.emit()
#         status_sig.emit('Done')
#     except Exception as exc:
#         raise RuntimeError("Couldn't group levels.") from exc

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
