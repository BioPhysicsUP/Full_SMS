from __future__ import annotations

__docformat__ = "NumPy"

import os
import pickle
import random
import tempfile
import time
from copy import copy
from time import sleep
from typing import TYPE_CHECKING, Union, List, Tuple, Any

import numpy as np
import pandas as pd
import pyqtgraph.graphicsItems.PlotCurveItem
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from PyQt5.QtCore import QObject, Qt
from PyQt5.QtGui import QPalette, QFont, QBrush, QPen, QColor
from PyQt5.QtWidgets import QWidget, QFrame, QFileDialog, QTextBrowser, QCheckBox
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
from pyqtgraph.exporters import ImageExporter

import grouping
from threads import ProcessThread, ProcessTaskResult
from dataclasses import dataclass

from my_logger import setup_logger
from smsh5 import Particle, ParticleAllHists, RasterScan, GlobalParticle
from grouping import GlobalLevel, Group
from change_point import Level
from tcspcfit import FittingParameters, FittingDialog
from quick_roi_dialog import QuickROIDialog
from thread_tasks import OpenFile

if TYPE_CHECKING:
    from main import MainWindow

EXPORT_WIDTH = 1500
EXPORT_MPL_WIDTH = 10
EXPORT_MPL_HEIGHT = 4.5
EXPORT_MPL_DPI = 300

SIG_ROI_CHANGE_THRESHOLD = 0.1  # counts/s
logger = setup_logger(__name__)


def export_plot_item(plot_item: pg.PlotItem, path: str, text: str = None):
    left_autofill = plot_item.getAxis("left").autoFillBackground()
    bottom_autofill = plot_item.getAxis("bottom").autoFillBackground()
    vb_autofill = plot_item.vb.autoFillBackground()
    # left_label_font = plot_item.getAxis('left').label.font()
    # bottom_label_font = plot_item.getAxis('bottom').label.font()
    # left_axis_font = plot_item.getAxis('left').font()
    # bottom_axis_font = plot_item.getAxis('bottom').font()

    if text is not None:
        f_size = plot_item.height() * 7 / 182
        font = QFont()
        font.setPixelSize(f_size)
        text_item = pg.TextItem(text=text, color="k", anchor=(1, 0))
        text_item.setFont(font)
        plot_item.addItem(text_item, ignoreBounds=True)
        text_item.setPos(plot_item.vb.width(), 0)
        text_item.setParentItem(plot_item.vb)

    plot_item.getAxis("left").setAutoFillBackground(True)
    plot_item.getAxis("bottom").setAutoFillBackground(True)
    plot_item.vb.setAutoFillBackground(True)

    # new_label_point_size = plot_item.height() * 10.0/486.0
    # plot_item.getAxis('left').label.font().setPointSizeF(new_label_point_size)
    # plot_item.getAxis('bottom').label.font().setPointSizeF(new_label_point_size)
    # new_axis_point_size = plot_item.height() * 8.25/486
    # plot_item.getAxis('left').font().setPointSizeF(new_axis_point_size)
    # plot_item.getAxis('bottom').font().setPointSizeF(new_axis_point_size)

    ex = ImageExporter(plot_item.scene())
    ex.parameters()["width"] = EXPORT_WIDTH
    ex.export(path)

    plot_item.getAxis("left").setAutoFillBackground(left_autofill)
    plot_item.getAxis("bottom").setAutoFillBackground(bottom_autofill)
    plot_item.vb.setAutoFillBackground(vb_autofill)
    # plot_item.getAxis('left').label.setFont(left_label_font)
    # plot_item.getAxis('bottom').label.setFont(bottom_label_font)
    # plot_item.getAxis('left').setFont(left_axis_font)
    # plot_item.getAxis('bottom').setFont(bottom_axis_font)


class IntController(QObject):
    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.main_window = main_window
        self.resolve_mode = None
        self.results_gathered = False

        self.int_widget = self.main_window.pgIntensity_PlotWidget
        self.int_plot = self.main_window.pgIntensity_PlotWidget.getPlotItem()

        self.setup_widget(self.int_widget)
        self.setup_plot(self.int_plot)

        self.int_hist_container = self.main_window.wdgInt_Hist_Container
        self.show_int_hist = self.main_window.chbInt_Show_Hist.isChecked()
        self.int_hist_line = self.main_window.lineInt_Hist
        self.int_hist_widget = self.main_window.pgInt_Hist_PlotWidget
        self.int_hist_plot = self.main_window.pgInt_Hist_PlotWidget.getPlotItem()
        self.setup_widget(self.int_hist_widget)
        self.setup_plot(self.int_hist_plot, is_int_hist=True)

        self.lifetime_widget = self.main_window.pgLifetime_Int_PlotWidget
        self.lifetime_plot = self.main_window.pgLifetime_Int_PlotWidget.getPlotItem()
        self.setup_widget(self.lifetime_widget)
        self.setup_plot(self.lifetime_plot)

        self.group_int_widget = self.main_window.pgGroups_Int_PlotWidget
        self.groups_int_plot = self.main_window.pgGroups_Int_PlotWidget.getPlotItem()
        self.setup_widget(self.group_int_widget)
        self.setup_plot(self.groups_int_plot)

        self.groups_hist_widget = self.main_window.pgGroups_Hist_PlotWidget
        self.groups_hist_plot = self.main_window.pgGroups_Hist_PlotWidget.getPlotItem()
        self.setup_widget(self.groups_hist_widget)
        self.setup_plot(self.groups_hist_plot, is_group_hist=True)

        self.int_level_info_container = self.main_window.wdgInt_Level_Info_Container
        self.level_info_text = self.main_window.txtLevelInfoInt
        self.int_level_line = self.main_window.lineInt_Level
        self.show_level_info = self.main_window.chbInt_Show_Level_Info.isChecked()
        self.hide_show_chb(chb_obj=self.main_window.chbInt_Show_Level_Info, show=False)
        mw_bg_colour = self.main_window.palette().color(QPalette.Background)
        level_info_palette = self.level_info_text.viewport().palette()
        level_info_palette.setColor(QPalette.Base, mw_bg_colour)
        self.level_info_text.viewport().setPalette(level_info_palette)

        self.show_exp_trace = self.main_window.chbInt_Exp_Trace.isChecked()

        self.int_plot.vb.scene().sigMouseClicked.connect(self.any_int_plot_double_click)
        self.groups_int_plot.vb.scene().sigMouseClicked.connect(
            self.any_int_plot_double_click
        )
        self.lifetime_plot.vb.scene().sigMouseClicked.connect(
            self.any_int_plot_double_click
        )

        self.temp_fig = None
        self.temp_ax = None
        self.temp_bins = None

        # Setup and addition of Linear Region Item for ROI
        pen = QPen()
        pen.setCosmetic(True)
        pen.setWidthF(1)
        pen.setStyle(Qt.DashLine)
        pen.setColor(QColor("grey"))

        hover_pen = QPen()
        hover_pen.setCosmetic(True)
        hover_pen.setWidthF(2)
        hover_pen.setStyle(Qt.DashLine)
        hover_pen.setColor(QColor("red"))

        brush_color = QColor("lightgreen")
        brush_color.setAlpha(20)
        brush = QBrush()
        brush.setStyle(Qt.SolidPattern)
        brush.setColor(brush_color)

        hover_brush_color = QColor("lightgreen")
        hover_brush_color.setAlpha(80)
        hover_brush = QBrush()
        hover_brush.setStyle(Qt.SolidPattern)
        hover_brush.setColor(hover_brush_color)

        self.int_ROI = pg.LinearRegionItem(
            brush=brush, hoverBrush=hover_brush, pen=pen, hoverPen=hover_pen
        )
        self.int_ROI.sigRegionChangeFinished.connect(self.roi_region_changed)

        # Setup axes and limits
        # self.groups_hist_plot.getAxis('bottom').setLabel('Relative Frequency')

        self.confidence_index = {0: 99, 1: 95, 2: 90, 3: 69}

        self.main_window.btnApplyBin.clicked.connect(self.gui_apply_bin)
        self.main_window.btnApplyBinAll.clicked.connect(self.gui_apply_bin_all)
        self.main_window.btnResolve.clicked.connect(self.gui_resolve)
        self.main_window.btnResolve_Selected.clicked.connect(self.gui_resolve_selected)
        self.main_window.btnResolveAll.clicked.connect(self.gui_resolve_all)
        self.main_window.chbInt_Show_ROI.stateChanged.connect(self.roi_chb_changed)
        self.main_window.chbInt_Show_Hist.stateChanged.connect(self.hist_chb_changed)
        self.main_window.chbInt_Show_Level_Info.stateChanged.connect(
            self.level_info_chb_changed
        )
        self.main_window.chbInt_Show_Groups.stateChanged.connect(
            self.gui_chb_show_groups
        )
        self.main_window.chbInt_Show_Global_Groups.stateChanged.connect(
            self.gui_chb_show_global_groups
        )
        self.main_window.btnQuickROI.clicked.connect(self.gui_quick_roi)
        self.main_window.chbInt_Exp_Trace.stateChanged.connect(
            self.exp_trace_chb_changed
        )
        self.main_window.chbSecondCard.stateChanged.connect(self.plot_all)

    def setup_plot(
        self,
        plot_item: pg.PlotItem,
        is_int_hist: bool = False,
        is_group_hist: bool = False,
        is_lifetime: bool = False,
    ):
        # Set axis label bold and size
        axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)

        left_axis = plot_item.getAxis("left")
        bottom_axis = plot_item.getAxis("bottom")

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
            bottom_axis.setLabel("Relative Frequency")
            if is_int_hist:
                plot_item.setYLink(self.int_plot.getViewBox())
            else:
                plot_item.setYLink(self.groups_int_plot.getViewBox())
            plot_item.vb.setLimits(xMin=0, xMax=1, yMin=0)
        else:
            left_axis.setLabel("Intensity", "counts/100ms")
            if not is_lifetime:
                bottom_axis.setLabel("Time", "s")
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
        if chb_obj is self.main_window.chbInt_Show_Level_Info:
            if show:
                self.int_level_info_container.show()
                self.int_level_line.show()
            else:
                self.int_level_info_container.hide()
                self.int_level_line.hide()
        elif chb_obj is self.main_window.chbInt_Show_Hist:
            if show:
                self.int_hist_container.show()
                self.int_hist_line.show()
            else:
                self.int_hist_container.hide()
                self.int_hist_line.hide()

    def roi_chb_changed(self):
        roi_chb = self.main_window.chbInt_Show_ROI
        chb_text = "Hide ROI"
        if roi_chb.checkState() == 1:
            chb_text = "Show ROI"
        elif roi_chb.checkState() == 2:
            chb_text = "Edit ROI"
        roi_chb.setText(chb_text)
        self.plot_all()

    def roi_region_changed(self):
        if self.main_window.chbInt_Show_ROI.checkState() == 2:
            cur_part = self.main_window.current_particle
            cur_tab_name = self.main_window.tabWidget.currentWidget().objectName()
            if cur_part is not None and cur_tab_name == "tabIntensity":
                new_region = self.int_ROI.getRegion()
                old_region = cur_part.roi_region[0:2]
                significant_start_change = (
                    np.abs(new_region[0] - old_region[0]) > SIG_ROI_CHANGE_THRESHOLD
                )
                significant_end_change = (
                    np.abs(new_region[1] - old_region[1]) > SIG_ROI_CHANGE_THRESHOLD
                )
                if significant_start_change or significant_end_change:
                    cur_part.roi_region = self.int_ROI.getRegion()
                    self.main_window.lifetime_controller.test_need_roi_apply(
                        particle=cur_part, update_buttons=False
                    )
                    if cur_part.level_or_group_selected is not None:
                        if (
                            cur_part.first_level_ind_in_roi
                            < cur_part.level_or_group_selected
                            > cur_part.last_level_ind_in_roi
                        ):
                            cur_part.level_or_group_selected = None
                self.plot_all()
                # self.plot_hist()
                if self.main_window.chbInt_Show_Level_Info.isChecked():
                    self.update_level_info()

    def hist_chb_changed(self):
        self.show_int_hist = self.main_window.chbInt_Show_Hist.isChecked()

        if self.show_int_hist:
            if self.show_level_info:
                self.hide_show_chb(
                    chb_obj=self.main_window.chbInt_Show_Level_Info, show=False
                )
                self.show_level_info = False
            self.int_hist_container.show()
            self.int_hist_line.show()
            self.plot_hist()
        else:
            self.int_hist_container.hide()
            self.int_hist_line.hide()

    def level_info_chb_changed(self):
        self.show_level_info = self.main_window.chbInt_Show_Level_Info.isChecked()

        if self.show_level_info:
            if self.show_int_hist:
                self.hide_show_chb(
                    chb_obj=self.main_window.chbInt_Show_Hist, show=False
                )
                self.show_int_hist = False
            self.int_level_info_container.show()
            self.int_level_line.show()
            self.update_level_info()
        else:
            self.int_level_info_container.hide()
            self.int_level_line.hide()

    def exp_trace_chb_changed(self):
        self.show_exp_trace = self.main_window.chbInt_Exp_Trace.isChecked()
        self.main_window.display_data()
        self.main_window.repaint()
        logger.info("Show experimental trace")

    def gui_apply_bin(self):
        """Changes the bin size of the data of the current particle and then displays the new trace."""
        try:
            self.main_window.current_particle.binints(self.get_bin())
        except Exception as err:
            logger.error("Error Occured:")
        else:
            self.main_window.display_data()
            self.main_window.repaint()
            logger.info("Single trace binned")

    def gui_chb_show_groups(self):
        if self.main_window.chbInt_Show_Groups.isChecked():
            self.main_window.chbInt_Show_Global_Groups.setChecked(False)
        self.plot_all()

    def gui_chb_show_global_groups(self):
        if self.main_window.chbInt_Show_Global_Groups.isChecked():
            self.main_window.chbInt_Show_Groups.setChecked(False)
        self.plot_all()

    def get_bin(self) -> int:
        """Returns current GUI value for bin size in ms.

        Returns
        -------
        int
            The value of the bin size on the GUI in spbBinSize.
        """

        return self.main_window.spbBinSize.value()

    def set_bin(self, new_bin: int):
        """Sets the GUI value for the bin size in ms

        Parameters
        ----------
        new_bin: int
            Value to set bin size to, in ms.
        """
        self.main_window.spbBinSize.setValue(new_bin)

    def gui_apply_bin_all(self):
        """Changes the bin size of the data of all the particles and then displays the new trace of the current particle."""

        self.start_binall_thread(self.get_bin())

    def start_binall_thread(self, bin_size) -> None:
        """

        Parameters
        ----------
        bin_size
        """

        mw = self.main_window
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
            logger.error("Error Occured:")
        else:
            mw.display_data()
            mw.repaint()
            mw.status_message("Done")
            mw.end_progress()
            logger.info("All traces binned")

    def binall_thread_complete(self):
        self.main_window.status_message("Done")
        self.plot_trace()
        logger.info("Binnig all levels complete")

    # def ask_end_time(self):
    #     """ Prompts the user to supply an end time."""
    #
    #     end_time_s, ok = QInputDialog.getDouble(self.mainwindow, 'End Time',
    #                                             'Provide end time in seconds', 0, 1, 10000, 3)
    #     return end_time_s, ok
    #
    # def time_resolve_current(self):
    #     """ Resolves the levels of the current particle to an end time asked of the user."""
    #
    #     end_time_s, ok = self.ask_end_time()
    #     if ok:
    #         self.gui_resolve(end_time_s=end_time_s)
    #
    # def time_resolve_selected(self):
    #     """ Resolves the levels of the selected particles to an end time asked of the user."""
    #
    #     end_time_s, ok = self.ask_end_time()
    #     if ok:
    #         self.gui_resolve_selected(end_time_s=end_time_s)
    #
    # def time_resolve_all(self):
    #     """ Resolves the levels of all the particles to an end time asked of the user."""
    #
    #     end_time_s, ok = self.ask_end_time()
    #     if ok:
    #         self.gui_resolve_all(end_time_s=end_time_s)
    #
    def gui_resolve(self):  # , end_time_s=None):
        """Resolves the levels of the current particle and displays it."""

        self.start_resolve_thread(mode="current")  # , end_time_s=end_time_s)

    def gui_resolve_selected(self):  # , end_time_s=None):
        """Resolves the levels of the selected particles and displays the levels of the current particle."""

        self.start_resolve_thread(mode="selected")  # , end_time_s=end_time_s)

    def gui_resolve_all(self):  # , end_time_s=None):
        """Resolves the levels of the all the particles and then displays the levels of the current particle."""

        self.start_resolve_thread(mode="all")  # , end_time_s=end_time_s)

    def plot_trace(
        self,
        particle: Particle = None,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
    ) -> None:
        """Used to display the trace from the absolute arrival time data of the current particle."""

        mw = self.main_window
        plot_2_trace = False
        if mw.current_particle.sec_part is not None:
            if (
                mw.current_particle.sec_part.tcspc_card != "None"
                and mw.chbSecondCard.isChecked()
            ):
                plot_2_trace = True

        if type(export_path) is bool:
            lock = export_path
            export_path = None
        try:
            # self.currentparticle = self.treemodel.data(self.current_ind, Qt.UserRole)
            if particle is None:
                particle = mw.current_particle
            trace2 = None
            times2 = None
            if self.show_exp_trace and particle.int_trace is not None:
                trace = particle.int_trace[:]
                times = np.linspace(0, np.size(trace) * 0.1, np.size(trace))
            else:
                trace = particle.binnedtrace.intdata
                times = particle.binnedtrace.inttimes / 1e3
                if plot_2_trace:
                    trace2 = particle.sec_part.binnedtrace.intdata
                    times2 = particle.sec_part.binnedtrace.inttimes / 1e3
        except AttributeError:
            logger.error("No trace!")
        else:
            plot_pen = QPen()
            plot_pen.setCosmetic(True)
            plot_pen2 = QPen()
            plot_pen2.setCosmetic(True)
            roi_chb_value = mw.chbInt_Show_ROI.checkState()
            roi_state = "none"
            if roi_chb_value == 1:
                roi_state = "show"
            elif roi_chb_value == 2:
                roi_state = "edit"
            if for_export:
                cur_tab_name = "tabIntensity"
            else:
                cur_tab_name = mw.tabWidget.currentWidget().objectName()

            if cur_tab_name != "tabSpectra":
                if cur_tab_name == "tabIntensity":
                    plot_item = self.int_plot
                    plot_pen.setWidthF(1.5)
                    plot_pen.setColor(QColor("green"))
                    plot_pen2.setWidthF(1.5)
                    plot_pen2.setColor(QColor("blue"))
                elif cur_tab_name == "tabLifetime":
                    plot_item = self.lifetime_plot
                    plot_pen.setWidthF(1.1)
                    plot_pen.setColor(QColor("green"))
                    plot_pen2.setWidthF(1.1)
                    plot_pen2.setColor(QColor("blue"))
                elif cur_tab_name == "tabGrouping":
                    plot_item = self.groups_int_plot
                    plot_pen.setWidthF(1.1)
                    plot_pen.setColor(QColor(0, 0, 0, 50))
                    plot_pen2.setWidthF(1.1)
                    plot_pen2.setColor(QColor("blue"))
                else:
                    return

                unit = "counts/" + str(self.get_bin()) + "ms"
                if not for_export:
                    plot_pen.setJoinStyle(Qt.RoundJoin)

                    plot_item.clear()
                    if roi_state != "none":
                        if roi_state == "edit" and cur_tab_name == "tabIntensity":
                            self.int_ROI.setMovable(True)
                            self.int_ROI.setBounds((0, times[-1]))
                        else:
                            self.int_ROI.setMovable(False)

                        new_region = self.int_ROI.getRegion()
                        old_region = particle.roi_region[0:2]
                        significant_start_change = (
                            np.abs(new_region[0] - old_region[0])
                            > SIG_ROI_CHANGE_THRESHOLD
                        )
                        significant_end_change = (
                            np.abs(new_region[1] - old_region[1])
                            > SIG_ROI_CHANGE_THRESHOLD
                        )
                        if significant_start_change or significant_end_change:
                            self.int_ROI.setRegion(particle.roi_region)
                        plot_item.addItem(self.int_ROI)
                    plot_item.getAxis("left").setLabel(text="Intensity", units=unit)
                    plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=times[-1])
                    plot_item.plot(x=times, y=trace, pen=plot_pen, symbol=None)
                    if plot_2_trace and cur_tab_name == "tabIntensity":
                        # print("shloop")
                        # print(mw.current_particle.sec_part.tcspc_card)
                        plot_item.plot(x=times2, y=trace2, pen=plot_pen2, symbol=None)

                else:
                    if self.temp_fig is None:
                        self.temp_fig = plt.figure()
                        self.temp_fig.set_size_inches(
                            EXPORT_MPL_WIDTH, EXPORT_MPL_HEIGHT
                        )
                    else:
                        self.temp_fig.clf()
                    gs = self.temp_fig.add_gridspec(
                        nrows=1, ncols=5, wspace=0, left=0.07, right=0.98
                    )
                    int_ax = self.temp_fig.add_subplot(gs[0, :-1])
                    hist_ax = self.temp_fig.add_subplot(gs[0, -1])
                    self.temp_ax = {"int_ax": int_ax, "hist_ax": hist_ax}
                    hist_ax.tick_params(
                        direction="in", labelleft=False, labelbottom=False
                    )
                    hist_ax.spines["top"].set_visible(False)
                    hist_ax.spines["right"].set_visible(False)
                    int_ax.plot(times, trace)
                    int_ax.set(
                        xlabel="time (s)",
                        ylabel=f"intensity {unit}",
                        xlim=[0, times[-1]],
                        ylim=[0, max(trace)],
                    )
                    int_ax.spines["top"].set_visible(False)
                    self.temp_fig.suptitle(f"{particle.name} Intensity Trace")
                    self.plot_hist(
                        particle=particle,
                        for_export=for_export,
                        export_path=export_path,
                        for_levels=False,
                    )
        if lock:
            mw.lock.release()

    def plot_levels(
        self,
        particle: Particle = None,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
    ):
        """Used to plot the resolved intensity levels of the current particle."""
        if type(export_path) is bool:
            lock = export_path
            export_path = None
        if particle is None:
            particle = self.main_window.current_particle
        if not particle.has_levels:
            return
        try:
            is_tab_intensity = (
                self.main_window.tabWidget.currentWidget().objectName()
                == "tabIntensity"
            )
            should_use_global = (
                self.main_window.chbInt_Show_Global_Groups.isChecked()
                and is_tab_intensity
            )
            do_use_global = particle.has_global_grouping and should_use_global
            use_roi = self.main_window.chbInt_Show_ROI.isChecked()
            level_ints, times = particle.levels2data(
                use_roi=use_roi, use_global_groups=do_use_global
            )
            level_ints = level_ints * self.get_bin() / 1e3
        except AttributeError:
            logger.error("No levels!")
            return

        cur_tab_name = self.main_window.tabWidget.currentWidget().objectName()
        if not for_export:
            plot_pen = QPen()
            if cur_tab_name == "tabIntensity":
                plot_item = self.int_plot
                # pen_width = 1.5
                plot_pen.setWidthF(1.5)
                plot_pen.setColor(QColor("black"))
            elif cur_tab_name == "tabLifetime":
                plot_item = self.lifetime_plot
                # pen_width = 1.1
                plot_pen.setWidthF(1.1)
                plot_pen.setColor(QColor("black"))
            elif cur_tab_name == "tabGrouping":
                plot_item = self.groups_int_plot
                plot_pen.setWidthF(1)
                plot_pen.setColor(QColor(0, 0, 0, 100))
            else:
                return

            # plot_pen.brush()
            plot_pen.setJoinStyle(Qt.RoundJoin)
            plot_pen.setCosmetic(True)

            _ = plot_item.plot(x=times, y=level_ints, pen=plot_pen, symbol=None)
        else:
            self.temp_ax["int_ax"].plot(times, level_ints, linewidth=0.7)
            self.temp_fig.suptitle(f"{particle.name} Intensity Trace with Levels")
            self.plot_hist(
                particle=particle,
                for_export=True,
                export_path=export_path,
                for_levels=True,
            )

        if not for_export and (
            cur_tab_name == "tabLifetime" or cur_tab_name == "tabIntensity"
        ):
            selected = particle.level_or_group_selected
            is_level = type(selected) is Level or type(selected) is GlobalLevel
            current_level = selected if is_level else None
            current_group = selected if not is_level else None
            if selected is not None:
                if is_level:
                    current_int = current_level.int_p_s
                    current_times = current_level.times_s
                else:
                    current_int = current_group.int_p_s
                    current_times = times[0], times[-1]
                current_int = current_int * self.get_bin() / 1e3
                current_ints = [current_int] * 2

                if not (current_ints[0] == np.inf or current_ints[1] == np.inf):
                    level_plot_pen = QPen()
                    level_plot_pen.setCosmetic(True)
                    level_plot_pen.setJoinStyle(Qt.RoundJoin)
                    level_plot_pen.setColor(QColor("red"))
                    level_plot_pen.setWidthF(3)
                    plot_item.plot(
                        x=current_times, y=current_ints, pen=level_plot_pen, symbol=None
                    )
                else:
                    logger.info("Infinity in level")
        if lock:
            self.main_window.lock.release()

    def plot_hist(
        self,
        particle: Particle = None,
        for_export: bool = False,
        export_path: str = None,
        for_levels: bool = False,
        for_groups: bool = False,
    ):
        if particle is None:
            particle = self.main_window.current_particle
        try:
            int_data = particle.binnedtrace.intdata
        except AttributeError:
            logger.error("No trace!")
        else:
            if self.main_window.chbInt_Show_ROI.isChecked():
                roi_start = particle.roi_region[0]
                roi_end = particle.roi_region[1]
                time_ind_start = np.argmax(
                    roi_start < particle.binnedtrace.inttimes / 1e3
                )
                end_test = roi_end <= particle.binnedtrace.inttimes / 1e3
                if any(end_test):
                    time_ind_end = np.argmax(end_test)
                else:
                    time_ind_end = len(int_data)
                int_data = int_data[time_ind_start : time_ind_end + 1]
            plot_pen = QPen()
            plot_pen.setColor(QColor(0, 0, 0, 0))

            if for_export:
                cur_tab_name = "tabIntensity"
            else:
                cur_tab_name = self.main_window.tabWidget.currentWidget().objectName()

            if cur_tab_name == "tabIntensity":
                if self.show_int_hist or for_export:
                    plot_item = self.int_hist_plot
                else:
                    return
            elif cur_tab_name == "tabGrouping":
                plot_item = self.groups_hist_plot
            else:
                return

            if not for_export:
                plot_item.clear()

                bin_edges = np.histogram_bin_edges(np.negative(int_data), bins=100)
                freq, hist_bins = np.histogram(
                    np.negative(int_data), bins=bin_edges, density=True
                )
                freq /= np.max(freq)
                int_hist = pg.PlotCurveItem(
                    x=hist_bins,
                    y=freq,
                    pen=plot_pen,
                    stepMode=True,
                    fillLevel=0,
                    brush=(0, 0, 0, 50),
                )
                int_hist.setRotation(-90)
                plot_item.addItem(int_hist)
            elif not (for_levels or for_groups):
                hist_ax = self.temp_ax["hist_ax"]
                _, bins, _ = hist_ax.hist(
                    int_data,
                    bins=50,
                    orientation="horizontal",
                    density=True,
                    edgecolor="k",
                    range=self.temp_ax["int_ax"].get_ylim(),
                    label="Trace",
                )
                self.temp_bins = bins
                hist_ax.set_ylim(self.temp_ax["int_ax"].get_ylim())

            if particle.has_levels:
                if not self.main_window.chbInt_Show_ROI.isChecked():
                    level_ints = particle.level_ints
                    dwell_times = [level.dwell_time_s for level in particle.levels]
                else:
                    level_ints = particle.level_ints_roi
                    dwell_times = particle.level_dwelltimes_roi
                level_ints *= particle.bin_size / 1000
                if not for_export:
                    level_freq, level_hist_bins = np.histogram(
                        np.negative(level_ints),
                        bins=bin_edges,
                        weights=dwell_times,
                        density=True,
                    )
                    level_freq /= np.max(level_freq)
                    level_hist = pg.PlotCurveItem(
                        x=level_hist_bins,
                        y=level_freq,
                        stepMode=True,
                        pen=plot_pen,
                        fillLevel=0,
                        brush=(0, 0, 0, 255),
                    )

                    level_hist.setRotation(-90)
                    plot_item.addItem(level_hist)
                elif for_levels and particle.has_levels:
                    hist_ax = self.temp_ax["hist_ax"]
                    hist_ax.hist(
                        level_ints,
                        bins=50,
                        weights=dwell_times,
                        orientation="horizontal",
                        density=True,
                        # rwidth=0.5,
                        edgecolor="k",
                        linewidth=0.5,
                        alpha=0.4,
                        range=self.temp_ax["int_ax"].get_ylim(),
                        label="Resolved",
                    )
                elif for_groups and particle.has_groups:
                    group_ints = np.array(particle.groups_ints)
                    group_ints *= particle.bin_size / 1000
                    group_dwell_times = [
                        group.dwell_time_s for group in particle.groups
                    ]
                    hist_ax = self.temp_ax["hist_ax"]
                    hist_ax.hist(
                        group_ints,
                        bins=50,
                        weights=group_dwell_times,
                        orientation="horizontal",
                        density=True,
                        # rwidth=0.3,
                        # color='k',
                        fill=False,
                        hatch="///",
                        edgecolor="k",
                        linewidth=0.5,
                        # alpha=0.4,
                        range=self.temp_ax["int_ax"].get_ylim(),
                        label="Grouped",
                    )

        if for_export and export_path is not None:
            if not (os.path.exists(export_path) and os.path.isdir(export_path)):
                raise AssertionError("Provided path not valid")
            if not (for_levels or for_groups):
                full_path = os.path.join(export_path, particle.name + " trace.png")
            elif for_levels:
                full_path = os.path.join(
                    export_path, particle.name + " trace (levels).png"
                )
            else:
                full_path = os.path.join(
                    export_path, particle.name + " trace (levels and groups).png"
                )
            hist_ax.legend(prop={"size": 6}, frameon=False)
            self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
            sleep(1)

    def update_level_info(self, particle: Particle = None):
        if particle is None:
            particle = self.main_window.current_particle

        cur_tab_name = self.main_window.tabWidget.currentWidget().objectName()
        if cur_tab_name == "tabIntensity" and self.show_level_info:
            info = ""
            if particle.level_or_group_selected is None:
                info = info + "Whole Trace"
                info = info + f"\n{'*' * len(info)}"
                info = info + f"\nTotal Dwell Time (s) = {particle.dwell_time_s: .3g}"
                info = info + f"\n# of Photons = {particle.num_photons}"
                if particle.has_levels:
                    info = info + f"\n# of Levels = {particle.num_levels}"
                if particle.has_groups:
                    info = info + f"\n# of Groups = {particle.num_groups}"
                if particle.has_levels:
                    info = info + f"\nHas Photon Bursts = {particle.has_burst}"

                if self.main_window.chbInt_Show_ROI.isChecked:
                    info += f"\n\nWhole Trace (ROI)\n{'*' * len('Whole Trace (ROI)')}"
                    info = (
                        info
                        + f"\nTotal Dwell Time (s) = {particle.dwell_time_roi: .3g}"
                    )
                    info = info + f"\n# of Photons = {particle.num_photons_roi}"
                    if particle.has_levels:
                        info = info + f"\n# of Levels = {particle.num_levels_roi}"
                    if particle.has_groups:
                        info = info + f"\n# of Groups = {particle.num_groups}"
                    if particle.has_levels:
                        info = info + f"\nHas Photon Bursts = {particle.has_burst}"
            elif particle.has_levels:
                selected = particle.level_or_group_selected
                is_level = type(selected) is Level or type(selected) is GlobalLevel
                current_level = selected if is_level else None
                current_group = selected if not is_level else None
                if is_level:
                    all_levels = (
                        particle.levels
                        if type(current_level) is Level
                        else particle.group_levels
                    )
                    level_ind = np.argmax(
                        [current_level is level for level in all_levels]
                    )
                    info = info + f"Level {level_ind + 1}"
                else:
                    current_level = current_group
                    is_global = False
                    if current_group in particle.groups:
                        group_ind = np.argmax(
                            [current_group is group for group in particle.groups]
                        )
                    elif current_group in particle.global_particle.groups:
                        group_ind = np.argmax(
                            [
                                current_group is group
                                for group in particle.global_particle.groups
                            ]
                        )
                        is_global = True
                    else:
                        raise AttributeError("Group not found in list of known groups?")
                    group_header = (
                        f"Global Group {group_ind + 1}"
                        if is_global
                        else f"Group {group_ind + 1}"
                    )
                    info = info + group_header
                info = info + f"\n{'*' * len(info)}"
                info = info + f"\nIntensity (counts/s) = {current_level.int_p_s: .3g}"
                info = info + f"\nDwell Time (s) = {current_level.dwell_time_s: .3g}"
                info = info + f"\n# of Photons = {current_level.num_photons}"
            self.level_info_text.setText(info)

    def plot_group_bounds(
        self,
        particle: Particle = None,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
    ):
        if type(export_path) is bool:
            lock = export_path
            export_path = None

        if for_export:
            cur_tab_name = "tabIntensity"
        else:
            cur_tab_name = self.main_window.tabWidget.currentWidget().objectName()

        if particle is None:
            particle = self.main_window.current_particle
            grouping_mode_tab_name = (
                self.main_window.tabGroupingMode.currentWidget().objectName()
            )
            is_global_fitting_ui = (
                cur_tab_name == "tabGrouping" and grouping_mode_tab_name == "tabGlobal"
            )
            should_use_global = (
                self.main_window.chbInt_Show_Global_Groups.isChecked()
                or is_global_fitting_ui
            )
            if particle.has_global_grouping and should_use_global:
                particle = self.main_window.current_dataset.global_particle
        if (
            cur_tab_name == "tabIntensity"
            or cur_tab_name == "tabGrouping"
            or cur_tab_name == "tabLifetime"
        ):
            if (
                not particle.has_groups
                or particle.ahca.best_step.single_level
                or particle.ahca.selected_step.num_groups < 2
            ):
                if lock:
                    self.main_window.lock.release()
                return
            try:
                # if particle.has_global_grouping and self.main_window.chbInt_Show_Global_Groups.isChecked():
                #     groups = particle.global_particle.ahca.selected_step.groups
                #     group_bounds = particle.global_particle.ahca.selected_step.calc_int_bounds()
                # else:
                groups = particle.groups
                group_bounds = particle.groups_bounds
            except AttributeError:
                logger.error("No groups!")
                return
            int_plot = None
            if cur_tab_name == "tabIntensity":
                mw = self.main_window
                if (
                    mw.chbInt_Show_Groups.isChecked()
                    or mw.chbInt_Show_Global_Groups.isChecked()
                    or for_export
                ):
                    int_plot = self.int_plot
                else:
                    return
            elif cur_tab_name == "tabGrouping":
                int_plot = self.groups_int_plot
            elif cur_tab_name == "tabLifetime":
                if self.main_window.chbLifetime_Show_Groups.isChecked():
                    int_plot = self.lifetime_plot
                else:
                    return

            int_conv = self.main_window.current_particle.bin_size / 1000

            if for_export:
                int_ax = self.temp_ax["int_ax"]
            for i, bound in enumerate(group_bounds):
                if i % 2:
                    bound = (bound[0] * int_conv, bound[1] * int_conv)
                    if not for_export:
                        int_plot.addItem(
                            pg.LinearRegionItem(
                                values=bound,
                                orientation="horizontal",
                                movable=False,
                                pen=QPen().setWidthF(0),
                            )
                        )
                    else:
                        ymin, ymax = bound
                        int_ax.axhspan(
                            ymin=ymin, ymax=ymax, color="k", alpha=0.15, linestyle=""
                        )

            if not for_export:
                line_pen = QPen()
                line_pen.setWidthF(1)
                line_pen.setStyle(Qt.DashLine)
                line_pen.brush()
                # plot_pen.setJoinStyle(Qt.RoundJoin)
                line_pen.setColor(QColor(0, 0, 0, 150))
                line_pen.setCosmetic(True)
                line_times = [0, self.main_window.current_particle.dwell_time_s]
            for group in groups:
                g_int = group.int_p_s * int_conv
                if not for_export:
                    g_ints = [g_int] * 2
                    int_plot.plot(x=line_times, y=g_ints, pen=line_pen, symbol=None)
                else:
                    int_ax.axhline(g_int, linestyle="--", linewidth=0.5, color="k")

            if for_export and export_path is not None:
                if not (os.path.exists(export_path) and os.path.isdir(export_path)):
                    raise AssertionError("Provided path not valid")
                full_path = os.path.join(
                    export_path, particle.name + " trace (levels and groups).png"
                )
                self.temp_fig.suptitle(
                    f"{particle.name} Intensity Trace with Levels and Groups"
                )
                self.plot_hist(
                    particle=particle,
                    for_export=for_export,
                    export_path=export_path,
                    for_groups=True,
                )
                # self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
                # export_plot_item(plot_item=int_plot, path=full_path)
        if lock:
            self.main_window.lock.release()

    def plot_all(self):
        self.plot_trace()
        self.plot_levels()
        self.plot_hist()
        self.plot_group_bounds()

    def start_resolve_thread(
        self, mode: str = "current", thread_finished=None, end_time_s=None
    ) -> None:
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

        mw = self.main_window
        if thread_finished is None:
            if mw.data_loaded:
                thread_finished = self.resolve_thread_complete
            else:
                thread_finished = mw.open_file_thread_complete

        _, conf = self.get_gui_confidence()
        data = mw.tree2dataset()
        currentparticle = mw.current_particle

        self.resolve_mode = mode
        if mode == "current":
            status_message = "Resolving current particle levels..."
            cpt_objs = [currentparticle.cpts]
        elif mode == "selected":
            status_message = "Resolving selected particle levels..."
            checked_parts = mw.get_checked_particles()
            cpt_objs = [part.cpts for part in checked_parts]
        elif mode == "all":
            status_message = "Resolving all particle levels..."
            cpt_objs = [part.cpts for part in data.particles]
        else:
            logger.error(msg="Provided mode not valid")
            raise TypeError

        all_sums = self.main_window.current_dataset.all_sums
        r_process_thread = ProcessThread()
        r_process_thread.add_tasks_from_methods(
            objects=cpt_objs, method_name="run_cpa", args=(all_sums, conf, True)
        )

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

    def gather_replace_results(
        self, results: Union[List[ProcessTaskResult], ProcessTaskResult]
    ):
        particles = self.main_window.current_dataset.particles
        part_uuids = [part.uuid for part in particles]
        if type(results) is not list:
            results = [results]
        result_part_uuids = [result.new_task_obj.uuid for result in results]
        try:
            for num, result in enumerate(results):
                result_part_ind = part_uuids.index(result_part_uuids[num])
                # target_particle = self.mainwindow.tree2particle(result_part_ind).cpts._particle
                target_particle = particles[result_part_ind]
                result.new_task_obj._particle = target_particle
                result.new_task_obj._cpa._particle = target_particle
                if result.new_task_obj.has_levels:
                    for level in result.new_task_obj.levels:
                        level._particle = target_particle
                        level.microtimes._particle = target_particle
                if result.new_task_obj._cpa.has_levels:
                    for level in result.new_task_obj._cpa.levels:
                        level._particle = target_particle
                        level.microtimes._particle = target_particle
                target_particle.cpts = result.new_task_obj
                # target_particle
            self.results_gathered = True
        except ValueError as e:
            logger.error(e)

    def resolve_thread_complete(self, thread: ProcessThread):
        count = 0

        # print("resolve complete")
        while self.results_gathered is False:
            time.sleep(1)
            count += 1
            if count >= 2:
                logger.error(msg="Results gathering timeout")
                break
                # raise RuntimeError

        self.main_window.current_dataset.has_levels = True
        if (
            self.main_window.treeViewParticles.currentIndex().data(Qt.UserRole)
            is not None
        ):
            self.main_window.display_data()
        self.main_window.remove_bursts(mode=self.resolve_mode)
        # self.mainwindow.chbEx_Levels.setEnabled(True)
        # self.mainwindow.rdbWith_Levels.setEnabled(True)
        self.main_window.set_startpoint()
        self.main_window.reset_gui()
        self.main_window.status_message("Done")
        logger.info("Resolving levels complete")

        self.results_gathered = False

    def get_gui_confidence(self):
        """Return current GUI value for confidence percentage."""

        return [
            self.main_window.cmbConfIndex.currentIndex(),
            self.confidence_index[self.main_window.cmbConfIndex.currentIndex()],
        ]

    def gui_quick_roi(self, mode: str):
        dialog = QuickROIDialog(mainwindow=self.main_window)
        dialog.exec_()
        # if dialog.should_trim_traces:
        #     if dialog.rdbCurrent.isChecked():
        #         particles = [self.mainwindow.current_particle]
        #     elif dialog.rdbSelected.isChecked():
        #         particles = self.mainwindow.get_checked_particles()
        #     elif dialog.rdbAll.isChecked():
        #         particles = self.mainwindow.current_dataset.particles
        #
        #     for particle in particles:
        #         if dialog.rdbManual.isChecked():
        #             trimmed = particle.trim_trace(min_level_int=dialog.spbManual_Min_Int.value(),
        #                                           min_level_dwell_time=dialog.dsbManual_Min_Time.value(),
        #                                           reset_roi=dialog.chbReset_ROI.isChecked())
        #             if trimmed is False and dialog.chbUncheck_If_Not_Valid.isChecked():
        #                 self.mainwindow.set_particle_check_state(particle.dataset_ind, False)
        #     self.mainwindow.lifetime_controller.test_need_roi_apply()
        #     self.plot_all()

    # def gui_reset_roi_current(self):
    #     self.reset_roi(mode='current')
    #
    # def gui_reset_roi_selected(self):
    #     self.reset_roi(mode='selected')
    #
    # def gui_reset_roi_all(self):
    #     self.reset_roi(mode='all')
    #
    # def reset_roi(self, mode=str):
    #     if mode == 'current':
    #         particles = [self.mainwindow.current_particle]
    #     elif mode == 'selected':
    #         particles = self.mainwindow.get_checked_particles()
    #     elif mode == 'all':
    #         particles = self.mainwindow.current_dataset.particles
    #     else:
    #         return
    #
    #     for particle in particles:
    #         particle.roi_region = (0, particle.abstimes[-1])
    #     self.plot_all()

    def any_int_plot_double_click(self, event: MouseClickEvent):
        if event.double():
            event.accept()
            cp = self.main_window.current_particle
            if cp.has_levels:
                use_groups = False
                use_global_groups = False
                select_groups = self.main_window.chbInt_Select_Groups.isChecked()
                current_tab = self.main_window.tabWidget.currentWidget().objectName()
                if current_tab == "tabIntensity":
                    use_groups = self.main_window.chbInt_Show_Groups.isChecked()
                    use_global_groups = (
                        self.main_window.chbInt_Show_Global_Groups.isChecked()
                    )
                elif current_tab == "tabGrouping":
                    use_groups = True
                    use_global_groups = (
                        self.main_window.chbInt_Show_Global_Groups.isChecked()
                    )
                elif current_tab == "tabLifetime":
                    use_groups = self.main_window.chbLifetime_Show_Groups.isChecked()
                    use_global_groups = (
                        self.main_window.chbInt_Show_Global_Groups.isChecked()
                    )

                if (
                    type(event.currentItem)
                    is pyqtgraph.graphicsItems.PlotCurveItem.PlotCurveItem
                    or type(event.currentItem) is pyqtgraph.LinearRegionItem
                ):
                    clicked_mapped_pos = event.currentItem.getViewBox().mapSceneToView(
                        event.scenePos()
                    )
                elif type(event.currentItem) is pyqtgraph.ViewBox:
                    clicked_mapped_pos = event.currentItem.mapSceneToView(
                        event.scenePos()
                    )
                else:
                    try:
                        clicked_mapped_pos = (
                            event.currentItem.getViewBox().mapSceneToView(
                                event.scenePos()
                            )
                        )
                    except AttributeError:
                        cp.level_or_group_selected = None
                        self.main_window.display_data()
                        return

                if select_groups and (
                    (use_groups and cp.has_groups)
                    or (use_global_groups and cp.global_particle.has_groups)
                ):
                    clicked_int = clicked_mapped_pos.y()
                    clicked_int = clicked_int * (
                        1000 / self.main_window.spbBinSize.value()
                    )
                    clicked_group = None
                    groups = cp.groups if use_groups else cp.global_particle.groups
                    # groups = groups[::-1]
                    group_bounds = (
                        cp.groups_bounds
                        if use_groups
                        else cp.global_particle.groups_bounds
                    )
                    group_bounds = group_bounds[::-1]
                    for group, (group_low, group_high) in zip(groups, group_bounds):
                        if group_low <= clicked_int <= group_high:
                            clicked_group = group
                            break
                    if clicked_group is not None:
                        cp.level_or_group_selected = clicked_group
                        self.main_window.display_data()
                else:
                    clicked_time = clicked_mapped_pos.x()
                    levels = (
                        cp.levels if not use_global_groups else cp.global_group_levels
                    )
                    level_times = [lvl.times_s for lvl in levels]
                    clicked_level = None
                    for level, (start, end) in zip(levels, level_times):
                        if start <= clicked_time <= end:
                            clicked_level = level
                            break
                    if clicked_level is not None:
                        cp.level_or_group_selected = clicked_level
                        self.main_window.display_data()

    def error(self, e):
        logger.error(e)


class LifetimeController(QObject):
    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.all_should_apply = None
        self.main_window = main_window

        self.lifetime_hist_widget = self.main_window.pgLifetime_Hist_PlotWidget
        self.life_hist_plot = self.main_window.pgLifetime_Hist_PlotWidget.getPlotItem()
        self.setup_widget(self.lifetime_hist_widget)

        self.residual_widget = self.main_window.pgLieftime_Residuals_PlotWidget
        self.residual_plot = (
            self.main_window.pgLieftime_Residuals_PlotWidget.getPlotItem()
        )
        self.setup_widget(self.residual_widget)
        self.residual_widget.hide()

        self.residual_plot.vb.setXLink(self.life_hist_plot.vb)

        self.setup_plot(self.life_hist_plot)
        self.setup_plot(self.residual_plot, is_residuals=True)

        self.fitparamdialog = FittingDialog(self.main_window, self)
        self.fitparam = FittingParameters(self)
        self.irf_loaded = False

        self.first = 0
        self.startpoint = None
        self.tmin = 0

        self.temp_fig = None
        self.temp_ax = None

        self.main_window.btnPrevLevel.clicked.connect(self.gui_prev_lev)
        self.main_window.btnNextLevel.clicked.connect(self.gui_next_lev)
        self.main_window.btnWholeTrace.clicked.connect(self.gui_whole_trace)
        self.main_window.chbLifetime_Show_Groups.stateChanged.connect(self.plot_all)
        self.main_window.chbShow_Residuals.stateChanged.connect(
            self.gui_show_hide_residuals
        )
        self.main_window.chbLifetime_Use_ROI.stateChanged.connect(
            self.gui_use_roi_changed
        )
        self.main_window.btnLifetime_Apply_ROI.clicked.connect(
            self.gui_apply_roi_current
        )
        self.main_window.btnLifetime_Apply_ROI_Selected.clicked.connect(
            self.gui_apply_roi_selected
        )
        self.main_window.btnLifetime_Apply_ROI_All.clicked.connect(
            self.gui_apply_roi_all
        )
        self.main_window.btnJumpToGroups.clicked.connect(self.gui_jump_to_groups)
        self.main_window.btnLoadIRF.clicked.connect(self.gui_load_irf)
        self.main_window.btnFitParameters.clicked.connect(self.gui_fit_param)
        self.main_window.btnFitCurrent.clicked.connect(self.gui_fit_current)
        self.main_window.btnFit.clicked.connect(self.gui_fit_levels)
        self.main_window.btnFitSelected.clicked.connect(self.gui_fit_selected)
        self.main_window.btnFitAll.clicked.connect(self.gui_fit_all)

    def setup_plot(self, plot: pg.PlotItem, is_residuals: bool = False):
        # Set axis label bold and size
        axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)
        plot.getAxis("left").setPen(axis_line_pen)
        plot.getAxis("bottom").setPen(axis_line_pen)
        plot.getAxis("left").label.font().setBold(True)
        plot.getAxis("bottom").label.font().setBold(True)
        plot.getAxis("left").label.font().setPointSize(16)
        plot.getAxis("bottom").label.font().setPointSize(16)

        # Setup axes and limits
        if not is_residuals:
            plot.getAxis("left").setLabel("Num. of occur.", "counts/bin")
            plot.getAxis("bottom").setLabel("Decay time", "ns")
            plot.getViewBox().setLimits(xMin=0, yMin=0)
        else:
            plot.getAxis("left").setLabel("Weighted residual", "au")
            plot.getAxis("bottom").setLabel("Time", "ns")
            plot.getViewBox().setLimits(xMin=0)

    @staticmethod
    def setup_widget(plot_widget: pg.PlotWidget):
        # Set widget background and antialiasing
        plot_widget.setBackground(background=None)
        plot_widget.setAntialiasing(True)

    def gui_prev_lev(self):
        """Moves to the previous resolves level and displays its decay curve."""
        cp = self.main_window.current_particle
        selected_level_or_group = cp.level_or_group_selected
        if selected_level_or_group is cp.groups[0]:
            cp.level_or_group_selected = cp.levels[-1]
        elif selected_level_or_group is cp.levels[0]:
            cp.level_or_group_selected = None
        elif selected_level_or_group is None:
            return
        elif type(selected_level_or_group) in [Level, GlobalLevel]:
            level_ind = np.argmax(
                [selected_level_or_group is level for level in cp.levels]
            )
            cp.level_or_group_selected = cp.levels[level_ind - 1]
        else:
            group_ind = np.argmax(
                [selected_level_or_group is group for group in cp.groups]
            )
            cp.level_or_group_selected = cp.groups[group_ind - 1]

        self.main_window.display_data()

    def gui_next_lev(self):
        """Moves to the next resolves level and displays its decay curve."""

        cp = self.main_window.current_particle
        selected_level_or_group = cp.level_or_group_selected
        if selected_level_or_group is None:
            cp.level_or_group_selected = cp.levels[0]
        elif selected_level_or_group is cp.levels[-1]:
            cp.level_or_group_selected = cp.groups[0]
        elif selected_level_or_group is cp.groups[-1]:
            return
        elif type(selected_level_or_group) in [Level, GlobalLevel]:
            level_ind = np.argmax(
                [selected_level_or_group is level for level in cp.levels]
            )
            cp.level_or_group_selected = cp.levels[level_ind + 1]
        else:
            group_ind = np.argmax(
                [selected_level_or_group is group for group in cp.groups]
            )
            cp.level_or_group_selected = cp.groups[group_ind + 1]

        self.main_window.display_data()

    def gui_whole_trace(self):
        "Unselects selected level and shows whole trace's decay curve"

        # self.mainwindow.current_level = None
        self.main_window.current_particle.level_or_group_selected = None
        self.main_window.display_data()

    def gui_jump_to_groups(self):
        cp = self.main_window.current_particle
        if cp.has_groups:
            cp.level_or_group_selected = cp.groups[0]
            self.main_window.display_data()

    def gui_show_hide_residuals(self):
        show = self.main_window.chbShow_Residuals.isChecked()

        if show:
            self.residual_widget.show()
        else:
            self.residual_widget.hide()

    def gui_load_irf(self):
        """Allow the user to load a IRF instead of the IRF that has already been loaded."""

        file_path = QFileDialog.getOpenFileName(
            self.main_window, "Open HDF5 file", "", "HDF5 files (*.h5)"
        )
        if file_path != ("", ""):  # fname will equal ('', '') if the user canceled.
            mw = self.main_window
            mw.status_message(message="Opening IRF file...")
            of_process_thread = ProcessThread(num_processes=1)
            of_process_thread.worker_signals.add_datasetindex.connect(mw.add_dataset)
            of_process_thread.worker_signals.add_particlenode.connect(mw.add_node)
            of_process_thread.worker_signals.add_all_particlenodes.connect(
                mw.add_all_nodes
            )
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
            of_process_thread.add_tasks_from_methods(of_obj, "open_irf")
            mw.threadpool.start(of_process_thread)
            mw.active_threads.append(of_process_thread)

    def add_irf(self, decay, t, irfdata):
        self.fitparam.irf = decay
        self.fitparam.irft = t
        # self.fitparam.irfdata = irfdata
        self.irf_loaded = True
        self.main_window.set_startpoint(irf_data=irfdata)
        self.main_window.dataset_node.dataobj.irf = decay
        self.main_window.dataset_node.dataobj.irf_t = t
        self.main_window.dataset_node.dataobj.has_irf = True
        self.fitparamdialog.updateplot()

    def gui_fit_param(self):
        """Opens a dialog to choose the setting with which the decay curve will be fitted."""

        if self.fitparamdialog.exec():
            self.fitparam.getfromdialog()
            if self.fitparam.fwhm is not None:
                self.irf_loaded = True
                self.main_window.reset_gui()

    def gui_fit_current(self):
        """Fits the currently selected level's decay curve using the provided settings."""

        cp = self.main_window.current_particle
        selected_level = cp.level_or_group_selected
        if selected_level is None:
            histogram = cp.histogram
        else:
            # level = self.mainwindow.current_level
            if selected_level <= cp.num_levels - 1:
                histogram = cp.cpts.levels[selected_level].histogram
            else:
                selected_group = selected_level - cp.num_levels
                histogram = cp.groups[selected_group].histogram
        try:
            channelwidth = self.main_window.current_particle.channelwidth
            f_p = self.fitparam
            shift = f_p.shift[:-1] / channelwidth
            shiftfix = f_p.shift[-1]
            shift = [*shift, shiftfix]
            if f_p.autostart != "Manual":
                start = None
            elif f_p.start is not None:
                start = int(f_p.start / channelwidth)
            else:
                start = None
            # print(f_p.autoend, f_p.end)
            if f_p.autoend:
                end = None
            elif f_p.end is not None:
                end = int(f_p.end / channelwidth)
            else:
                end = None
            boundaries = [start, end, f_p.autostart, f_p.autoend]
            if not histogram.fit_intensity_lifetime(
                f_p.numexp,
                f_p.tau,
                f_p.amp,
                shift,
                f_p.decaybg,
                f_p.irfbg,
                boundaries,
                f_p.addopt,
                f_p.irf,
                f_p.fwhm,
            ):
                return  # fit unsuccessful
            else:
                cp.has_fit_a_lifetime = True
        except AttributeError:
            logger.error("No decay")
        else:
            # self.mainwindow.display_data()
            self.fitting_thread_complete("current")

    def gui_fit_selected(self):
        """Fits the all the levels decay curves in the all the selected particles using the provided settings."""

        self.start_fitting_thread(mode="selected")

    def gui_fit_all(self):
        """Fits the all the levels decay curves in the all the particles using the provided settings."""

        self.start_fitting_thread(mode="all")

    def gui_fit_levels(self):
        """Fits the all the levels decay curves for the current particle."""

        self.start_fitting_thread()

    def gui_use_roi_changed(self):
        use_roi = self.main_window.chbLifetime_Use_ROI.isChecked()
        for particle in self.main_window.current_dataset.particles:
            particle.use_roi_for_histogram = use_roi
        if use_roi:
            self.test_need_roi_apply()
        else:
            self.update_apply_roi_button_colors()
        self.plot_all()

    def test_need_roi_apply(
        self, particle: Particle = None, update_buttons: bool = True
    ):
        if self.all_should_apply is None:
            self.all_should_apply = np.empty(self.main_window.current_dataset.num_parts)
            particle = None

        if particle is not None:
            particles_to_check = [particle]
        else:
            particles_to_check = self.main_window.current_dataset.particles
        for part in particles_to_check:
            part_ind = part.dataset_ind
            region_same = (
                part.roi_region[0:2] == part._histogram_roi.roi_region_used[0:2]
            )
            self.all_should_apply[part_ind] = not region_same

        if update_buttons:
            self.update_apply_roi_button_colors()

    def update_apply_roi_button_colors(self):
        use_roi_checked = self.main_window.chbLifetime_Use_ROI.isChecked()
        color_current = "None"
        color_selected = "None"
        color_all = "None"
        if use_roi_checked and self.all_should_apply is not None:
            if self.all_should_apply[self.main_window.current_particle.dataset_ind]:
                color_current = "red"
            if any(
                self.all_should_apply[
                    [
                        part.dataset_ind
                        for part in self.main_window.get_checked_particles()
                    ]
                ]
            ):
                color_selected = "red"
            if any(self.all_should_apply):
                color_all = "red"
        self.main_window.btnLifetime_Apply_ROI.setStyleSheet(
            f"background-color: {color_current}"
        )
        self.main_window.btnLifetime_Apply_ROI_Selected.setStyleSheet(
            f"background-color: {color_selected}"
        )
        self.main_window.btnLifetime_Apply_ROI_All.setStyleSheet(
            f"background-color: {color_all}"
        )

    def apply_roi(self, particles: list):
        for part in particles:
            if self.all_should_apply[part.dataset_ind]:
                if (
                    self.main_window.chbLifetime_Use_ROI.isChecked()
                    and not part.use_roi_for_histogram
                ):
                    part.use_roi_for_histogram = True
                part._histogram_roi.update_roi()
                self.all_should_apply[part.dataset_ind] = False
        self.test_need_roi_apply()
        self.plot_all()

    def gui_apply_roi_current(self):
        self.apply_roi(particles=[self.main_window.current_particle])

    def gui_apply_roi_selected(self):
        self.apply_roi(particles=self.main_window.get_checked_particles())

    def gui_apply_roi_all(self):
        self.apply_roi(particles=self.main_window.current_dataset.particles)

    def plot_all(self):
        self.main_window.display_data()

    def update_results(
        self,
        use_selected: bool = False,
        selected_level_or_group: Union[Level, GlobalLevel, Group] = None,
        particle: Particle = None,
        for_export: bool = False,
        str_return: bool = False,
    ) -> Union[str, None]:
        if use_selected:
            if selected_level_or_group is None:
                selected_level_or_group = (
                    self.main_window.current_particle.level_or_group_selected
                )
        else:
            selected_level_or_group = None
        if particle is None:
            particle = self.main_window.current_particle
        is_group = False
        is_level = False

        fit_name = f"{particle.name}"
        if selected_level_or_group is None:
            histogram = particle.histogram
            fit_name = fit_name + ", Whole Trace"
        else:
            histogram = selected_level_or_group.histogram
            if type(selected_level_or_group) in [Level, GlobalLevel]:
                level_ind = np.argmax(
                    [selected_level_or_group is level for level in particle.levels]
                )
                fit_name = fit_name + f", Level #{level_ind + 1}"
                is_level = True
            elif type(selected_level_or_group) is Group:
                group_ind = np.argmax(
                    [selected_level_or_group is group for group in particle.groups]
                )
                histogram = selected_level_or_group.histogram
                is_group = True
                fit_name = fit_name + f", Group #{group_ind + 1}"
            else:
                raise AssertionError(
                    "Provided `selected_level_or_group` is not a level or a group"
                )

        if not histogram.fitted:
            self.main_window.textBrowser.setText("")
            return

        info = ""
        if not for_export:
            info = fit_name + f"\n{len(fit_name) * '*'}\n"

        tau = histogram.tau
        amp = histogram.amp
        stds = histogram.stds
        avtau = np.dot(histogram.amp, histogram.tau)
        avtaustd = histogram.avtaustd
        if type(avtau) is list or type(avtau) is np.ndarray:
            avtau = avtau[0]
        if np.size(tau) == 1:
            info = info + f"Tau = {tau[0]:.3g} ± {stds[0]:.1g} ns"
            info = info + f"\nAmp = {amp[0]:.3g}"
        elif np.size(tau) == 2:
            info = info + f"Tau 1 = {tau[0]:.3g} ± {stds[0]:.1g} ns"
            info = info + f"\nTau 2 = {tau[1]:.3g} ± {stds[1]:.1g} ns"
            info = info + f"\nAmp 1 = {amp[0]:.3g} ± {stds[2]:.1g}"
            info = info + f"\nAmp 2 = {amp[1]:.3g} ± {stds[3]:.1g}"
        elif np.size(tau) == 3:
            info = info + f"Tau 1 = {tau[0]:.3g} ± {stds[0]:.1g} ns"
            info = info + f"\nTau 2 = {tau[1]:.3g} ± {stds[1]:.1g} ns"
            info = info + f"\nTau 3 = {tau[2]:.3g} ± {stds[2]:.1g} ns"
            info = info + f"\nAmp 1 = {amp[0]:.3g} ± {stds[3]:.1g}"
            info = info + f"\nAmp 2 = {amp[1]:.3g} ± {stds[4]:.1g}"
            info = info + f"\nAmp 3 = {amp[2]:.3g} ± {stds[5]:.1g}"
        info = info + f"\nAverage Tau = {avtau:.3g} ± {avtaustd:.1g} ns"

        info = (
            info
            + f"\n\nShift = {histogram.shift: .3g} ± {stds[2 * np.size(tau)]: .1g} ns"
        )
        if not for_export:
            info = info + f"\nDecay BG = {histogram.bg: .3g}"
            info = info + f"\nIRF BG = {histogram.irfbg: .3g}"
        if hasattr(histogram, "fwhm") and histogram.fwhm is not None:
            info = (
                info
                + f"\nSim. IRF FWHM = {histogram.fwhm: .3g} ± {stds[2 * np.size(tau) + 1]: .1g} ns"
            )

        info = info + f"\nChi-Sq = {histogram.chisq: .3g}"
        if not for_export:
            info = info + f"\n(0.8 <- 1 -> 1.3)"
        info = info + f"\nDurbin-Watson = {histogram.dw: .3g}"
        if not for_export:
            info = info + f"\n(DW (5%) > {histogram.dw_bound[0]: .4g})"
            info = info + f"\n(DW (1%) > {histogram.dw_bound[1]: .4g})"
            info = info + f"\n(DW (0.3%) > {histogram.dw_bound[2]: .4g})"
            info = info + f"\n(DW (0.1%) > {histogram.dw_bound[3]: .4g})"

        if is_group:
            group = selected_level_or_group
            info = info + f"\n\nTotal Dwell Time (s) = {group.dwell_time_s: .3g}"
            info = info + f"\n# of photons = {group.num_photons}"
            info = info + f"\n# used for fit = {group.histogram.num_photons_used}"
        elif is_level:
            level = selected_level_or_group
            info = info + f"\n\nDwell Time (s) {level.dwell_time_s: .3g}"
            info = info + f"\n# of photons = {level.num_photons}"
            info = info + f"\n# used for fit = {level.histogram.num_photons_used}"
        else:
            info = info + f"\n\nDwell Times (s) = {particle.dwell_time_s: .3g}"
            info = info + f"\n# of photons = {particle.num_photons}"
            info = info + f"\n# used for fit = {particle.histogram.num_photons_used}"

        if not for_export:
            self.main_window.textBrowser.setText(info)

        if str_return:
            return info

    def plot_decay_and_convd(
        self,
        particle: Particle,
        export_path: str,
        has_groups: bool,
        only_groups: bool = False,
        lock: bool = False,
    ):
        use_selected = True if particle.level_or_group_selected is None else False
        if not only_groups:
            for level in particle.levels:
                self.plot_decay(
                    selected_level_or_group=level,
                    use_selected=use_selected,
                    particle=particle,
                    remove_empty=False,
                    for_export=True,
                    export_path=None,
                )
                self.plot_convd(
                    selected_level_or_group=level,
                    use_selected=use_selected,
                    particle=particle,
                    remove_empty=False,
                    for_export=True,
                    export_path=export_path,
                )
        if has_groups:
            for group in particle.groups:
                self.plot_decay(
                    selected_level_or_group=group,
                    use_selected=use_selected,
                    particle=particle,
                    remove_empty=False,
                    for_export=True,
                    export_path=None,
                )
                self.plot_convd(
                    selected_level_or_group=group,
                    use_selected=use_selected,
                    particle=particle,
                    remove_empty=False,
                    for_export=True,
                    export_path=export_path,
                )
        if lock:
            self.main_window.lock.release()

    def plot_decay_convd_and_hist(
        self,
        particle: Particle,
        export_path: str,
        has_groups: bool,
        only_groups: bool = False,
        lock: bool = False,
    ):
        use_selected = True if particle.level_or_group_selected is None else False
        if not only_groups:
            for level in particle.levels:
                self.plot_decay(
                    selected_level_or_group=level,
                    use_selected=use_selected,
                    particle=particle,
                    remove_empty=False,
                    for_export=True,
                    export_path=None,
                )
                self.plot_convd(
                    selected_level_or_group=level,
                    use_selected=use_selected,
                    particle=particle,
                    remove_empty=False,
                    for_export=True,
                    export_path=None,
                )
                self.plot_residuals(
                    selected_level_or_group=level,
                    use_selected=use_selected,
                    particle=particle,
                    for_export=True,
                    export_path=export_path,
                )
        if has_groups:
            for group in particle.groups:
                self.plot_decay(
                    selected_level_or_group=group,
                    use_selected=use_selected,
                    particle=particle,
                    remove_empty=False,
                    for_export=True,
                    export_path=None,
                )
                self.plot_convd(
                    selected_level_or_group=group,
                    use_selected=use_selected,
                    particle=particle,
                    remove_empty=False,
                    for_export=True,
                    export_path=None,
                )
                self.plot_residuals(
                    selected_level_or_group=group,
                    use_selected=use_selected,
                    particle=particle,
                    for_export=True,
                    export_path=export_path,
                )
        if lock:
            self.main_window.lock.release()

    def plot_decay(
        self,
        selected_level_or_group: Union[None, Level, GlobalLevel, Group] = None,
        use_selected: bool = False,
        particle: Particle = None,
        remove_empty: bool = False,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
    ) -> None:
        """Used to display the histogram of the decay data of the current particle."""

        if type(export_path) is bool:
            lock = export_path
            export_path = None

        if use_selected:
            if selected_level_or_group is None:
                selected_level_or_group = (
                    self.main_window.current_particle.level_or_group_selected
                )
        else:
            selected_level_or_group = None
        if particle is None:
            particle = self.main_window.current_particle

        min_t = 0
        decay = None
        t = None
        if selected_level_or_group is None:
            if particle.histogram.fitted:
                decay = particle.histogram.fit_decay
                t = particle.histogram.convd_t
                min_t = particle.histogram.convd_t[0]
            else:
                try:
                    decay = particle.histogram.decay
                    t = particle.histogram.t
                except AttributeError:
                    logger.error("No Decay!")
                    return
        elif type(selected_level_or_group) in [Level, GlobalLevel]:
            if selected_level_or_group.histogram.fitted:
                decay = selected_level_or_group.histogram.fit_decay
                t = selected_level_or_group.histogram.convd_t
                min_t = t[0]
            else:
                try:
                    decay = selected_level_or_group.histogram.decay
                    t = selected_level_or_group.histogram.t
                except ValueError:
                    return
        elif type(selected_level_or_group) is Group:
            if selected_level_or_group.histogram.fitted:
                decay = selected_level_or_group.histogram.fit_decay
                t = selected_level_or_group.histogram.convd_t
                min_t = t[0]
            else:
                try:
                    decay = selected_level_or_group.histogram.decay
                    t = selected_level_or_group.histogram.t
                except ValueError:
                    return
        else:
            raise AttributeError(
                "Provided `selected_level_or_group` not a level or group"
            )

        try:
            decay.size
        except AttributeError as e:
            print(e)
        if decay.size == 0:
            return  # some levels have no photons

        cur_tab_name = self.main_window.tabWidget.currentWidget().objectName()
        if cur_tab_name == "tabLifetime" or for_export:
            if remove_empty:
                self.first = (decay > 4).argmax(axis=0)
                t = t[self.first : -1] - t[self.first]
                decay = decay[self.first : -1]
            else:
                self.first = 0
            unit = f"ns with {particle.channelwidth: .3g} ns bins"
            max_t = particle.histogram.t[-1]
            max_t_fitted = t[-1]

            if len(t) != len(decay):
                shortest = min([len(t), len(decay)])
                t = t[:shortest]
                decay = decay[:shortest]

            if not for_export:
                life_hist_plot = self.life_hist_plot
                life_hist_plot.clear()
                plot_pen = QPen()
                plot_pen.setWidthF(2)
                plot_pen.setJoinStyle(Qt.RoundJoin)
                plot_pen.setColor(QColor("blue"))
                plot_pen.setCosmetic(True)
                life_hist_plot.plot(x=t, y=decay, pen=plot_pen, symbol=None)

                life_hist_plot.getAxis("bottom").setLabel("Decay time", unit)
                life_hist_plot.getViewBox().setLimits(xMin=min_t, yMin=0, xMax=max_t)
                life_hist_plot.getViewBox().setRange(xRange=[min_t, max_t_fitted])
                self.fitparamdialog.updateplot()
            else:
                if self.temp_fig is None:
                    self.temp_fig = plt.figure()
                else:
                    self.temp_fig.clf()
                if self.main_window.rdbAnd_Residuals.isChecked():
                    self.temp_fig.set_size_inches(
                        EXPORT_MPL_WIDTH, 1.5 * EXPORT_MPL_HEIGHT
                    )
                    gs = self.temp_fig.add_gridspec(
                        5, 1, hspace=0, left=0.1, right=0.95
                    )
                    decay_ax = self.temp_fig.add_subplot(gs[0:-1, 0])
                    decay_ax.tick_params(direction="in", labelbottom=False)
                    residual_ax = self.temp_fig.add_subplot(gs[-1, 0])
                    residual_ax.spines["right"].set_visible(False)
                    self.temp_ax = {"decay_ax": decay_ax, "residual_ax": residual_ax}
                else:
                    self.temp_fig.set_size_inches(EXPORT_MPL_WIDTH, EXPORT_MPL_HEIGHT)
                    gs = self.temp_fig.add_gridspec(1, 1, left=0.05, right=0.95)
                    decay_ax = self.temp_fig.add_subplot(gs[0, 0])
                    self.temp_ax = {"decay_ax": decay_ax}

                decay_ax.spines["top"].set_visible(False)
                decay_ax.spines["right"].set_visible(False)
                decay_ax.semilogy(t, decay)

                min_pos_decay = decay[np.where(decay > 0, decay, np.inf).argmin()]
                min_pos_decay = max(
                    [min_pos_decay, 1e-5]
                )  # Min minimum positive decay set to be 1E-5
                max_decay = max(decay)
                if min_pos_decay >= max(decay):
                    max_decay = min_pos_decay * 2
                decay_ax.set(
                    xlabel=f"decay time ({unit})",
                    ylabel="counts",
                    xlim=[t[0], max_t],
                    ylim=[min_pos_decay, max_decay],
                )

            if for_export and export_path is not None:
                if not (os.path.exists(export_path) and os.path.isdir(export_path)):
                    raise AssertionError("Provided path not valid")
                pname = particle.unique_name
                logger.info(selected_level_or_group)
                if selected_level_or_group is None:
                    type_str = " hist (whole trace).png"
                    title_str = f"{pname} Decay Trace"
                elif type(selected_level_or_group) in [Level, GlobalLevel]:
                    level_ind = np.argmax(
                        [selected_level_or_group is level for level in particle.levels]
                    )
                    type_str = f" hist (level {level_ind + 1}).png"
                    title_str = f"{pname}, Level {level_ind + 1} Decay Trace"
                elif type(selected_level_or_group) is Group:
                    group_ind = np.argmax(
                        [selected_level_or_group is level for level in particle.groups]
                    )
                    type_str = f" hist (group {group_ind + 1}).png"
                    title_str = f"{pname}, Group {group_ind + 1} Decay Trace"
                else:
                    raise AttributeError(
                        "Provides `selected_level_or_group` is not a level or group"
                    )
                self.temp_fig.suptitle(title_str)
                full_path = os.path.join(export_path, pname + type_str)
                self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
                # sleep(1)
                # export_plot_item(plot_item=life_hist_plot, path=full_path)
        if lock:
            self.main_window.lock.release()

    def plot_convd(
        self,
        selected_level_or_group: Union[Level, GlobalLevel, Group] = None,
        use_selected: bool = False,
        particle: Particle = None,
        remove_empty: bool = False,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
    ) -> None:
        """Used to display the histogram of the decay data of the current particle."""

        if type(export_path) is bool:
            lock = export_path
            export_path = None

        if use_selected:
            if selected_level_or_group is None:
                selected_level_or_group = (
                    self.main_window.current_particle.level_or_group_selected
                )
        else:
            selected_level_or_group = None
        if particle is None:
            particle = self.main_window.current_particle

        group_ind = None
        convd = None
        t = None
        if selected_level_or_group is None:
            try:
                convd = particle.histogram.convd
                t = particle.histogram.convd_t
            except AttributeError:
                logger.error("No Decay!")
                return
        else:
            try:
                convd = selected_level_or_group.histogram.convd
                t = selected_level_or_group.histogram.convd_t
            except ValueError:
                return

        if convd is None or t is None:
            return

        # convd = convd / convd.max()

        cur_tab_name = self.main_window.tabWidget.currentWidget().objectName()
        decay_ax = None
        if cur_tab_name == "tabLifetime" or for_export:
            if not for_export:
                plot_pen = QPen()
                plot_pen.setWidthF(1)
                plot_pen.setJoinStyle(Qt.RoundJoin)
                plot_pen.setColor(QColor("red"))
                plot_pen.setCosmetic(True)

                self.life_hist_plot.plot(x=t, y=convd, pen=plot_pen, symbol=None)
                unit = f"ns with {particle.channelwidth: .3g} ns bins"
                self.life_hist_plot.getAxis("bottom").setLabel("Decay time", unit)
                # self.life_hist_plot.getViewBox().setXRange(min=t[0], max=t[-1], padding=0)
                # self.life_hist_plot.getViewBox().setLimits(xMin=0, yMin=0, xMax=t[-1])
            else:
                decay_ax = self.temp_ax["decay_ax"]
                decay_ax.semilogy(t, convd)
                _, max_y = decay_ax.get_ylim()
                min_y = min(convd)
                if min_y <= 0:
                    min_y = 1e-1
                if not min_y < max_y:
                    max_y = min_y + 10
                decay_ax.set_ylim(min_y, max_y)

            if for_export and export_path is not None:
                # plot_item = self.life_hist_plot
                if selected_level_or_group is None:
                    type_str = f"{particle.unique_name} hist-fitted (whole trace).png"
                    title_str = f"{particle.unique_name} Decay Trace and Fit"
                elif type(selected_level_or_group) in [Level, GlobalLevel]:
                    level_ind = np.argmax(
                        [selected_level_or_group is level for level in particle.levels]
                    )
                    type_str = f"{particle.unique_name} hist-fitted (level {selected_level_or_group + 1}).png"
                    title_str = f"{particle.unique_name}, Level {selected_level_or_group + 1} Decay Trace and Fit"
                elif type(selected_level_or_group) is Group:
                    group_ind = np.argmax(
                        [selected_level_or_group is level for level in particle.groups]
                    )
                    type_str = f"{particle.unique_name} hist-fitted (group {group_ind + 1}).png"
                    title_str = f"{particle.unique_name}, Group {group_ind + 1} Decay Trace and Fit"
                else:
                    raise AttributeError(
                        "Provided `selected_level_or_group` is not a level or a group"
                    )
                full_path = os.path.join(export_path, type_str)
                text_select_ind = selected_level_or_group
                if text_select_ind is None:
                    text_select_ind = -1
                text_str = self.update_results(
                    selected_level_or_group=text_select_ind,
                    particle=particle,
                    for_export=True,
                    str_return=True,
                )
                decay_ax.text(
                    0.8, 0.9, text_str, fontsize=6, transform=decay_ax.transAxes
                )
                self.temp_fig.suptitle(title_str)
                if EXPORT_MPL_DPI > 50:
                    export_dpi = 50
                else:
                    export_dpi = EXPORT_MPL_DPI
                self.temp_fig.savefig(full_path, dpi=export_dpi)
                # export_plot_item(plot_item=plot_item, path=full_path, text=text_str)
        if lock:
            self.main_window.lock.release()

    def plot_residuals(
        self,
        selected_level_or_group: Union[Level, GlobalLevel, Group] = None,
        use_selected: bool = False,
        particle: Particle = None,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
    ) -> None:
        """Used to display the histogram of the decay data of the current particle."""

        if type(export_path) is bool:
            lock = export_path
            export_path = None
        if selected_level_or_group is None:
            selected_level_or_group = (
                self.main_window.current_particle.level_or_group_selected
            )
        if particle is None:
            particle = self.main_window.current_particle

        group_ind = None
        residuals = None
        t = None
        if selected_level_or_group is None:
            selected_level_or_group = particle.level_or_group_selected
        if use_selected and selected_level_or_group is not None:
            try:
                residuals = selected_level_or_group.histogram.residuals
                t = selected_level_or_group.histogram.convd_t
            except ValueError:
                return
        else:
            selected_level_or_group = None
            residuals = particle.histogram.residuals
            t = particle.histogram.convd_t

        cur_tab_name = self.main_window.tabWidget.currentWidget().objectName()
        if cur_tab_name == "tabLifetime" or for_export:
            unit = f"ns with {particle.channelwidth: .3g} ns bins"
            if not for_export:
                if residuals is None or t is None:
                    self.residual_plot.clear()
                    return
                self.residual_plot.clear()
                scat_plot = pg.ScatterPlotItem(
                    x=t, y=residuals, symbol="o", size=3, pen="#0000CC", brush="#0000CC"
                )
                self.residual_plot.addItem(scat_plot)
                self.residual_plot.getAxis("bottom").setLabel("Decay time", unit)
                self.residual_plot.getViewBox().setXRange(
                    min=t[0], max=t[-1], padding=0
                )
                self.residual_plot.getViewBox().setYRange(
                    min=residuals.min(), max=residuals.max(), padding=0
                )
                self.residual_plot.getViewBox().setLimits(xMin=t[0], xMax=t[-1])
            else:
                residual_ax = self.temp_ax["residual_ax"]
                residual_ax.scatter(t, residuals, s=1)
                min_x, max_x = self.temp_ax["decay_ax"].get_xlim()
                residual_ax.set(xlim=[min_x, max_x], xlabel=f"decay time ({unit})")

            if for_export and export_path is not None:
                if selected_level_or_group is None:
                    type_str = " residuals (whole trace).png"
                    title_str = f"{particle.unique_name} Decay Trace, Fit and Residuals"
                elif type(selected_level_or_group) in [Level, GlobalLevel]:
                    level_ind = np.argmax(
                        [selected_level_or_group is level for level in particle.levels]
                    )
                    type_str = f" residuals (level {level_ind + 1} with residuals).png"
                    title_str = (
                        f"{particle.unique_name},"
                        f" Level {level_ind + 1} Decay Trace, Fit and Residuals"
                    )
                elif type(selected_level_or_group) is Group:
                    group_ind = np.argmax(
                        [selected_level_or_group is level for level in particle.groups]
                    )
                    type_str = f" residuals (group {group_ind + 1} with residuals).png"
                    title_str = (
                        f"{particle.unique_name}, Group {group_ind + 1}"
                        f" Decay Trace, Fit and Residuals"
                    )
                else:
                    raise AssertionError(
                        "Provided `selected_level_or_group` is not a level or a group"
                    )
                text_str = self.update_results(
                    selected_level_or_group=selected_level_or_group,
                    particle=particle,
                    for_export=True,
                    str_return=True,
                )
                decay_ax = self.temp_ax["decay_ax"]
                decay_ax.text(
                    0.9, 0.9, text_str, fontsize=6, transform=decay_ax.transAxes
                )
                full_path = os.path.join(export_path, particle.unique_name + type_str)
                self.temp_fig.suptitle(title_str)
                self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
        if lock:
            self.main_window.lock.release()

    def start_fitting_thread(self, mode: str = "current") -> None:
        """
        Creates a worker to resolve levels.ckibacxxx

        Depending on the ``current_selected_all`` parameter the worker will be
        given the necessary parameter to fit the current, selected or all particles.

        Parameters
        ----------
        mode : {'current', 'selected', 'all'}
            Possible values are 'current' (default), 'selected', and 'all'.
        """

        assert mode in [
            "current",
            "selected",
            "all",
        ], "'resolve_all' and 'resolve_selected' can not both be given as parameters."

        mw = self.main_window
        if mode == "current":
            status_message = "Fitting Levels for Current Particle..."
            particles = [mw.current_particle]
        elif mode == "selected":
            status_message = "Fitting Levels for Selected Particles..."
            particles = mw.get_checked_particles()
        elif mode == "all":
            status_message = "Fitting Levels for All Particles..."
            particles = mw.current_dataset.particles

        f_p = self.fitparam
        channelwidth = particles[0].channelwidth

        if f_p.autostart != "Manual":
            start = None
        elif f_p.start is not None:
            start = int(f_p.start / channelwidth)
        else:
            start = None
        if f_p.autoend:
            end = None
        elif f_p.end is not None:
            end = int(f_p.end / channelwidth)
        else:
            end = None

        part_hists = list()
        for part in particles:
            part_hists.append(ParticleAllHists(particle=part))

        clean_fit_param = copy(self.fitparam)
        clean_fit_param.parent = None
        clean_fit_param.fpd = None

        f_process_thread = ProcessThread()
        f_process_thread.add_tasks_from_methods(
            objects=part_hists,
            method_name="fit_part_and_levels",
            args=(channelwidth, start, end, clean_fit_param),
        )
        f_process_thread.signals.start_progress.connect(mw.start_progress)
        f_process_thread.signals.status_update.connect(mw.status_message)
        f_process_thread.signals.step_progress.connect(mw.update_progress)
        f_process_thread.signals.end_progress.connect(mw.end_progress)
        f_process_thread.signals.error.connect(self.error)
        f_process_thread.signals.results.connect(self.gather_replace_results)
        f_process_thread.signals.finished.connect(self.fitting_thread_complete)
        f_process_thread.worker_signals.reset_gui.connect(mw.reset_gui)
        f_process_thread.status_message = status_message

        mw.threadpool.start(f_process_thread)
        mw.active_threads.append(f_process_thread)

    def gather_replace_results(
        self, results: Union[List[ProcessTaskResult], ProcessTaskResult]
    ):
        particles = self.main_window.current_dataset.particles
        part_uuids = [part.uuid for part in particles]
        if type(results) is not list:
            results = [results]
        result_part_uuids = [result.new_task_obj.part_uuid for result in results]
        try:
            for num, result in enumerate(results):
                any_successful_fit = None
                result_part_ind = part_uuids.index(result_part_uuids[num])
                target_particle = self.main_window.current_dataset.particles[
                    result_part_ind
                ]

                target_hist = target_particle.histogram
                target_microtimes = target_hist.microtimes

                result.new_task_obj.part_hist._particle = target_particle
                result.new_task_obj.part_hist.microtimes = target_microtimes

                if not result.new_task_obj.part_hist.is_for_roi:
                    target_particle._histogram = result.new_task_obj.part_hist
                else:
                    target_particle._histogram_roi = result.new_task_obj.part_hist
                any_successful_fit = [result.new_task_obj.part_hist.fitted]

                if result.new_task_obj.has_level_hists:
                    for i, res_hist in enumerate(result.new_task_obj.level_hists):
                        target_level = target_particle.cpts.levels[i]
                        target_level_microtimes = target_level.microtimes

                        res_hist._particle = target_particle
                        res_hist.microtimes = target_level_microtimes
                        res_hist.level = target_level

                        target_level.histogram = res_hist
                        any_successful_fit.append(res_hist.fitted)

                if result.new_task_obj.has_group_hists:
                    for i, res_group_hist in enumerate(result.new_task_obj.group_hists):
                        target_group_lvls_inds = target_particle.groups[i].lvls_inds
                        target_g_lvls_microtimes = np.array([])
                        for lvls_ind in target_group_lvls_inds:
                            m_times = target_particle.cpts.levels[lvls_ind].microtimes
                            target_g_lvls_microtimes = np.append(
                                target_g_lvls_microtimes, m_times
                            )

                        res_group_hist._particle = target_particle
                        res_group_hist.microtimes = target_g_lvls_microtimes
                        res_group_hist.level = target_group_lvls_inds

                        target_particle.groups[i].histogram = res_group_hist
                        any_successful_fit.append(res_group_hist.fitted)

                    group_hists = [g.histogram for g in target_particle.groups]
                    for g_l in target_particle.ahca.selected_step.group_levels:
                        g_l.histogram = group_hists[g_l.group_ind]

                if any(any_successful_fit):
                    target_particle.has_fit_a_lifetime = True

        except ValueError as e:
            logger.error(e)

    def fitting_thread_complete(self, mode: str = None):
        if self.main_window.current_particle is not None:
            self.main_window.display_data()
        # self.mainwindow.chbEx_Lifetimes.setEnabled(False)
        # self.mainwindow.chbEx_Lifetimes.setEnabled(True)
        # self.mainwindow.chbEx_Hist.setEnabled(True)
        # self.mainwindow.rdbWith_Fit.setEnabled(True)
        # self.mainwindow.rdbAnd_Residuals.setEnabled(True)
        self.main_window.chbShow_Residuals.setChecked(True)
        if not mode == "current":
            self.main_window.status_message("Done")
        self.main_window.current_dataset.has_lifetimes = True
        logger.info("Fitting levels complete")

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
        self.main_window.chbShow_Residuals.setChecked(show)
        if lock:
            self.main_window.lock.release()

    def error(self, e):
        logger.error(e)


class GroupingController(QObject):
    def __init__(self, main_widow: MainWindow):
        super().__init__()
        self.main_window = main_widow

        # self.groups_hist_widget = groups_hist_widget
        # self.groups_hist_plot = groups_hist_widget.addPlot()
        # self.groups_hist_widget.setBackground(background=None)

        self.bic_plot_widget = self.main_window.pgGroups_BIC_PlotWidget
        self.bic_scatter_plot = self.bic_plot_widget.getPlotItem()
        self.bic_plot_widget.setBackground(background=None)

        # Set axis label bold and size
        axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)

        self.bic_scatter_plot.getAxis("left").setPen(axis_line_pen)
        self.bic_scatter_plot.getAxis("bottom").setPen(axis_line_pen)
        self.bic_scatter_plot.getAxis("left").label.font().setBold(True)
        self.bic_scatter_plot.getAxis("bottom").label.font().setBold(True)
        self.bic_scatter_plot.getAxis("left").label.font().setPointSize(12)
        self.bic_scatter_plot.getAxis("bottom").label.font().setPointSize(12)

        self.bic_scatter_plot.getAxis("left").setLabel("BIC")
        self.bic_scatter_plot.getAxis("bottom").setLabel("Number of Groups")
        self.bic_scatter_plot.getViewBox().setLimits(xMin=0)

        self.all_bic_plots = None
        self.all_last_solutions = None

        self.temp_dir = None
        self.temp_fig = None
        self.temp_ax = None

        self.main_window.btnGroupCurrent.clicked.connect(self.gui_group_current)
        self.main_window.btnGroupSelected.clicked.connect(self.gui_group_selected)
        self.main_window.btnGroupAll.clicked.connect(self.gui_group_all)
        self.main_window.btnApplyGroupsCurrent.clicked.connect(
            self.gui_apply_groups_current
        )
        self.main_window.btnApplyGroupsSelected.clicked.connect(
            self.gui_apply_groups_selected
        )
        self.main_window.btnApplyGroupsAll.clicked.connect(self.gui_apply_groups_all)
        self.main_window.btnGroupGlobal.clicked.connect(self.gui_group_global)

    def clear_bic(self):
        self.bic_scatter_plot.clear()

    def solution_clicked(self, plot, points):
        curr_part = self.main_window.current_particle
        last_solution = self.all_last_solutions[curr_part.dataset_ind]
        if last_solution != points[0]:
            curr_part = self.main_window.current_particle
            point_num_groups = int(points[0].pos()[0])
            new_ind = curr_part.ahca.steps_num_groups.index(point_num_groups)
            curr_part.ahca.set_selected_step(new_ind)
            curr_part.using_group_levels = False
            curr_part.level_or_group_selected = None
            if last_solution:
                last_solution.setPen(pg.mkPen(width=1, color="k"))
            for p in points:
                p.setPen("r", width=2)
            last_solution = points[0]
            self.all_last_solutions[curr_part.dataset_ind] = last_solution

            if curr_part.using_group_levels:
                curr_part.using_group_levels = False
            self.main_window.display_data()

    def plot_group_bic(
        self,
        particle: Particle = None,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
        is_global_group=False,
    ):
        if type(export_path) is bool:
            lock = export_path
            export_path = None

        self.clear_bic()
        if particle is None:
            if is_global_group:
                particle = self.main_window.current_dataset.global_particle
            else:
                particle = self.main_window.current_particle
                if (
                    self.main_window.tabGroupingMode.currentWidget().objectName()
                    == "tabGlobal"
                ):
                    if particle.has_global_grouping:
                        particle = self.main_window.current_dataset.global_particle
                        is_global_group = True
                    else:
                        return

        if not particle.has_groups:
            return

        if particle.ahca.best_step.single_level:
            self.bic_plot_widget.getPlotItem().clear()
            return
        try:
            grouping_bics = particle.grouping_bics.copy()
            grouping_selected_ind = particle.grouping_selected_ind
            best_grouping_ind = particle.best_grouping_ind
            grouping_num_groups = particle.grouping_num_groups.copy()
        except AttributeError:
            logger.error("No groups!")
            return

        cur_tab_name = "tabGrouping"
        if cur_tab_name == "tabGrouping":  # or for_export:
            num_parts = self.main_window.tree2dataset().num_parts
            if self.all_bic_plots is None and self.all_last_solutions is None:
                self.all_bic_plots = [None] * (num_parts + 1)
                self.all_last_solutions = [None] * (num_parts + 1)

            dataset_ind = particle.dataset_ind if not is_global_group else num_parts
            if particle.ahca.plots_need_to_be_updated:
                self.all_bic_plots[dataset_ind] = None
                self.all_last_solutions[dataset_ind] = None
            scat_plot_item = self.all_bic_plots[dataset_ind]
            if scat_plot_item is None:
                spot_other_pen = pg.mkPen(width=1, color="k")
                spot_selected_pen = pg.mkPen(width=2, color="r")
                spot_other_brush = pg.mkBrush(color="k")
                spot_best_brush = pg.mkBrush(color="g")

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

                    bic_spots.append(
                        {
                            "pos": (grouping_num_groups[i], g_bic),
                            "size": 10,
                            "pen": spot_pen,
                            "brush": spot_brush,
                        }
                    )

                scat_plot_item.addPoints(bic_spots)

                self.all_bic_plots[dataset_ind] = scat_plot_item
                best_solution = scat_plot_item.points()[best_grouping_ind]
                self.all_last_solutions[dataset_ind] = best_solution
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
                self.temp_ax.set(xlabel="Solution's Number of Groups", ylabel="BIC")
                # self.temp_ax.tick_params(labeltop=False, labelright=False)
                self.temp_ax.spines["right"].set_visible(False)
                self.temp_ax.spines["top"].set_visible(False)
                self.temp_ax.tick_params(axis="x", top=False)
                self.temp_ax.tick_params(axis="y", right=False)

                norm_points = [
                    [grouping_num_groups[i], bic]
                    for i, bic in enumerate(grouping_bics)
                    if (
                        i != best_grouping_ind
                        or i != grouping_selected_ind
                        or grouping_num_groups[i] == 1
                    )
                ]
                norm_points = np.array(norm_points)

                marker_size = 70
                marker_line_width = 2
                norm_color = "lightgrey"
                norm_line_color = "k"
                best_color = "lightgreen"
                selected_line_color = "r"

                self.temp_ax.scatter(
                    norm_points[:, 0],
                    norm_points[:, 1],
                    s=marker_size,
                    color=norm_color,
                    linewidths=marker_line_width,
                    edgecolors=norm_line_color,
                    label="Solutions",
                )

                if grouping_selected_ind == best_grouping_ind:
                    num_groups = grouping_num_groups[best_grouping_ind]
                    bic_value = grouping_bics[best_grouping_ind]
                    self.temp_ax.scatter(
                        num_groups,
                        bic_value,
                        s=marker_size,
                        color=best_color,
                        linewidths=marker_line_width,
                        edgecolors=selected_line_color,
                        label="Best Solution Selected",
                    )
                else:
                    selected_num_groups = grouping_num_groups[grouping_selected_ind]
                    selected_bic_value = grouping_bics[grouping_selected_ind]
                    self.temp_ax.scatter(
                        selected_num_groups,
                        selected_bic_value,
                        s=marker_size,
                        color=norm_color,
                        linewidths=marker_line_width,
                        edgecolors=selected_line_color,
                        label="Solution Selected",
                    )

                    best_num_groups = grouping_num_groups[best_grouping_ind]
                    best_bic_value = grouping_bics[best_grouping_ind]
                    self.temp_ax.scatter(
                        best_num_groups,
                        best_bic_value,
                        s=marker_size,
                        color=best_color,
                        linewidths=marker_line_width,
                        edgecolors=norm_line_color,
                        label="Best Solution",
                    )

                self.temp_ax.set_title(f"{particle.name} Grouping Steps")
                self.temp_ax.legend(frameon=False, loc="lower right")

            if for_export and export_path is not None:
                if not (os.path.exists(export_path) and os.path.isdir(export_path)):
                    raise AssertionError("Provided path not valid")
                full_path = os.path.join(export_path, particle.name + " BIC.png")
                self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
                sleep(1)
                # export_plot_item(plot_item=self.bic_scatter_plot, path=full_path)
        if lock:
            self.main_window.lock.release()

    def gui_group_current(self):
        self.start_grouping_thread(mode="current")

    def gui_group_selected(self):
        self.start_grouping_thread(mode="selected")

    def gui_group_all(self):
        self.start_grouping_thread(mode="all")

    def gui_apply_groups_current(self):
        self.apply_groups()

    def gui_apply_groups_selected(self):
        self.apply_groups("selected")

    def gui_apply_groups_all(self):
        self.apply_groups("all")

    def start_grouping_thread(self, mode: str = "current") -> None:
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

        mw = self.main_window

        if mode == "current":
            grouping_objs = [mw.current_particle.ahca]
            status_message = "Grouping levels for current particle..."
        elif mode == "selected":
            checked_particles = mw.get_checked_particles()
            grouping_objs = [particle.ahca for particle in checked_particles]
            status_message = "Grouping levels for selected particle..."
        elif mode == "all":
            all_particles = mw.current_dataset.particles
            grouping_objs = [particle.ahca for particle in all_particles]
            # print(grouping_objs)
            status_message = "Grouping levels for all particle..."

        # g_process_thread = ProcessThread(num_processes=1, task_buffer_size=1)

        self.temp_dir = tempfile.TemporaryDirectory(prefix="Full_SMS_Grouping")
        g_process_thread = ProcessThread(temp_dir=self.temp_dir)
        g_process_thread.add_tasks_from_methods(
            objects=grouping_objs, method_name="run_grouping"
        )

        g_process_thread.signals.status_update.connect(mw.status_message)
        g_process_thread.signals.start_progress.connect(mw.start_progress)
        g_process_thread.signals.step_progress.connect(mw.update_progress)
        g_process_thread.signals.end_progress.connect(mw.end_progress)
        g_process_thread.signals.error.connect(self.error)
        g_process_thread.signals.results.connect(self.gather_replace_results)
        g_process_thread.signals.finished.connect(self.grouping_thread_complete)
        g_process_thread.worker_signals.reset_gui.connect(mw.reset_gui)
        g_process_thread.status_message = status_message

        self.main_window.threadpool.start(g_process_thread)

    def gather_replace_results(
        self, results: Union[List[ProcessTaskResult], ProcessTaskResult]
    ):
        particles = self.main_window.current_dataset.particles
        part_uuids = [part.uuid for part in particles]
        if type(results) is not list:
            results = [results]
        result_part_uuids = [result.new_task_obj.uuid for result in results]
        try:
            for num, result in enumerate(results):
                result_part_ind = part_uuids.index(result_part_uuids[num])
                new_part = self.main_window.current_dataset.particles[result_part_ind]
                new_part.level_or_group_selected = None

                result_ahca = None
                if new_part.has_levels:
                    result_ahca = result.new_task_obj
                    result_ahca._particle = new_part
                    result_ahca.best_step._particle = new_part
                    for step in result_ahca.steps:
                        step._particle = new_part
                        for group_attr_name in [
                            "_ahc_groups",
                            "groups",
                            "_seed_groups",
                            "group_levels",
                        ]:
                            if hasattr(step, group_attr_name):
                                group_attr = getattr(step, group_attr_name)
                                for group in group_attr:
                                    lvls = new_part
                                    if group_attr_name == "group_levels":
                                        lvls = group_attr
                                    else:
                                        if hasattr(group, "lvls"):
                                            lvls = group.lvls
                                    if lvls is not None:
                                        for ahc_lvl in lvls:
                                            if hasattr(ahc_lvl, "_particle"):
                                                ahc_lvl._particle = new_part
                                            if hasattr(ahc_lvl.microtimes, "_particle"):
                                                ahc_lvl.microtimes._particle = new_part
                                            ahc_hist = ahc_lvl.histogram
                                            if hasattr(ahc_hist, "_particle"):
                                                ahc_hist._particle = new_part
                new_part.ahca = result_ahca

                if new_part.has_groups:
                    new_part.makegrouphists()
                    new_part.makegrouplevelhists()
        except ValueError as e:
            logger.error(e)

    def grouping_thread_complete(self, mode):
        # results = list()
        # for result_file in os.listdir(self.temp_dir.name):
        #     with open(os.path.join(self.temp_dir.name, result_file), 'rb') as f:
        #         results.append(pickle.load(f))
        # self.temp_dir.cleanup()
        # self.temp_dir = None
        # self.gather_replace_results(results=results)
        if self.main_window.chbGroup_Auto_Apply.isChecked():
            self.apply_groups(particles=self.main_window.current_dataset.particles)
        self.main_window.current_dataset.has_groups = True
        if self.main_window.current_particle is not None:
            self.main_window.display_data()
        self.main_window.status_message("Done")
        self.main_window.reset_gui()
        self.check_rois_and_set_label()
        logger.info("Grouping levels complete")

    def check_rois_and_set_label(self):
        export_group_roi_label = ""
        label_color = "black"
        all_has_groups = np.array(
            [p.has_groups for p in self.main_window.current_dataset.particles]
        )
        if any(all_has_groups):
            all_grouped_with_roi = np.array(
                [p.grouped_with_roi for p in self.main_window.current_dataset.particles]
            )
            all_grouped_and_with_roi = all_grouped_with_roi[all_has_groups]
            if all(all_grouped_and_with_roi):
                export_group_roi_label = "All have ROI\n"
            elif any(all_grouped_and_with_roi):
                export_group_roi_label = "Some have ROI\n"
                label_color = "red"

            checked_particles = self.main_window.get_checked_particles()
            if len(checked_particles) > 0:
                all_checked_has_groups = np.array(
                    [p.has_groups for p in checked_particles]
                )
                all_checked_grouped_with_roi = np.array(
                    [p.grouped_with_roi for p in checked_particles]
                )
                all_checked_grouped_and_with_roi = all_checked_grouped_with_roi[
                    all_checked_has_groups
                ]
                if all(all_checked_grouped_and_with_roi):
                    export_group_roi_label += "All selected have ROI\n"
                elif any(all_checked_grouped_and_with_roi):
                    export_group_roi_label += "Some selected have ROI\n"
                else:
                    export_group_roi_label += "None selected have ROI\n"

            if self.main_window.current_particle.has_groups:
                if self.main_window.current_particle.grouped_with_roi:
                    export_group_roi_label += "Current has ROI"
                else:
                    export_group_roi_label += "Current doesn't have ROI"

        if export_group_roi_label == "":
            self.main_window.lblGrouping_ROI.setVisible(False)
        else:
            if export_group_roi_label[-1] == "\n":
                export_group_roi_label = export_group_roi_label[:-1]
            self.main_window.lblGrouping_ROI.setVisible(True)
            self.main_window.lblGrouping_ROI.setText(export_group_roi_label)
            self.main_window.lblGrouping_ROI.setStyleSheet(f"color: {label_color}")

    def apply_groups(self, mode: str = "current", particles=None):
        if particles is None:
            if mode == "current":
                particles = [self.main_window.current_particle]
            elif mode == "selected":
                particles = self.main_window.get_checked_particles()
            else:
                particles = self.main_window.current_dataset.particles

        bool_use = not all([part.using_group_levels for part in particles])
        for particle in particles:
            particle.using_group_levels = bool_use
            particle.level_or_group_selected = None

        self.main_window.intensity_controller.plot_all()

    def build_global(self):
        if self.main_window.cmbGlobalParticleSelection.currentText() == "Selected":
            particles = self.main_window.get_checked_particles()
        else:
            particles = self.main_window.current_dataset.particles

        use_roi = self.main_window.chbGroup_Use_ROI.isChecked()
        self.main_window.current_dataset.global_particle = GlobalParticle(
            particles=particles, use_roi=use_roi
        )
        self.main_window.status_message("Global particle built...")

    def global_gather_replace_results(self, result: ProcessTaskResult):
        new_global_particle = self.main_window.current_dataset.global_particle
        try:
            result_ahca = None
            if new_global_particle.has_levels:
                result_ahca = result.new_task_obj
                result_ahca._particle = new_global_particle
                result_ahca.best_step._particle = new_global_particle
                for step in result_ahca.steps:
                    step._particle = new_global_particle
                    for group_attr_name in [
                        "_ahc_groups",
                        "groups",
                        "_seed_groups",
                        "group_levels",
                    ]:
                        if hasattr(step, group_attr_name):
                            group_attr = getattr(step, group_attr_name)
                            for group in group_attr:
                                lvls = new_global_particle
                                if group_attr_name == "group_levels":
                                    lvls = group_attr
                                else:
                                    if hasattr(group, "lvls"):
                                        lvls = group.lvls
                                if lvls is not None:
                                    for ahc_lvl in lvls:
                                        if hasattr(ahc_lvl, "_particle"):
                                            ahc_lvl._particle = new_global_particle
                                        # if hasattr(ahc_lvl.microtimes, '_particle'):
                                        #     ahc_lvl.microtimes._particle = new_global_particle
                                        # ahc_hist = ahc_lvl.histogram
                                        # if hasattr(ahc_hist, '_particle'):
                                        #     ahc_hist._particle = new_global_particle
            new_global_particle.ahca = result_ahca
            self.global_results_gathered = True
        except ValueError as e:
            logger.error(e)

    def global_grouping_thread_complete(self):
        # if self.main_window.chbGroup_Auto_Apply.isChecked():
        #     self.apply_groups(particles=self.main_window.current_dataset.particles)
        # self.main_window.current_dataset.has_groups = True
        self.main_window.display_data(is_global_group=True)
        self.main_window.status_message("Done")
        self.main_window.reset_gui()
        # self.check_rois_and_set_label()
        logger.info("Global Grouping has been completed.")

    def start_global_grouping_thread(self):
        mw = self.main_window
        global_ahca = mw.current_dataset.global_particle.ahca

        gg_process_thread = ProcessThread(num_processes=1)
        gg_process_thread.add_tasks_from_methods(
            objects=global_ahca, method_name="run_grouping"
        )

        gg_process_thread.signals.status_update.connect(mw.status_message)
        gg_process_thread.signals.start_progress.connect(mw.start_progress)
        gg_process_thread.signals.step_progress.connect(mw.update_progress)
        gg_process_thread.signals.set_progress.connect(mw.set_progress)
        gg_process_thread.signals.end_progress.connect(mw.end_progress)
        gg_process_thread.signals.error.connect(self.error)
        gg_process_thread.signals.results.connect(self.global_gather_replace_results)
        gg_process_thread.signals.finished.connect(self.global_grouping_thread_complete)
        gg_process_thread.worker_signals.reset_gui.connect(mw.reset_gui)
        gg_process_thread.status_message = "Starting Global Grouping..."

        mw.threadpool.start(gg_process_thread)

    def gui_group_global(self):
        dataset = self.main_window.current_dataset
        if not (
            hasattr(dataset, "global_particle") and dataset.global_particle is not None
        ):
            self.build_global()
        self.start_global_grouping_thread()
        # self.main_window.status_message("No global particle found.")

    def error(self, e: Exception):
        logger.error(e)
        print(e)
        raise e


class SpectraController(QObject):
    def __init__(self, main_window: MainWindow):
        super().__init__()

        self.main_window = main_window

        self.main_window.pgSpectra_Image_View = pg.ImageView(view=pg.PlotItem())
        self.spectra_image_view = self.main_window.pgSpectra_Image_View
        self.main_window.laySpectra.addWidget(self.spectra_image_view)
        self.spectra_image_view.setPredefinedGradient("plasma")
        self.spectra_image_view.view.getAxis("left").setLabel("Wavelength (nm)")
        self.spectra_image_view.view.getAxis("bottom").setLabel("Time (s)")
        self.spectra_image_view.show()

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
        """Used to subtract the background"""

        print("gui_sub_bkg")

    def plot_spectra(
        self,
        particle: Particle = None,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
    ):
        if type(export_path) is bool:
            lock = export_path
            export_path = None
        if particle is None:
            particle = self.main_window.current_particle
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
            y_ticks = [
                [(y_ticks_pixel[i], f"{y_ticks_wavelength[i]: .1f}") for i in range(15)]
            ]
            self.spectra_image_view.view.getAxis("left").setTicks(y_ticks)

            t_series = particle.spectra.series_times
            mod_selector = len(t_series) // 30 + 1
            x_ticks_t = list()
            for i in range(len(t_series)):
                if not (mod_selector + i) % mod_selector:
                    x_ticks_t.append(t_series[i])
            x_ticks_value = np.linspace(0, spectra_data.shape[0], len(x_ticks_t))
            x_ticks = [
                [
                    (x_ticks_value[i], f"{x_ticks_t[i]:.1f}")
                    for i in range(len(x_ticks_t))
                ]
            ]
            self.spectra_image_view.view.getAxis("bottom").setTicks(x_ticks)
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
            c = self.temp_ax.pcolormesh(
                times, wavelengths, spectra_data, shading="auto", cmap="inferno"
            )
            c_bar = self.temp_fig.colorbar(c, ax=self.temp_ax)
            c_bar.set_label("Intensity (counts/s)")
            self.temp_ax.set(
                xlabel="times (s)",
                ylabel="wavelength (nm)",
                title=f"{particle.name} Spectral Trace",
            )

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
            self.main_window.lock.release()


class MyCrosshairOverlay(pg.CrosshairROI):
    def __init__(self, pos=None, size=None, **kargs):
        self._shape = None
        pg.ROI.__init__(self, pos, size, **kargs)
        self.sigRegionChanged.connect(self.invalidate)
        self.aspectLocked = True


class RasterScanController(QObject):
    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.main_window = main_window

        self.raster_scan_image_view = self.main_window.pgRaster_Scan_Image_View
        self.raster_scan_image_view.setPredefinedGradient("plasma")
        self.list_text = self.main_window.txtRaster_Scan_List
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

    def plot_raster_scan(
        self,
        particle: Particle = None,
        raster_scan: RasterScan = None,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
    ):
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
            gs = self.temp_fig.add_gridspec(
                1, 1, left=0.1, right=0.97, bottom=0.05, top=0.97
            )
            self.temp_ax = self.temp_fig.add_subplot(gs[0, 0])

            left = raster_scan.x_start
            right = left + raster_scan.range
            bottom = raster_scan.y_start
            top = bottom + raster_scan.range
            raster_scan_data = np.flip(raster_scan_data, axis=1)
            raster_scan_data = raster_scan_data.transpose()
            c = self.temp_ax.imshow(
                raster_scan_data,
                cmap="inferno",
                aspect="equal",
                extent=(left, right, bottom, top),
            )
            if len(raster_scan.particle_indexes) > 1:
                first_part = raster_scan.particle_indexes[0] + 1
                last_part = raster_scan.particle_indexes[-1] + 1
                title = f"Raster Scan for Particles {first_part}-{last_part}"
            else:
                title = (
                    f"Raster Scan for Particle {raster_scan.particle_indexes[0] + 1}"
                )
            self.temp_ax.set(xlabel="x axis (um)", ylabel="y axis (um)", title=title)
            c_bar = self.temp_fig.colorbar(c, ax=self.temp_ax)
            c_bar.set_label("intensity (counts/s)")
            hw = 0.25
            for ind in raster_scan.particle_indexes:
                part_pos = dataset.particles[ind].raster_scan_coordinates
                h_line = (
                    [part_pos[0] - hw, part_pos[0] + hw],
                    [part_pos[1], part_pos[1]],
                )
                v_line = (
                    [part_pos[0], part_pos[0]],
                    [part_pos[1] - hw, part_pos[1] + hw],
                )
                self.temp_ax.plot(*h_line, linewidth=3, color="lightgreen")
                self.temp_ax.plot(*v_line, linewidth=3, color="lightgreen")
                self.temp_ax.text(
                    part_pos[0] - hw,
                    part_pos[1] - hw,
                    str(ind + 1),
                    fontsize=14,
                    color="lightgreen",
                    fontweight="heavy",
                )
            full_path = os.path.join(export_path, title + ".png")
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
            coords = (raw_coords[0] - raster_scan.x_start) / um_per_pixel, (
                raw_coords[1] - raster_scan.y_start
            ) / um_per_pixel

            if self._crosshair_item is None:
                self._crosshair_item = self.create_crosshair_item(pos=coords)
                self.raster_scan_image_view.getView().addItem(self._crosshair_item)
            else:
                self._crosshair_item.setPos(pos=coords)

            # part_text = particle.name.split(' ')[1]
            # Offset text
            text_coords = (coords[0] - 1, coords[1] + 1)
            if self._text_item is None:
                self._text_item = self.create_text_item(
                    text=str(particle.dataset_ind + 1), pos=text_coords
                )
                self.raster_scan_image_view.getView().addItem(self._text_item)
            else:
                self._text_item.setText(text=str(particle.dataset_ind + 1))
                self._text_item.setPos(*text_coords)

            self.list_text.clear()
            dataset = self.main_window.current_dataset
            all_text = f"<h3>Raster Scan {raster_scan.h5dataset_index + 1}</h3><p>"
            rs_part_coord = [
                dataset.particles[part_ind].raster_scan_coordinates
                for part_ind in raster_scan.particle_indexes
            ]
            all_text = (
                all_text + f"<p>Range (um) = {raster_scan.range}<br></br>"
                f"Pixels per line = {raster_scan.pixel_per_line}<br></br>"
                f"Int time (ms/um) = {raster_scan.integration_time}<br></br>"
                f"X Start (um) = {raster_scan.x_start: .1f}<br></br>"
                f"Y Start (um) = {raster_scan.y_start: .1f}</p><p>"
            )
            for num, part_index in enumerate(raster_scan.particle_indexes):
                if num != 0:
                    all_text = all_text + "<br></br>"
                if particle is dataset.particles[part_index]:
                    all_text = (
                        all_text + f"<strong>{num + 1}) {particle.name}</strong>: "
                    )
                else:
                    all_text = (
                        all_text + f"{num + 1}) {dataset.particles[part_index].name}: "
                    )
                all_text = (
                    all_text + f"x={rs_part_coord[num][0]: .1f}, "
                    f"y={rs_part_coord[num][1]: .1f}"
                )
            self.list_text.setText(all_text)

        if lock:
            self.main_window.lock.release()


class AntibunchingController(QObject):

    def __init__(self, mainwindow: MainWindow, corr_widget: pg.PlotWidget, corr_sum_widget: pg.PlotWidget):
        super().__init__()
        self.main_window = main_window

        self.resolve_mode = None
        self.results_gathered = False

        self.corr_widget = corr_widget
        self.corr_plot = corr_widget.getPlotItem()
        self.setup_widget(self.corr_widget)
        self.setup_plot(self.corr_plot)

        self.corr_sum_widget = corr_sum_widget
        self.corr_sum_plot = corr_sum_widget.getPlotItem()
        self.setup_widget(self.corr_sum_widget)
        self.setup_plot(self.corr_sum_plot)

        self.temp_fig = None
        self.temp_ax = None

        self.corr = None
        self.bins = None
        self.irfdiff = 0

        self.main_window.btnLoadIRFCorr.clicked.connect(self.gui_load_irf)
        self.main_window.btnCorrCurrent.clicked.connect(self.gui_correlate_current)
        self.main_window.btnCorrSelected.clicked.connect(self.gui_correlate_selected)
        self.main_window.btnCorrAll.clicked.connect(self.gui_correlate_all)
        self.main_window.chbIRFCorrDiff.stateChanged.connect(self.gui_irf_chb)
        self.main_window.chbCurrCorrDiff.stateChanged.connect(self.gui_curr_chb)

    def setup_plot(self, plot_item: pg.PlotItem):
        # Set axis label bold and size
        axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)

        left_axis = plot_item.getAxis("left")
        bottom_axis = plot_item.getAxis("bottom")

        left_axis.setPen(axis_line_pen)
        bottom_axis.setPen(axis_line_pen)

        font = left_axis.label.font()
        font.setPointSize(10)

        left_axis.label.setFont(font)
        bottom_axis.label.setFont(font)

        left_axis.setLabel('Number of occur.', 'counts/bin')
        bottom_axis.setLabel('Delay time', 'ns')
        # plot_item.vb.setLimits(xMin=0, yMin=0)

    @staticmethod
    def setup_widget(plot_widget: pg.PlotWidget):
        # Set widget background and antialiasing
        plot_widget.setBackground(background=None)
        plot_widget.setAntialiasing(True)

    @property
    def difftime(self):
        return self.main_window.spbCorrDiff.value()

    def gui_correlate_current(self):
        self.start_corr_thread("current")

    def gui_correlate_selected(self):
        self.start_corr_thread("selected")
        # checked_parts = self.mainwindow.get_checked_particles()
        # allcorr = None
        # for part in checked_parts:
        #     bins, corr, events = self.correlate_particle(part, self.difftime)
        #     if allcorr is None:
        #         allcorr = corr
        #     else:
        #         allcorr += corr
        # self.bins = bins[:-1]
        # self.corr = allcorr
        # # plt.plot(bins[:-1], corr)
        # print(np.size(events))
        # self.plot_corr()

    def gui_correlate_all(self):
        self.start_corr_thread("all")


    def plot_corr(
        self,
        particle: Particle = None,
        for_export: bool = False,
        export_path: str = None,
        lock: bool = False,
    ) -> None:
        if type(export_path) is bool:
            lock = export_path
            export_path = None

        if particle is None:
            particle = self.main_window.current_particle

        plot_item = self.corr_widget
        plot_item.clear()

        ab_analysis = self.main_window.current_particle.ab_analysis
        if not ab_analysis.has_corr:
            logger.info("No correlation for this particle")
            return
        bins = ab_analysis.corr_bins
        corr = ab_analysis.corr_hist

        if not for_export:
            plot_pen = QPen()
            plot_pen.setCosmetic(True)

            plot_pen.setWidthF(1.5)
            plot_pen.setColor(QColor("green"))

            plot_pen.setJoinStyle(Qt.RoundJoin)
            plot_item.plot(x=bins, y=corr, pen=plot_pen, symbol=None)
            self.plot_corr_sum()
        else:
            if self.temp_fig is None:
                self.temp_fig = plt.figure()
            else:
                self.temp_fig.clf()
            self.temp_fig.set_size_inches(EXPORT_MPL_WIDTH, EXPORT_MPL_HEIGHT)
            gs = self.temp_fig.add_gridspec(1, 1, left=0.05, right=0.95)
            corr_ax = self.temp_fig.add_subplot(gs[0, 0])
            self.temp_ax = {"decay_ax": corr_ax}

            corr_ax.spines["top"].set_visible(False)
            corr_ax.spines["right"].set_visible(False)
            corr_ax.semilogy(bins, corr)

            corr_ax.set(xlabel=f"Delay time (ns)", ylabel="Correlation (counts/bin)")

        if for_export and export_path is not None:
            if not (os.path.exists(export_path) and os.path.isdir(export_path)):
                raise AssertionError("Provided path not valid")
            pname = particle.unique_name
            type_str = " corr hist.png"
            title_str = f"{pname} Second Order Correlation"
            self.temp_fig.suptitle(title_str)
            full_path = os.path.join(export_path, pname + type_str)
            self.temp_fig.savefig(full_path, dpi=EXPORT_MPL_DPI)
        if lock:
            self.main_window.lock.release()

    def plot_corr_sum(self):
        plot_item = self.corr_sum_widget
        plot_item.clear()

        allcorr = None
        bins = None
        for particle in self.mainwindow.get_checked_particles():
            ab_analysis = particle.ab_analysis
            if not ab_analysis.has_corr:
                logger.info(particle.name + ' has no correlation')
                return
            else:
                bins = ab_analysis.corr_bins
                corr = ab_analysis.corr_hist
                if allcorr is None:
                    allcorr = corr.copy()
                else:
                    allcorr += corr

        plot_pen = QPen()
        plot_pen.setCosmetic(True)

        plot_pen.setWidthF(1.5)
        plot_pen.setColor(QColor('green'))

        plot_pen.setJoinStyle(Qt.RoundJoin)
        plot_item.plot(x=bins, y=allcorr, pen=plot_pen, symbol=None)

    def gui_curr_chb(self, checked):
        mw = self.main_window
        if checked or mw.chbIRFCorrDiff.isChecked():
            mw.spbCorrDiff.setEnabled(False)
        else:
            mw.spbCorrDiff.setEnabled(True)
        if checked:
            if mw.chbIRFCorrDiff.isChecked():
                mw.chbIRFCorrDiff.setChecked(False)
            self.update_corr_diff()

    def start_corr_thread(self, mode: str = "current") -> None:
        """
        Creates a worker to calculate correlations.

        Depending on the ``current_selected_all`` parameter the worker will be
        given the necessary parameter to correlate the current, selected or all particles.

        Parameters
        ----------
        mode : {'current', 'selected', 'all'}
            Possible values are 'current' (default), 'selected', and 'all'.
        """

        assert mode in [
            "current",
            "selected",
            "all",
        ], "'corr_all' and 'corr_selected' can not both be given as parameters."

        mw = self.main_window
        cp = mw.current_particle
        if cp.is_secondary_part:
            cp = cp.prim_part
        if mode == "current":
            status_message = "Calculating correlation for Current Particle..."
            ab_objs = [cp.ab_analysis]
        elif mode == "selected":
            status_message = "Calculating correlations for Selected Particles..."
            ab_objs = [part.ab_analysis for part in mw.get_checked_particles()]
        elif mode == "all":
            status_message = "Calculating correlations for All Particles..."
            ab_objs = [part.ab_analysis for part in mw.current_dataset.particles]

        difftime = self.difftime
        window = self.main_window.spbWindow.value()
        binsize = self.main_window.spbBinSizeCorr.value()
        binsize = binsize / 1000  # convert to ns

        c_process_thread = ProcessThread()
        c_process_thread.add_tasks_from_methods(
            objects=ab_objs,
            method_name="correlate_particle",
            args=(difftime, window, binsize),
        )
        c_process_thread.signals.start_progress.connect(mw.start_progress)
        c_process_thread.signals.status_update.connect(mw.status_message)
        c_process_thread.signals.step_progress.connect(mw.update_progress)
        c_process_thread.signals.end_progress.connect(mw.end_progress)
        c_process_thread.signals.error.connect(self.error)
        c_process_thread.signals.results.connect(self.gather_replace_results)
        c_process_thread.signals.finished.connect(self.corr_thread_complete)
        c_process_thread.worker_signals.reset_gui.connect(mw.reset_gui)
        c_process_thread.status_message = status_message

        mw.threadpool.start(c_process_thread)
        mw.active_threads.append(c_process_thread)

    def gather_replace_results(
        self, results: Union[List[ProcessTaskResult], ProcessTaskResult]
    ):
        particles = self.main_window.current_dataset.particles
        part_uuids = [part.uuid for part in particles]
        if type(results) is not list:
            results = [results]
        result_part_uuids = [result.new_task_obj.uuid for result in results]
        try:
            for num, result in enumerate(results):
                result_part_ind = part_uuids.index(result_part_uuids[num])
                target_particle = particles[result_part_ind]
                result.new_task_obj._particle = target_particle
                target_particle.ab_analysis = result.new_task_obj
                print('lala', target_particle)
            self.results_gathered = True
        except ValueError as e:
            logger.error(e)

    def corr_thread_complete(self, mode: str = None):
        if self.mainwindow.current_particle is not None:
            self.mainwindow.display_data()
        if not mode == 'current':
            self.mainwindow.status_message("Done")
        self.mainwindow.current_dataset.has_corr = True
        logger.info('Correlation complete')

    def rebin_corrs(self):
        window = self.mainwindow.spbWindow.value()
        binsize = self.mainwindow.spbBinSizeCorr.value()
        if binsize == 0:
            return
        binsize = binsize / 1000  # convert to ns
        ab_objs = [part.ab_analysis for part in self.mainwindow.current_dataset.particles]
        for ab in ab_objs:
            if ab.has_corr:
                ab.rebin_corr(window, binsize)
        self.mainwindow.display_data()

    def error(self, e):
        logger.error(e)


@dataclass
class PlotFeature:
    PhotonNumber = "Photon Number"
    Intensity = "Intensity"
    Lifetime = "Lifetime"
    DW = "DW"
    ChiSquared = "Chi-Squared"
    IRFShift = "IRF Shift"

    @classmethod
    def get_list(cls) -> list:
        return [
            cls.PhotonNumber,
            cls.Intensity,
            cls.Lifetime,
            cls.DW,
            cls.ChiSquared,
            cls.IRFShift,
        ]

    @classmethod
    def get_dict(cls) -> dict:
        all_dict = {
            str(cls.PhotonNumber): cls.PhotonNumber,
            str(cls.Intensity): cls.Intensity,
            str(cls.Lifetime): cls.Lifetime,
            str(cls.DW): cls.DW,
            str(cls.ChiSquared): cls.ChiSquared,
            str(cls.IRFShift): cls.IRFShift,
        }
        return all_dict


class FilteringController(QObject):
    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.main_window = main_window
        mw = main_window

        self.plot_widget = mw.pgFilteringPlotWidget
        self.plot = self.plot_widget.getPlotItem()
        self.plot_widget.setBackground(background=None)
        self.plot_widget.setAntialiasing(True)

        self.option_linker = {
            "min_photons": (mw.chbFiltMinPhotons, mw.spnFiltMinPhotons),
            "min_intensity": (mw.chbFiltMinIntensity, mw.dsbFiltMinIntensity),
            "max_intensity": (mw.chbFiltMaxIntensity, mw.dsbFiltMaxIntensity),
            "min_lifetime": (mw.chbFiltMinLifetime, mw.dsbFiltMinLifetime),
            "max_lifetime": (mw.chbFiltMaxLifetime, mw.dsbFiltMaxLifetime),
            "use_dw": (mw.chbFiltUseDW, mw.cmbFiltDWTest),
            "min_chi_squared": (mw.chbFiltMinChiSquared, mw.dsbFiltMinChiSquared),
            "max_chi_squared": (mw.chbFiltMaxChiSquared, mw.dsbFiltMaxChiSquared),
            "min_irf_shift": (mw.chbFiltMinIRFShift, mw.dsbFiltMinIRFShift),
            "max_irf_shift": (mw.chbFiltMaxIRFShift, mw.dsbFiltMaxIRFShift),
            "force_through_origin": (mw.chbFiltForceOrigin, None),
        }

        for option, check_box_and_value_object in self.option_linker.items():
            check_box, value_object = check_box_and_value_object
            if value_object is not None:
                check_box.stateChanged.connect(
                    (lambda opt: lambda: self.filter_option_changed(option=opt))(option)
                )

        mw.btnFiltPhotonNumberDistribution.clicked.connect(
            lambda: self.plot_features(feature_x=PlotFeature.PhotonNumber)
        )
        mw.btnFiltIntensityDistribution.clicked.connect(
            lambda: self.plot_features(feature_x=PlotFeature.Intensity)
        )
        mw.btnFiltLifetimeDistribution.clicked.connect(
            lambda: self.plot_features(feature_x=PlotFeature.Lifetime)
        )
        mw.btnFiltDWDistribution.clicked.connect(
            lambda: self.plot_features(feature_x=PlotFeature.DW)
        )
        mw.btnFiltChiSquaredDistribution.clicked.connect(
            lambda: self.plot_features(feature_x=PlotFeature.ChiSquared)
        )
        mw.btnFiltIRFShiftDistribution.clicked.connect(
            lambda: self.plot_features(feature_x=PlotFeature.IRFShift)
        )

        filter_nums = [
            mw.spnFiltMinPhotons,
            mw.dsbFiltMinIntensity,
            mw.dsbFiltMaxIntensity,
            mw.dsbFiltMinLifetime,
            mw.dsbFiltMaxLifetime,
            mw.dsbFiltMinChiSquared,
            mw.dsbFiltMaxChiSquared,
            mw.dsbFiltMinIRFShift,
            mw.dsbFiltMaxIRFShift,
        ]

        for num_control in filter_nums:
            num_control.valueChanged.connect(
                lambda: self.plot_features(use_current_plot=True)
            )
        mw.cmbFiltDWTest.currentTextChanged.connect(
            lambda: self.plot_features(use_current_plot=True)
        )

        self.current_plot_type = (
            None,
            None,
        )  # (PlotFeature.Intensity, PlotFeature.Lifetime)
        self.current_particles_to_use = None
        self.current_data_points_to_use = None
        self.current_particles = list()
        self.levels_to_use = None
        self.fit_result = None
        self.has_fit = False
        self.is_normalized = False

        self.setup_plot(*self.current_plot_type, clear_plot=False, is_first_setup=True)

        mw.cmbFiltFeatureX.currentTextChanged.connect(
            lambda: self.two_features_changed(x_changed=True)
        )
        mw.cmbFiltFeatureY.currentTextChanged.connect(
            lambda: self.two_features_changed(x_changed=False)
        )
        mw.tlbFiltSwitchFeatures.clicked.connect(lambda: self.switch_two_features())
        mw.btnFiltPlotTwoFeatures.clicked.connect(
            lambda: self.plot_features(use_selected_two_features=True)
        )
        mw.btnFiltFit.clicked.connect(self.fit_intensity_lifetime)

        mw.chbFiltAutoNumBins.stateChanged.connect(self.auto_num_bins_changed)
        mw.spnFiltNumBins.valueChanged.connect(
            lambda: self.plot_features(use_current_plot=True)
        )

        mw.cmbFiltParticlesToUse.currentTextChanged.connect(
            lambda: self.plot_features(use_current_plot=True)
        )
        mw.cmbFiltUseResolvedOrGrouped.currentTextChanged.connect(
            lambda: self.plot_features(use_current_plot=True)
        )
        mw.chbFiltUseROI.stateChanged.connect(
            lambda: self.plot_features(use_current_plot=True)
        )

        mw.chbFiltApplyAllFilters.stateChanged.connect(
            lambda: self.plot_features(use_current_plot=True)
        )

        mw.btnFiltApplyFilters.clicked.connect(self.apply_filters)
        mw.btnFiltApplyNormalization.clicked.connect(self.apply_normalization)
        mw.btnFiltResetFilters.clicked.connect(self.reset_filters)
        mw.btnFiltResetAllFilters.clicked.connect(self.reset_dataset_filter)
        mw.btnFiltResetNormalization.clicked.connect(self.reset_normalization)

        self.plot_pen = QPen()
        self.plot_pen.setCosmetic(True)
        self.plot_pen.setWidthF(2)
        self.plot_pen.setColor(QColor("black"))

        self.distribution_item = pg.PlotCurveItem(
            x=[0, 0],
            y=[0],
            stepMode=True,
            pen=self.plot_pen,
            fillLevel=0,
            brush=QColor("lightGray"),
        )

        self.two_feature_item = pg.ScatterPlotItem(
            x=[0], y=[0], pen=self.plot_pen, size=3
        )

        self.plot_fit_pen = QPen()
        self.plot_fit_pen.setCosmetic(True)
        self.plot_fit_pen.setWidthF(2)
        self.plot_fit_pen.setColor(QColor("red"))

        self.int_lifetime_fit_item = pg.PlotCurveItem(
            x=[0],
            y=[0],
            pen=self.plot_fit_pen,
        )

    @staticmethod
    def _get_label(feature: PlotFeature = None) -> tuple:
        label = "Count"
        unit = None
        if feature is not None:
            label = str(feature)
            if feature == PlotFeature.Intensity:
                unit = "counts/s"
            elif feature == PlotFeature.Lifetime or feature == PlotFeature.IRFShift:
                unit = "ns"
            else:
                unit = None
        return label, unit

    def setup_plot(
        self,
        feature_x: PlotFeature,
        feature_y: PlotFeature = None,
        clear_plot: bool = True,
        is_first_setup: bool = False,
    ):
        left_axis = self.plot.getAxis("left")
        bottom_axis = self.plot.getAxis("bottom")
        if is_first_setup:
            self.plot.vb.setLimits(yMin=0)
            axis_line_pen = pg.mkPen(color=(0, 0, 0), width=2)
            left_axis.setPen(axis_line_pen)
            bottom_axis.setPen(axis_line_pen)
            font = left_axis.label.font()
            font.setPointSize(10)
            left_axis.label.setFont(font)
            bottom_axis.label.setFont(font)
            # lef
        # if clear_plot:
        #     plot_item.clear()

        bottom_axis.setLabel(*self._get_label(feature_x))
        left_axis.setLabel(*self._get_label(feature_y))

        self.current_plot_type = (feature_x, feature_y)

    def filter_option_changed(self, option: str):
        check_box, value_object = self.option_linker[option]
        value_object.setEnabled(check_box.isChecked())
        self.plot_features(*self.current_plot_type)

    def auto_num_bins_changed(self):
        self.main_window.spnFiltNumBins.setEnabled(
            not self.main_window.chbFiltAutoNumBins.isChecked()
        )
        self.plot_features(*self.current_plot_type)

    def get_two_features(self) -> tuple:
        return (
            self.main_window.cmbFiltFeatureX.currentText(),
            self.main_window.cmbFiltFeatureY.currentText(),
        )

    def change_plot_type(
        self,
        feature_x: Union[PlotFeature, str] = None,
        feature_y: Union[PlotFeature, str] = None,
    ):
        if feature_x is None and feature_y is None:
            feature_x, feature_y = self.get_two_features()
        self.setup_plot(feature_x=feature_x, feature_y=feature_y)

    def set_levels_to_use(self):
        particles_changed = False
        particles_to_use = self.main_window.cmbFiltParticlesToUse.currentText()
        if particles_to_use != self.current_particles_to_use:
            particles = list()
            if particles_to_use == "Current":
                particles = [self.main_window.current_particle]
            elif particles_to_use == "Selected":
                particles = self.main_window.get_checked_particles()
            elif particles_to_use == "All":
                particles = self.main_window.current_dataset.particles
            self.current_particles = particles

            self.current_particles_to_use = particles_to_use
            particles_changed = True

        data_points_to_use = self.main_window.cmbFiltUseResolvedOrGrouped.currentText()
        if data_points_to_use != self.current_data_points_to_use or particles_changed:
            particles = self.current_particles
            if data_points_to_use == "Grouped":
                if not self.main_window.chbFiltUseROI.isChecked():
                    levels = [
                        particle.group_levels
                        for particle in particles
                        if particle.has_groups
                    ]
                else:
                    levels = [
                        particle.group_levels_roi
                        for particle in particles
                        if particle.has_groups
                    ]
            else:
                if not self.main_window.chbFiltUseROI.isChecked():
                    levels = [
                        particle.cpts.levels
                        for particle in particles
                        if particle.has_levels
                    ]
                else:
                    levels = [
                        particle.levels_roi_force
                        for particle in particles
                        if particle.has_levels
                    ]
            self.levels_to_use = (
                list(np.concatenate(levels)) if len(levels) > 0 else list()
            )
            self.current_data_points_to_use = data_points_to_use

    def get_data(self) -> tuple:
        self.set_levels_to_use()

        ints = list()
        taus = list()
        for level in self.levels_to_use:
            if level.histogram.fitted:
                ints.append(level.int_p_s)
                taus.append(level.histogram.avtau)
        else:
            ints, taus = None, None

        return ints, taus

    def two_features_changed(self, x_changed: bool = None):
        self.set_levels_to_use()
        if x_changed is None:
            raise ValueError("No argument provided.")
        feature_changed_cmb = (
            self.main_window.cmbFiltFeatureX
            if x_changed
            else self.main_window.cmbFiltFeatureY
        )
        other_feature_cmb = (
            self.main_window.cmbFiltFeatureY
            if x_changed
            else self.main_window.cmbFiltFeatureX
        )

        feature_changed_cmb.blockSignals(True)
        other_feature_cmb.blockSignals(True)

        new_other_features = PlotFeature.get_dict()
        feature_selected = new_other_features.pop(feature_changed_cmb.currentText())

        new_changed_features = PlotFeature.get_dict()
        other_selected_feature = new_changed_features.pop(
            other_feature_cmb.currentText()
        )

        feature_changed_cmb.clear()
        feature_changed_cmb.addItems(new_changed_features.keys())
        feature_changed_cmb.setCurrentText(feature_selected)

        other_feature_cmb.clear()
        other_feature_cmb.addItems(new_other_features.keys())
        other_feature_cmb.setCurrentText(other_selected_feature)

        feature_changed_cmb.blockSignals(False)
        other_feature_cmb.blockSignals(False)

        current_feature_x, current_feature_y = self.current_plot_type
        if (
            current_feature_x is not None and current_feature_y is not None
        ):  # True if current is two feature plot
            self.plot_features(use_current_plot=True)

    def switch_two_features(self):
        self.set_levels_to_use()
        selected_feature_x = self.main_window.cmbFiltFeatureX.currentText()
        selected_feature_y = self.main_window.cmbFiltFeatureY.currentText()
        items_x = PlotFeature.get_dict()
        _ = items_x.pop(selected_feature_y)
        items_y = PlotFeature.get_dict()
        _ = items_y.pop(selected_feature_x)

        selected_feature_x, selected_feature_y = selected_feature_y, selected_feature_x
        items_x, items_y = items_y, items_x

        self.main_window.cmbFiltFeatureX.blockSignals(True)
        self.main_window.cmbFiltFeatureX.clear()
        self.main_window.cmbFiltFeatureX.addItems(items_x)
        self.main_window.cmbFiltFeatureX.setCurrentText(selected_feature_x)
        self.main_window.cmbFiltFeatureX.blockSignals(False)

        self.main_window.cmbFiltFeatureY.blockSignals(True)
        self.main_window.cmbFiltFeatureY.clear()
        self.main_window.cmbFiltFeatureY.addItems(items_y)
        self.main_window.cmbFiltFeatureY.setCurrentText(selected_feature_y)
        self.main_window.cmbFiltFeatureY.blockSignals(False)

        self.plot_features(use_selected_two_features=True)

    def plot_features(
        self,
        feature_x: Union[PlotFeature, str] = None,
        feature_y: Union[PlotFeature, str] = None,
        use_current_plot: bool = False,
        use_selected_two_features: bool = False,
    ):
        self.set_levels_to_use()
        if use_current_plot:
            if use_selected_two_features:
                raise ValueError("Conflicting options")
            if feature_x is not None or feature_y is not None:
                raise ValueError("Use current plot option excludes provided features")
            feature_x, feature_y = self.current_plot_type
            use_selected_two_features = feature_x is not None and feature_y is not None

        if not use_current_plot or use_selected_two_features:
            if use_selected_two_features:
                if not use_current_plot and (
                    feature_x is not None or feature_y is not None
                ):
                    raise ValueError(
                        "Use selected two features excludes provided features"
                    )
                else:
                    feature_x = self.main_window.cmbFiltFeatureX.currentText()
                    feature_y = self.main_window.cmbFiltFeatureY.currentText()
            else:
                if feature_x is None and feature_y is not None:
                    raise ValueError(
                        "Can not provide only a plot feature for the Y-Axis"
                    )
                elif (
                    feature_x is not None or feature_y is not None
                ) and feature_x == feature_y:
                    raise ValueError("Can not provide the same feature for x and y")
                if feature_x is None and feature_y is None:
                    logger.warning("No feature(s) provided and no options selected")
                    return None

        if (feature_x, feature_y) != (None, None):
            if self.current_plot_type != (feature_x, feature_y):
                self.change_plot_type(feature_x=feature_x, feature_y=feature_y)

            self.set_levels_to_use()

            is_distribution = True if feature_y is None else False
            plot_item = (
                self.distribution_item if is_distribution else self.two_feature_item
            )
            could_have_fit_and_shouldnt = self.has_fit
            if could_have_fit_and_shouldnt:
                could_have_fit_and_shouldnt &= (feature_x, feature_y) != (
                    PlotFeature.Intensity,
                    PlotFeature.Lifetime,
                )
            if plot_item not in self.plot.items or could_have_fit_and_shouldnt:
                self.plot.clear()
                self.plot.addItem(plot_item)
            if is_distribution:
                self.plot_distribution()
            else:
                self.plot_two_features()

    @staticmethod
    def _filter_numeric_data(
        feature_data: np.ndarray,
        are_used_flags: List[bool],
        test_min: bool = False,
        test_max: bool = False,
        min_value=None,
        max_value=None,
    ):
        num_datapoints_filtered = None

        if test_min:
            are_used_flags = np.logical_and(
                are_used_flags,
                [not np.isnan(value) and value >= min_value for value in feature_data],
            )
            feature_data = np.array(
                [
                    value if passed_filter else np.NaN
                    for value, passed_filter in zip(feature_data, are_used_flags)
                ]
            )
        if test_max:
            are_used_flags = np.logical_and(
                are_used_flags,
                [not np.isnan(value) and value <= max_value for value in feature_data],
            )
            feature_data = np.array(
                [
                    value if passed_filter else np.NaN
                    for value, passed_filter in zip(feature_data, are_used_flags)
                ]
            )
        if test_min or test_max:
            num_datapoints_filtered = np.sum(~np.isnan(feature_data))

        return feature_data, num_datapoints_filtered, are_used_flags

    def get_feature_data(self, feature: Union[PlotFeature, str]) -> tuple:
        if self.levels_to_use is None:
            self.set_levels_to_use()
        levels = self.levels_to_use
        histograms = [level.histogram for level in levels]
        feature_data = None
        num_datapoints = 0
        num_datapoints_filtered = None
        is_intensity_or_histogram = None
        are_used_flags = None

        if feature == PlotFeature.PhotonNumber:
            are_used_flags = [level.num_photons is not None for level in levels]
            feature_data = np.array(
                [
                    level.num_photons if is_used else np.NaN
                    for level, is_used in zip(levels, are_used_flags)
                ]
            )
            num_datapoints = np.sum(~np.isnan(feature_data))
            (
                feature_data,
                num_datapoints_filtered,
                passed_filter_flags,
            ) = self._filter_numeric_data(
                feature_data=feature_data,
                are_used_flags=are_used_flags,
                test_min=self.main_window.chbFiltMinPhotons.isChecked(),
                min_value=self.main_window.spnFiltMinPhotons.value(),
            )
            are_used_flags = np.logical_and(are_used_flags, passed_filter_flags)
            is_intensity_or_histogram = "level"

        elif feature == PlotFeature.Intensity:
            are_used_flags = [level.int_p_s is not None for level in levels]
            feature_data = np.array(
                [
                    level.int_p_s if is_used else np.NaN
                    for level, is_used in zip(levels, are_used_flags)
                ]
            )
            num_datapoints = np.sum(~np.isnan(feature_data))
            (
                feature_data,
                num_datapoints_filtered,
                passed_filter_flags,
            ) = self._filter_numeric_data(
                feature_data=feature_data,
                are_used_flags=are_used_flags,
                test_min=self.main_window.chbFiltMinIntensity.isChecked(),
                test_max=self.main_window.chbFiltMaxIntensity.isChecked(),
                min_value=self.main_window.dsbFiltMinIntensity.value(),
                max_value=self.main_window.dsbFiltMaxIntensity.value(),
            )
            are_used_flags = np.logical_and(are_used_flags, passed_filter_flags)
            is_intensity_or_histogram = "level"

        elif feature == PlotFeature.Lifetime:
            are_used_flags = [
                histogram.fitted and histogram.avtau is not None
                for histogram in histograms
            ]
            feature_data = [
                histogram.avtau if is_used else np.NaN
                for histogram, is_used in zip(histograms, are_used_flags)
            ]
            feature_data = np.array(
                [
                    value[0] if type(value) is list and len(value) == 1 else value
                    for value in feature_data
                ]
            )
            num_datapoints = np.sum(~np.isnan(feature_data))
            (
                feature_data,
                num_datapoints_filtered,
                passed_filter_flags,
            ) = self._filter_numeric_data(
                feature_data=feature_data,
                are_used_flags=are_used_flags,
                test_min=self.main_window.chbFiltMinLifetime.isChecked(),
                test_max=self.main_window.chbFiltMaxLifetime.isChecked(),
                min_value=self.main_window.dsbFiltMinLifetime.value(),
                max_value=self.main_window.dsbFiltMaxLifetime.value(),
            )
            are_used_flags = np.logical_and(are_used_flags, passed_filter_flags)
            is_intensity_or_histogram = "histogram"

        elif feature == PlotFeature.DW:
            are_used_flags = [
                histogram.fitted and histogram.dw is not None
                for histogram in histograms
            ]
            feature_data = np.array(
                [
                    histogram.dw if is_used else np.NaN
                    for histogram, is_used in zip(histograms, are_used_flags)
                ]
            )
            num_datapoints = np.sum(~np.isnan(feature_data))
            if self.main_window.chbFiltUseDW.isChecked():
                selected_dw_test = self.main_window.cmbFiltDWTest.currentText()
                dw_ind = None
                if selected_dw_test == "5%":
                    dw_ind = 0
                elif selected_dw_test == "1%":
                    dw_ind = 1
                elif selected_dw_test == "0.3%":
                    dw_ind = 2
                elif selected_dw_test == "0.1%":
                    dw_ind = 3
                are_used_flags = np.logical_and(
                    are_used_flags,
                    [
                        not np.isnan(value)
                        and histogram.dw >= histogram.dw_bound[dw_ind]
                        for value, histogram in zip(feature_data, histograms)
                    ],
                )
                feature_data = np.array(
                    [
                        value if is_used else np.NaN
                        for (value, is_used) in zip(feature_data, are_used_flags)
                    ]
                )
                num_datapoints_filtered = np.sum(~np.isnan(feature_data))
            is_intensity_or_histogram = "histogram"

        elif feature == PlotFeature.IRFShift:
            are_used_flags = [
                histogram.fitted and histogram.shift is not None
                for histogram in histograms
            ]
            feature_data = np.array(
                [
                    histogram.shift if is_used else np.NaN
                    for histogram, is_used in zip(histograms, are_used_flags)
                ]
            )
            num_datapoints = np.sum(~np.isnan(feature_data))
            (
                feature_data,
                num_datapoints_filtered,
                passed_filter_flags,
            ) = self._filter_numeric_data(
                feature_data=feature_data,
                are_used_flags=are_used_flags,
                test_min=self.main_window.chbFiltMinIRFShift.isChecked(),
                test_max=self.main_window.chbFiltMaxIRFShift.isChecked(),
                min_value=self.main_window.dsbFiltMinIRFShift.value(),
                max_value=self.main_window.dsbFiltMaxIRFShift.value(),
            )
            are_used_flags = np.logical_and(are_used_flags, passed_filter_flags)
            is_intensity_or_histogram = "histogram"

        elif feature == PlotFeature.ChiSquared:
            are_used_flags = [
                histogram.fitted and histogram.chisq is not None
                for histogram in histograms
            ]
            feature_data = np.array(
                [
                    histogram.chisq if is_used else np.NaN
                    for histogram, is_used in zip(histograms, are_used_flags)
                ]
            )
            num_datapoints = np.sum(~np.isnan(feature_data))
            (
                feature_data,
                num_datapoints_filtered,
                passed_filter_flags,
            ) = self._filter_numeric_data(
                feature_data=feature_data,
                are_used_flags=are_used_flags,
                test_min=self.main_window.chbFiltMinChiSquared.isChecked(),
                test_max=self.main_window.chbFiltMaxChiSquared.isChecked(),
                min_value=self.main_window.dsbFiltMinChiSquared.value(),
                max_value=self.main_window.dsbFiltMaxChiSquared.value(),
            )
            are_used_flags = np.logical_and(are_used_flags, passed_filter_flags)
            is_intensity_or_histogram = "histogram"

        return (
            feature_data,
            num_datapoints,
            num_datapoints_filtered,
            are_used_flags,
            is_intensity_or_histogram,
        )

    def get_all_feature_data(
        self,
    ) -> tuple[dict[Any, dict[str, Any]], list[bool] | Any, list[bool] | Any]:
        all_features = PlotFeature.get_dict()
        all_feature_data = dict()
        levels_used = [True] * len(self.levels_to_use)
        histograms_used = [True] * len(self.levels_to_use)
        for key, item in all_features.items():
            (
                feature_data,
                num_datapoints,
                num_datapoints_filtered,
                filter_flags,
                level_or_hist,
            ) = self.get_feature_data(feature=item)
            all_feature_data[key] = {
                "feature_data": feature_data,
                "num_datapoints": num_datapoints,
                "num_datapoints_filtered": num_datapoints_filtered,
            }
            if level_or_hist == "level":
                levels_used = np.logical_and(levels_used, filter_flags)
            else:
                histograms_used = np.logical_and(histograms_used, filter_flags)

        return all_feature_data, levels_used, histograms_used

    def get_all_filter(self) -> np.ndarray:
        all_feature_data, levels_used, histograms_used = self.get_all_feature_data()
        all_filter = None
        for _, feature_all_data in all_feature_data.items():
            if all_filter is None:
                all_filter = ~np.isnan(feature_all_data["feature_data"])
            else:
                all_filter &= ~np.isnan(feature_all_data["feature_data"])
        return all_filter

    def set_limits(
        self,
        feature_x: Union[PlotFeature, str] = None,
        feature_y: Union[PlotFeature, str] = None,
    ):
        if feature_x is None and feature_y is None:
            feature_x, feature_y = self.current_plot_type

        if feature_y == PlotFeature.IRFShift:
            self.plot.vb.setLimits(yMin=None)
        else:
            self.plot.vb.setLimits(yMin=0)

        if feature_x == PlotFeature.IRFShift:
            self.plot.vb.setLimits(xMin=None)
        else:
            self.plot.vb.setLimits(xMin=0)

    def plot_distribution(self):
        feature, _ = self.current_plot_type

        (
            feature_data,
            num_datapoints,
            num_datapoints_filtered,
            _,
            _,
        ) = self.get_feature_data(feature=feature)
        feature_data = feature_data[~np.isnan(feature_data)]

        if feature_data is not None:
            self.set_limits(feature_x=feature)

            is_auto_num_bins = self.main_window.chbFiltAutoNumBins.isChecked()
            if is_auto_num_bins:
                bin_edges = "auto"
            else:
                num_bins = self.main_window.spnFiltNumBins.value()
                bin_edges = np.histogram_bin_edges(feature_data, num_bins)

            bin_edges, hist_data = np.histogram(
                feature_data, bins=bin_edges, density=False
            )

            if is_auto_num_bins:
                self.main_window.spnFiltNumBins.blockSignals(True)
                self.main_window.spnFiltNumBins.setValue(len(bin_edges))
                self.main_window.spnFiltNumBins.blockSignals(False)

            self.distribution_item.setData(x=hist_data, y=bin_edges)

            if num_datapoints_filtered is None:
                num_datapoints_text = f"# Datapoints: {num_datapoints}"
            else:
                num_datapoints_text = f"# Datapoints: {num_datapoints_filtered} ({num_datapoints} unfiltered)"
            self.main_window.lblFiltNumDatapoints.setText(num_datapoints_text)
            self.plot.autoRange()
        else:
            logger.warning("No feature data found")

    def plot_two_features(self):
        feature_x, feature_y = self.current_plot_type

        self.set_limits(feature_x=feature_x, feature_y=feature_y)

        featured_x_data, num_data_x, num_data_x_filt, _, _ = self.get_feature_data(
            feature=feature_x
        )
        featured_y_data, num_data_y, num_data_y_filt, _, _ = self.get_feature_data(
            feature=feature_y
        )

        not_nan_values = (~np.isnan(featured_x_data)) & (~np.isnan(featured_y_data))
        did_all_filter = False
        if self.main_window.chbFiltApplyAllFilters.isChecked():
            all_filter = self.get_all_filter()
            did_all_filter = np.sum(all_filter) < np.sum(not_nan_values)
            not_nan_values &= all_filter

        featured_x_data = featured_x_data[not_nan_values]
        featured_y_data = featured_y_data[not_nan_values]

        self.two_feature_item.setData(x=featured_x_data, y=featured_y_data)

        num_data = np.min([num_data_y, num_data_y])
        if not did_all_filter and (num_data_x_filt is None and num_data_y_filt is None):
            num_datapoints_text = f"# Datapoints: {num_data}"
        else:
            num_data_filtered = np.sum(not_nan_values)
            num_datapoints_text = (
                f"# Datapoints: {num_data_filtered} ({num_data} unfiltered)"
            )
        self.main_window.lblFiltNumDatapoints.setText(num_datapoints_text)

        if (feature_x, feature_y) == (
            PlotFeature.Intensity,
            PlotFeature.Lifetime,
        ) and self.has_fit:
            self.plot_fit_result()
        self.plot.autoRange()

    def prepare_plot_for_int_lifetime_fit(self):
        feature_x, feature_y = self.current_plot_type
        if (feature_x, feature_y) != (PlotFeature.Intensity, PlotFeature.Lifetime):
            self.plot_features(
                feature_x=PlotFeature.Intensity, feature_y=PlotFeature.Lifetime
            )
            feature_x, feature_y = (PlotFeature.Intensity, PlotFeature.Lifetime)
            self.main_window.cmbFiltFeatureX.blockSignals(True)
            self.main_window.cmbFiltFeatureY.blockSignals(True)

            other_x_features = PlotFeature.get_dict()
            _ = other_x_features.pop(feature_x)
            other_y_features = PlotFeature.get_dict()
            _ = other_y_features.pop(feature_y)

            self.main_window.cmbFiltFeatureX.clear()
            self.main_window.cmbFiltFeatureX.addItems(other_y_features.keys())
            self.main_window.cmbFiltFeatureX.setCurrentText(feature_x)

            self.main_window.cmbFiltFeatureY.clear()
            self.main_window.cmbFiltFeatureY.addItems(other_x_features.keys())
            self.main_window.cmbFiltFeatureY.setCurrentText(feature_y)

            self.main_window.cmbFiltFeatureX.blockSignals(False)
            self.main_window.cmbFiltFeatureY.blockSignals(False)

    def plot_fit_result(self):
        assert self.has_fit, "No fit to plot"
        assert self.current_plot_type == (
            PlotFeature.Intensity,
            PlotFeature.Lifetime,
        ), "Incorrect plot type for fit"

        slope = self.fit_result["slope"]
        slope_err = self.fit_result["slope_err"]
        has_intercept = self.fit_result["has_intercept"]
        intercept = self.fit_result["intercept"]
        intercept_err = self.fit_result["intercept_err"]
        rsquared = self.fit_result["rsquared"]

        int_data, tau_data = self.two_feature_item.getData()

        int_model = np.linspace(0, np.max(int_data), 100)
        tau_model = int_model * slope
        if has_intercept:
            tau_model += intercept

        self.int_lifetime_fit_item.setData(x=int_model, y=tau_model)
        if self.int_lifetime_fit_item in self.plot.items:
            pass
        else:
            self.plot.addItem(self.int_lifetime_fit_item)

        fit_result_text = f"Fit: tau = ({slope:.3e} +- {slope_err:.1e})*int"
        fit_result_text += (
            f" + ({intercept:.3e} +- {intercept_err:.1e})" if has_intercept else ""
        )
        fit_result_text += f"  with R^2 = {rsquared:.3f}"
        self.main_window.lblFiltResults.setText(fit_result_text)

    def fit_intensity_lifetime(self):
        self.prepare_plot_for_int_lifetime_fit()
        force_origin = self.main_window.chbFiltForceOrigin.isChecked()
        int_data, lifetime_data = self.two_feature_item.getData()
        df = pd.DataFrame(data={"int": int_data, "tau": lifetime_data})
        formula = "tau ~ int + 0" if force_origin else "tau ~ int"
        model = smf.ols(formula=formula, data=df)
        fit = model.fit()

        slope = fit.params.int
        slope_err = fit.bse.int
        intercept = None
        intercept_err = None
        if not force_origin:
            intercept = fit.params.Intercept
            intercept_err = fit.bse.Intercept
        rsquared = fit.rsquared
        self.fit_result = {
            "slope": slope,
            "slope_err": slope_err,
            "has_intercept": not force_origin,
            "intercept": intercept,
            "intercept_err": intercept_err,
            "rsquared": rsquared,
            "fit": fit,
        }
        self.has_fit = True

        self.plot_fit_result()

    def apply_filters(self):
        self.set_levels_to_use()
        all_filters = self.get_all_filter()
        for level, level_filter in zip(self.levels_to_use, all_filters):
            level.is_filtered_out = not level_filter
        self.main_window.lblFiltResults.setText("Filters Applied")

    def reset_filters(self):
        self.set_levels_to_use()
        for level in self.levels_to_use:
            level.is_filtered_out = False
        self.main_window.lblFiltResults.setText("Filters Reset")

    def reset_dataset_filter(self):
        for particle in self.main_window.current_dataset.particles:
            if particle.has_levels:
                for level in particle.cpts.levels:
                    level.is_filtered_out = False
                if particle.has_groups:
                    for step in particle.ahca.steps:
                        for group_level in step.group_levels:
                            group_level.is_filtered_out = False
        self.main_window.lblFiltResults.setText("All Filters Reset")

    def get_all_levels_with_lifetime(self) -> list:
        levels = list()
        if not self.main_window.chbFiltApplyNormalizationAll.isChecked():
            levels = self.levels_to_use
        else:
            for particle in self.main_window.current_dataset.particles:
                if particle.has_levels:
                    for level in particle.cpts.levels:
                        if level.histogram.fitted:
                            levels.append(level)
                    if particle.has_groups:
                        for level in particle.ahca.selected_step.group_levels:
                            if (
                                hasattr(level, "histogram")
                                and level.histogram is not None
                                and level.histogram.fitted
                            ):
                                levels.append(level)
        return levels

    def reset_normalization(self):
        all_levels = self.get_all_levels_with_lifetime()
        for level in all_levels:
            if hasattr(level, "is_normalized") and level.is_normalized:
                level.int_p_s = level.unnorm_int_p_s
                level.num_photons = level.unnorm_num_photons
            level.is_normalized = False
        self.is_normalized = False

    def apply_normalization(self):
        fit_result = self.fit_result
        intercept = fit_result["intercept"]
        intercept = 0 if intercept is None else intercept
        only_drift = self.main_window.chbFiltOnlyDriftNormalization.isChecked()

        levels_to_norm = self.get_all_levels_with_lifetime()

        for level in levels_to_norm:
            if hasattr(level, "is_filtered_out") and level.is_filtered_out is True:
                if hasattr(level, "is_normalized") and level.is_normalized is True:
                    level.num_photons = level.unnorm_num_photons
                    level.int_p_s = level.unnorm_int_p_s
                    level.is_normalized = False
                continue
            level.unnorm_int_p_s = level.int_p_s
            level.unnorm_num_photons = level.num_photons
            avtau = level.histogram.tau
            if type(avtau) is list:
                if len(avtau) == 1:
                    avtau = avtau[0]
                else:
                    raise ValueError("Multiple average lifetime values")
            norm_int_p_s = (avtau - intercept) / fit_result["slope"]
            if only_drift:
                if norm_int_p_s >= level.int_p_s:
                    level.is_normalized = True
                    continue
            level.num_photons = int(np.round(norm_int_p_s * level.dwell_time_s))
            level.int_p_s = level.num_photons / level.dwell_time_s
            level.is_normalized = True

        self.is_normalized = True
        self.main_window.lblFiltResults.setText("Applied Normalization")
        self.plot_features(use_current_plot=True)

