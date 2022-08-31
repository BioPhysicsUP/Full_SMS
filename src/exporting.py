from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, List
from time import sleep
import sys

from PyQt5.QtWidgets import QFileDialog
import numpy as np
import pandas as pd
from pyarrow import feather

from my_logger import setup_logger
from PyQt5.QtCore import QRunnable, pyqtSlot
from signals import WorkerSignals
from threading import Lock
# from multiprocessing import Lock
import matplotlib
matplotlib.use('Agg')

if TYPE_CHECKING:
    from main import MainWindow
    from change_point import Level
    from grouping import Group

logger = setup_logger(__name__)


DATAFRAME_FORMATS = {'Parquet (*.parquet)': 0,
                     'Feather (*.ftr)': 1,
                     'Feather (*.df)': 2,
                     'Pickle (*.pkl)': 3,
                     'HDF (*.h5)': 4,
                     'Excel (*.xlsx)': 5,
                     'CSV (*.csv)': 6}


class ExportWorker(QRunnable):
    def __init__(self, mainwindow: MainWindow,
                 mode: str = None,
                 lock: Lock = None):
        super(ExportWorker, self).__init__()
        self.main_window = mainwindow
        self.mode = mode
        self.lock = lock
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            export_data(mainwindow=self.main_window,
                        mode=self.mode,
                        signals=self.signals,
                        lock=self.lock)
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.fitting_finished.emit(self.mode)


def export_data(mainwindow: MainWindow,
                mode: str = None,
                signals: WorkerSignals = None,
                lock: Lock = None):
    assert mode in ['current', 'selected', 'all'], "MainWindow\tThe mode parameter is invalid"

    if mode == 'current':
        particles = [mainwindow.current_particle]
    elif mode == 'selected':
        particles = mainwindow.get_checked_particles()
    else:
        particles = mainwindow.current_dataset.particles

    f_dir = QFileDialog.getExistingDirectory(mainwindow)
    f_dir = os.path.abspath(f_dir)

    if not f_dir:
        return
    else:
        raster_scans_use = [part.raster_scan.dataset_index for part in particles]
        raster_scans_use = np.unique(raster_scans_use).tolist()

    if not lock:
        lock = Lock()

    use_roi = mainwindow.chbEx_Use_ROI.isChecked()

    ex_traces = mainwindow.chbEx_Trace.isChecked()
    ex_levels = mainwindow.chbEx_Levels.isChecked()
    ex_plot_intensities = mainwindow.chbEx_Plot_Intensity.isChecked()
    ex_plot_with_levels = False
    ex_plot_and_groups = False
    if ex_plot_intensities:
        ex_plot_int_only = mainwindow.rdbInt_Only.isChecked()
        if not ex_plot_int_only:
            if mainwindow.rdbWith_Levels.isChecked():
                ex_plot_with_levels = True
            else:
                ex_plot_and_groups = True
    ex_grouped_levels = mainwindow.chbEx_Grouped_Levels.isChecked()
    ex_grouping_info = mainwindow.chbEx_Grouping_Info.isChecked()
    ex_grouping_results = mainwindow.chbEx_Grouping_Results.isChecked()
    ex_plot_grouping_bics = mainwindow.chbEx_Plot_Group_BIC.isChecked()
    ex_lifetime = mainwindow.chbEx_Lifetimes.isChecked()
    ex_hist = mainwindow.chbEx_Hist.isChecked()
    ex_plot_lifetimes = mainwindow.chbEx_Plot_Lifetimes.isChecked()
    ex_plot_with_fit = False
    ex_plot_and_residuals = False
    if ex_plot_lifetimes:
        ex_plot_hist_only = mainwindow.rdbHist_Only.isChecked()
        if not ex_plot_hist_only:
            if mainwindow.rdbWith_Fit.isChecked():
                ex_plot_with_fit = True
            else:
                ex_plot_and_residuals = True
        ex_plot_lifetimes_only_groups = mainwindow.chbEx_Plot_Lifetimes_Only_Groups.isChecked()
    ex_spectra_2d = mainwindow.chbEx_Spectra_2D.isChecked()
    ex_plot_spectra = mainwindow.chbEx_Plot_Spectra.isChecked()
    ex_raster_scan_2d = mainwindow.chbEx_Raster_Scan_2D.isChecked()
    ex_plot_raster_scans = mainwindow.chbEx_Plot_Raster_Scans.isChecked()

    ex_df_levels = mainwindow.chbEx_DF_Levels.isChecked()
    ex_df_levels_lifetimes = mainwindow.chbEx_DF_Levels_Lifetimes.isChecked()
    ex_df_grouped_levels = mainwindow.chbEx_DF_Grouped_Levels.isChecked()
    ex_df_grouped_levels_lifetimes = mainwindow.chbEx_DF_Grouped_Levels_Lifetimes.isChecked()
    ex_df_grouping_info = mainwindow.chbEx_DF_Grouping_Info.isChecked()

    ex_df_format = mainwindow.cmbEx_DataFrame_Format.currentIndex()

    any_particle_text_plot = any([ex_traces, ex_levels, ex_plot_intensities, ex_grouped_levels,
                                  ex_grouping_info, ex_grouping_results, ex_plot_grouping_bics,
                                  ex_lifetime, ex_hist, ex_plot_lifetimes, ex_spectra_2d,
                                  ex_plot_spectra])
    if signals:
        prog_num = 0
        if any_particle_text_plot:
            prog_num = prog_num + len(particles)
        if ex_raster_scan_2d or ex_plot_raster_scans:
            prog_num = prog_num + len(raster_scans_use)
        if ex_df_levels:
            prog_num = prog_num + 1
        if ex_df_grouped_levels:
            prog_num = prog_num + 1
        if ex_df_grouping_info:
            prog_num = prog_num + 1

        signals.start_progress.emit(prog_num)
        signals.status_message.emit(f"Exporting data for {mode} particles...")

    # any_particle_text_plot = any([any_particle_text_plot, ex_raster_scan_2d, ex_plot_raster_scans])

    def open_file(path: str):
        return open(path, 'w', newline='')

    # Export fits of whole traces
    lifetime_path = os.path.join(f_dir, 'Whole trace lifetimes.csv')
    all_fitted = [part._histogram.fitted for part in particles]
    if ex_lifetime and any(all_fitted):
        p = particles[0]
        if p._histogram.numexp == 1:
            taucol = ['Lifetime (ns)']
            ampcol = ['Amp']
        elif p._histogram.numexp == 2:
            taucol = ['Lifetime 1 (ns)', 'Lifetime 2 (ns)']
            ampcol = ['Amp 1', 'Amp 2']
        elif p._histogram.numexp == 3:
            taucol = ['Lifetime 1 (ns)', 'Lifetime 2 (ns)', 'Lifetime 3 (ns)']
            ampcol = ['Amp 1', 'Amp 2', 'Amp 3']
        rows = list()
        rows.append(['Particle #'] + taucol + ampcol +
                    ['Av. Lifetime (ns)', 'IRF Shift (ns)', 'Decay BG', 'IRF BG',
                     'Chi Squared', 'Sim. IRF FWHM (ns)'])
        for i, p in enumerate(particles):
            if p._histogram.fitted:
                if p._histogram.tau is None or p._histogram.amp is None:  # Problem with fitting the level
                    tauexp = ['0' for i in range(p._histogram.numexp)]
                    ampexp = ['0' for i in range(p._histogram.numexp)]
                    other_exp = ['0', '0', '0', '0']
                else:
                    if p.numexp == 1:
                        tauexp = [str(p._histogram.tau)]
                        ampexp = [str(p._histogram.amp)]
                    else:
                        tauexp = [str(tau) for tau in p._histogram.tau]
                        ampexp = [str(amp) for amp in p._histogram.amp]
                    other_exp = [str(p._histogram.avtau), str(p._histogram.shift),
                                 str(p._histogram.bg), str(p._histogram.irfbg),
                                 str(p._histogram.chisq), str(p._histogram.fwhm)]
                rows.append([str(i + 1)] + tauexp + ampexp + other_exp)
        with open_file(lifetime_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    lifetime_path = os.path.join(f_dir, 'Whole trace lifetimes (ROI).csv')
    all_fitted = [part._histogram_roi.fitted for part in particles]
    if ex_lifetime and any(all_fitted):
        p = particles[0]
        if p._histogram_roi.numexp == 1:
            taucol = ['Lifetime (ns)']
            ampcol = ['Amp']
        elif p._histogram_roi.numexp == 2:
            taucol = ['Lifetime 1 (ns)', 'Lifetime 2 (ns)']
            ampcol = ['Amp 1', 'Amp 2']
        elif p._histogram_roi.numexp == 3:
            taucol = ['Lifetime 1 (ns)', 'Lifetime 2 (ns)', 'Lifetime 3 (ns)']
            ampcol = ['Amp 1', 'Amp 2', 'Amp 3']
        rows = list()
        rows.append(['Particle #'] + taucol + ampcol +
                    ['Av. Lifetime (ns)', 'IRF Shift (ns)', 'Decay BG', 'IRF BG',
                     'Chi Squared', 'Sim. IRF FWHM (ns)'])
        for i, p in enumerate(particles):
            if p._histogram_roi.fitted:
                if p._histogram_roi.tau is None or p._histogram_roi.amp is None:  # Problem with fitting the level
                    tauexp = ['0' for i in range(p._histogram_roi.numexp)]
                    ampexp = ['0' for i in range(p._histogram_roi.numexp)]
                    other_exp = ['0', '0', '0', '0']
                else:
                    if p.numexp == 1:
                        tauexp = [str(p._histogram_roi.tau)]
                        ampexp = [str(p._histogram_roi.amp)]
                    else:
                        tauexp = [str(tau) for tau in p._histogram_roi.tau]
                        ampexp = [str(amp) for amp in p._histogram_roi.amp]
                    other_exp = [str(p._histogram_roi.avtau), str(p._histogram_roi.shift),
                                 str(p._histogram_roi.bg), str(p._histogram_roi.irfbg),
                                 str(p._histogram_roi.chisq), str(p._histogram_roi.fwhm)]
                rows.append([str(i + 1)] + tauexp + ampexp + other_exp)
        with open_file(lifetime_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    # Export data for levels
    if any_particle_text_plot:
        for num, p in enumerate(particles):
            if ex_traces:
                tr_path = os.path.join(f_dir, p.name + ' trace.csv')
                ints = p.binnedtrace.intdata
                times = p.binnedtrace.inttimes / 1E3
                rows = list()
                rows.append(['Bin #', 'Bin Time (s)', f'Bin Int (counts/{p.bin_size}ms)'])
                for i in range(len(ints)):
                    rows.append([str(i + 1), str(times[i]), str(ints[i])])

                with open_file(tr_path) as f:
                    writer = csv.writer(f, dialect=csv.excel)
                    writer.writerows(rows)

                if use_roi:
                    tr_path = os.path.join(f_dir, p.name + ' trace (ROI).csv')
                    roi_filter = (p.roi_region[0] > times) ^ (times <= p.roi_region[1])
                    roi_ints = ints[roi_filter]
                    roi_times = times[roi_filter]
                    rows = list()
                    rows.append(['Bin #', 'Bin Time (s)', f'Bin Int (counts/{p.bin_size}ms)'])
                    for i in range(len(roi_ints)):
                        rows.append([str(i + 1), str(roi_times[i]), str(roi_ints[i])])

                    with open_file(tr_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

            if ex_plot_intensities and ex_plot_int_only:
                if signals:
                    signals.plot_trace_export_lock.emit(p, True, f_dir, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                else:
                    mainwindow.int_controller.plot_trace(particle=p,
                                                         for_export=True,
                                                         export_path=f_dir)

            if ex_levels:
                if p.has_levels:
                    lvl_tr_path = os.path.join(f_dir, p.name + ' levels-plot.csv')
                    ints, times = p.levels2data(use_grouped=False, use_roi=False)
                    rows = list()
                    rows.append(['Level #', 'Time (s)', 'Int (counts/s)'])
                    for i in range(len(ints)):
                        rows.append([str((i // 2) + 1), str(times[i]), str(ints[i])])
                    with open_file(lvl_tr_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)
                    if use_roi:
                        lvl_tr_path = os.path.join(f_dir, p.name + ' levels-plot (ROI).csv')
                        ints, times = p.levels2data(use_grouped=False, use_roi=use_roi)
                        rows = list()
                        rows.append(['Level #', 'Time (s)', 'Int (counts/s)'])
                        for i in range(len(ints)):
                            rows.append([str((i // 2) + 1), str(times[i]), str(ints[i])])
                        with open_file(lvl_tr_path) as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

                    lvl_path = os.path.join(f_dir, p.name + ' levels.csv')
                    rows = list()
                    rows.append(['Level #', 'Start Time (s)', 'End Time (s)', 'Dwell Time (/s)',
                                 'Int (counts/s)', 'Num of Photons'])
                    for i, l in enumerate(p.cpts.levels):
                        rows.append(
                            [str(i + 1), str(l.times_s[0]), str(l.times_s[1]),
                             str(l.dwell_time_s),
                             str(l.int_p_s), str(l.num_photons)])
                    with open_file(lvl_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

                    if use_roi:
                        lvl_path = os.path.join(f_dir, p.name + ' levels (ROI).csv')
                        rows = list()
                        rows.append(['Level #', 'Start Time (s)', 'End Time (s)', 'Dwell Time (/s)',
                                     'Int (counts/s)', 'Num of Photons'])
                        for i, l in enumerate(p.levels_roi):
                            rows.append(
                                [str(i + 1), str(l.times_s[0]), str(l.times_s[1]),
                                 str(l.dwell_time_s),
                                 str(l.int_p_s), str(l.num_photons)])
                        with open_file(lvl_path) as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

            if ex_plot_intensities and ex_plot_with_levels:
                if p.has_levels:
                    if signals:
                        signals.plot_trace_lock.emit(p, True, True)
                        lock.acquire()
                        while lock.locked():
                            sleep(0.1)
                        signals.plot_levels_export_lock.emit(p, True, f_dir, True)
                        lock.acquire()
                        while lock.locked():
                            sleep(0.1)
                    else:
                        mainwindow.int_controller.plot_trace(particle=p, for_export=True)
                        mainwindow.int_controller.plot_levels(particle=p,
                                                              for_export=True,
                                                              export_path=f_dir)

            if ex_grouped_levels:
                if p.has_groups:
                    grp_lvl_tr_path = os.path.join(f_dir, p.name + ' levels-grouped-plot')
                    if not p.grouped_with_roi:
                        grp_lvl_tr_path += '.csv'
                    else:
                        grp_lvl_tr_path += ' (ROI).csv'
                    ints, times = p.levels2data(use_grouped=True)
                    rows = list()
                    rows.append(['Grouped Level #', 'Time (s)', 'Int (counts/s)'])
                    for i in range(len(ints)):
                        rows.append([str((i // 2) + 1), str(times[i]), str(ints[i])])
                    with open_file(grp_lvl_tr_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

                    grp_lvl_path = os.path.join(f_dir, p.name + ' levels-grouped')
                    if not p.grouped_with_roi:
                        grp_lvl_path += '.csv'
                    else:
                        grp_lvl_path += ' (ROI).csv'
                    rows = list()
                    rows.append(['Grouped Level #', 'Start Time (s)', 'End Time (s)',
                                 'Dwell Time (/s)', 'Int (counts/s)', 'Num of Photons',
                                 'Group Index'])
                    for i, l in enumerate(p.group_levels):
                        rows.append(
                            [str(i + 1), str(l.times_s[0]), str(l.times_s[1]),
                             str(l.dwell_time_s),
                             str(l.int_p_s), str(l.num_photons), str(l.group_ind + 1)])
                    with open_file(grp_lvl_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

            if ex_plot_intensities and ex_plot_and_groups:
                if p.has_groups:
                    if signals:
                        signals.plot_trace_lock.emit(p, True, True)
                        lock.acquire()
                        while lock.locked():
                            sleep(0.1)
                        signals.plot_levels_lock.emit(p, True, True)
                        lock.acquire()
                        while lock.locked():
                            sleep(0.1)
                        signals.plot_group_bounds_export_lock.emit(p, True, f_dir, True)
                        lock.acquire()
                        while lock.locked():
                            sleep(0.1)
                    else:
                        mainwindow.int_controller.plot_trace(particle=p, for_export=True)
                        mainwindow.int_controller.plot_levels(particle=p, for_export=True)
                        mainwindow.int_controller.plot_group_bounds(particle=p,
                                                                    for_export=True,
                                                                    export_path=f_dir)

            if ex_grouping_info:
                if p.has_groups:
                    group_info_path = os.path.join(f_dir, p.name + ' groups-info')
                    if not p.grouped_with_roi:
                        group_info_path += '.csv'
                    else:
                        group_info_path += ' (ROI).csv'
                    with open_file(group_info_path) as f:
                        f.write(f"# of Groups:,{p.ahca.best_step.num_groups}\n")
                        if p.ahca.best_step_ind == p.ahca.selected_step_ind:
                            answer = 'TRUE'
                        else:
                            answer = 'FALSE'
                        f.write(f"Selected solution highest BIC value?,{answer}\n\n")

                        rows = list()
                        rows.append(['Group #', 'Int (counts/s)', 'Total Dwell Time (s)',
                                     '# of Levels', '# of Photons'])
                        for num, group in enumerate(p.ahca.selected_step.groups):
                            rows.append([str(num + 1), str(group.int_p_s),
                                         str(group.dwell_time_s), str(len(group.lvls)),
                                         str(group.num_photons)])
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

            if ex_grouping_results:
                if p.has_groups:
                    grouping_results_path = os.path.join(f_dir, p.name + ' grouping-results')
                    if not p.grouped_with_roi:
                        grouping_results_path += '.csv'
                    else:
                        grouping_results_path += ' (ROI).csv'
                    with open_file(grouping_results_path) as f:
                        f.write(f"# of Steps:,{p.ahca.num_steps}\n")
                        f.write(f"Step with highest BIC value:,{p.ahca.best_step.bic}\n")
                        f.write(f"Step selected:,{p.ahca.selected_step_ind}\n\n")

                        rows = list()
                        rows.append(['Step #', '# of Groups', 'BIC value'])
                        for num, step in enumerate(p.ahca.steps):
                            rows.append([str(num + 1), str(step.num_groups), str(step.bic)])

                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

            if ex_plot_grouping_bics:
                if signals:
                    signals.plot_grouping_bic_export_lock.emit(p, True, f_dir, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                else:
                    mainwindow.grouping_controller.plot_group_bic(particle=p,
                                                                  for_export=True,
                                                                  export_path=f_dir)

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

                all_fitted_lvls = [lvl.histogram.fitted for lvl in p.cpts.levels]
                if p.has_levels and any(all_fitted_lvls):
                    lvl_path = os.path.join(f_dir, p.name + ' levels-lifetimes.csv')
                    rows = list()
                    rows.append(['Level #', 'Start Time (s)', 'End Time (s)',
                                 'Dwell Time (/s)', 'Int (counts/s)',
                                 'Num of Photons', 'Num of Photons Used'] + taucol + ampcol +
                                ['Av. Lifetime (ns)', 'IRF Shift (ns)', 'Decay BG',
                                 'IRF BG', 'Chi Squared', 'Sim. IRF FWHM (ns)'])
                    for i, l in enumerate(p.cpts.levels):
                        if l.histogram.fitted:
                            # Problem with fitting the level
                            if l.histogram.tau is None or l.histogram.amp is None:
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
                                if hasattr(l.histogram, 'fwhm'):
                                    sim_irf_fwhm = str(l.histogram.fwhm)
                                else:
                                    sim_irf_fwhm = ''
                                other_exp = [str(l.histogram.avtau), str(l.histogram.shift),
                                             str(l.histogram.bg), str(l.histogram.irfbg),
                                             str(l.histogram.chisq), sim_irf_fwhm]

                            rows.append([str(i + 1), str(l.times_s[0]), str(l.times_s[1]),
                                         str(l.dwell_time_s), str(l.int_p_s),
                                         str(l.num_photons), str(l.histogram.num_photons_used)]
                                        + tauexp + ampexp + other_exp)
                    with open_file(lvl_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

                    all_fitted_grps = [grp.histogram.fitted for grp in p.groups]
                    if p.has_groups and any(all_fitted_grps):
                        group_path = os.path.join(f_dir, p.name + ' groups-lifetimes')
                        if not p.grouped_with_roi:
                            group_path += '.csv'
                        else:
                            group_path += ' (ROI).csv'
                        rows = list()
                        rows.append(['Group #', 'Dwell Time (/s)',
                                     'Int (counts/s)', 'Num of Photons', 'Num of Photons Used']
                                    + taucol + ampcol +
                                    ['Av. Lifetime (ns)', 'IRF Shift (ns)', 'Decay BG', 'IRF BG',
                                     'Chi Squared', 'Sim. IRF FWHM'])
                        for i, g in enumerate(p.groups):
                            if g.histogram.fitted:
                                # Problem with fitting the level
                                if g.histogram.tau is None or g.histogram.amp is None:
                                    tauexp = ['0' for i in range(p.numexp)]
                                    ampexp = ['0' for i in range(p.numexp)]
                                    other_exp = ['0', '0', '0', '0']
                                else:
                                    if p.numexp == 1:
                                        tauexp = [str(g.histogram.tau)]
                                        ampexp = [str(g.histogram.amp)]
                                    else:
                                        tauexp = [str(tau) for tau in g.histogram.tau]
                                        ampexp = [str(amp) for amp in g.histogram.amp]
                                    other_exp = [str(g.histogram.avtau), str(g.histogram.shift),
                                                 str(g.histogram.bg), str(g.histogram.irfbg),
                                                 str(g.histogram.chisq), str(g.histogram.fwhm)]

                                rows.append(
                                    [str(i + 1), str(g.dwell_time_s), str(g.int_p_s),
                                     str(g.num_photons), str(g.histogram.num_photons_used)]
                                    + tauexp + ampexp + other_exp)
                    with open_file(group_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

                if use_roi:
                    all_fitted_lvls_roi = [lvl.histogram.fitted for lvl in p.levels_roi]
                    if p.has_levels and any(all_fitted_lvls_roi):
                        lvl_path = os.path.join(f_dir, p.name + ' levels-lifetimes (ROI).csv')
                        rows = list()
                        rows.append(['Level #', 'Start Time (s)', 'End Time (s)',
                                     'Dwell Time (/s)', 'Int (counts/s)',
                                     'Num of Photons', 'Num of Photons Used'] + taucol + ampcol +
                                    ['Av. Lifetime (ns)', 'IRF Shift (ns)', 'Decay BG',
                                     'IRF BG', 'Chi Squared', 'Sim. IRF FWHM (ns)'])
                        for i, l in enumerate(p.levels_roi):
                            if l.histogram.fitted:
                                # Problem with fitting the level
                                if l.histogram.tau is None or l.histogram.amp is None:
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
                                    if hasattr(l.histogram, 'fwhm'):
                                        sim_irf_fwhm = str(l.histogram.fwhm)
                                    else:
                                        sim_irf_fwhm = ''
                                    other_exp = [str(l.histogram.avtau), str(l.histogram.shift),
                                                 str(l.histogram.bg), str(l.histogram.irfbg),
                                                 str(l.histogram.chisq), sim_irf_fwhm]

                                rows.append([str(i + 1), str(l.times_s[0]), str(l.times_s[1]),
                                             str(l.dwell_time_s), str(l.int_p_s),
                                             str(l.num_photons), str(l.histogram.num_photons_used)]
                                            + tauexp + ampexp + other_exp)
                        with open_file(lvl_path) as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

            if ex_hist:
                tr_path = os.path.join(f_dir, p.name + ' hist.csv')
                if not p._histogram.fitted:
                    decay = p._histogram.decay
                    times = p._histogram.t
                    rows = list()
                    rows.append(['Times (ns)', 'Decay'])
                    for i, time in enumerate(times):
                        rows.append([str(time), str(decay[i])])
                    with open_file(tr_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)
                else:
                    times = p._histogram.convd_t
                    if times is not None:
                        decay = p._histogram.fit_decay
                        convd = p._histogram.convd
                        residual = p._histogram.residuals
                        rows = list()
                        rows.append(['Time (ns)', 'Decay', 'Fitted', 'Residual'])
                        for i, time in enumerate(times):
                            rows.append([str(time), str(decay[i]), str(convd[i]), str(residual[i])])

                        with open_file(tr_path) as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

                if use_roi:
                    tr_path = os.path.join(f_dir, p.name + ' hist (ROI).csv')
                    if not p._histogram_roi.fitted:
                        decay = p._histogram_roi.decay
                        times = p._histogram_roi.t
                        rows = list()
                        rows.append(['Times (ns)', 'Decay'])
                        for i, time in enumerate(times):
                            rows.append([str(time), str(decay[i])])
                        with open_file(tr_path) as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)
                    else:
                        times = p._histogram_roi.convd_t
                        if times is not None:
                            decay = p._histogram_roi.fit_decay
                            convd = p._histogram_roi.convd
                            residual = p._histogram_roi.residuals
                            rows = list()
                            rows.append(['Time (ns)', 'Decay', 'Fitted', 'Residual'])
                            for i, time in enumerate(times):
                                rows.append([str(time), str(decay[i]), str(convd[i]), str(residual[i])])

                            with open_file(tr_path) as f:
                                writer = csv.writer(f, dialect=csv.excel)
                                writer.writerows(rows)

                if p.has_levels:
                    dir_path = os.path.join(f_dir, p.name + ' hists')
                    try:
                        os.mkdir(dir_path)
                    except FileExistsError:
                        pass
                    # if not p.using_group_levels:
                    #     roi_start_ind = p.first_level_ind_in_roi
                    #     roi_end_ind = p.last_level_ind_in_roi
                    # else:
                    #     roi_start_ind = p.first_group_level_ind_in_roi
                    #     roi_end_ind = p.last_group_level_ind_in_roi
                    roi_start_ind = p.first_level_ind_in_roi
                    roi_end_ind = p.last_level_ind_in_roi
                    for i, l in enumerate(p.cpts.levels):
                        roi_tag = ' (ROI)' if use_roi and roi_start_ind <= i <= roi_end_ind else ''
                        hist_path = os.path.join(dir_path,
                                                 'level ' + str(i + 1) + roi_tag + ' hist.csv')
                        times = l.histogram.convd_t
                        if times is None:
                            continue
                        decay = l.histogram.fit_decay
                        convd = l.histogram.convd
                        residual = l.histogram.residuals
                        rows = list()
                        rows.append(['Time (ns)', 'Decay', 'Fitted', 'Residual'])
                        for j, time in enumerate(times):
                            rows.append([str(time), str(decay[j]), str(convd[j]), str(residual[j])])

                        with open_file(hist_path) as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

                    if p.has_groups:
                        roi_start_ind = p.first_group_level_ind_in_roi
                        roi_end_ind = p.last_group_level_ind_in_roi
                        for i, g in enumerate(p.group_levels):
                            roi_tag = ' (ROI)' if use_roi and roi_start_ind <= i <= roi_end_ind else ''
                            hist_path = os.path.join(dir_path,
                                                     'group level ' + str(i + 1) + roi_tag + ' hist.csv')
                            times = g.histogram.convd_t
                            if times is None:
                                continue
                            decay = g.histogram.fit_decay
                            convd = g.histogram.convd
                            residual = g.histogram.residuals
                            rows = list()
                            rows.append(['Time (ns)', 'Decay', 'Fitted', 'Residual'])
                            for j, time in enumerate(times):
                                rows.append([str(time), str(decay[j]), str(convd[j]),
                                             str(residual[j])])

                            with open_file(hist_path) as f:
                                writer = csv.writer(f, dialect=csv.excel)
                                writer.writerows(rows)

                        for i, g in enumerate(p.groups):
                            hist_path = os.path.join(dir_path,
                                                     'group ' + str(i + 1) + ' hist.csv')
                            times = g.histogram.convd_t
                            if times is None:
                                continue
                            decay = g.histogram.fit_decay
                            convd = g.histogram.convd
                            residual = g.histogram.residuals
                            rows = list()
                            rows.append(['Time (ns)', 'Decay', 'Fitted', 'Residual'])
                            for j, time in enumerate(times):
                                rows.append([str(time), str(decay[j]), str(convd[j]),
                                             str(residual[j])])

                            with open_file(hist_path) as f:
                                writer = csv.writer(f, dialect=csv.excel)
                                writer.writerows(rows)

            # TODO: Fix problems
            # Current problems
            # 1. When use_roi is off, only levels in roi print
            # 2. Hist only looks good, but all other options go into E-14 range in y. Looks bad
            # 3. Some plots are empty when fit is included
            if ex_plot_lifetimes and ex_plot_hist_only:
                if signals:
                    signals.plot_decay_export_lock.emit(-1, p, False, True, f_dir, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                else:
                    mainwindow.lifetime_controller.plot_decay(select_ind=-1, particle=p,
                                                              for_export=True,
                                                              export_path=f_dir)
                dir_path = os.path.join(f_dir, p.name + ' hists')
                try:
                    os.mkdir(dir_path)
                except FileExistsError:
                    pass
                if p.has_levels:
                    if not ex_plot_lifetimes_only_groups:
                        for i in range(p.num_levels):
                            if signals:
                                signals.plot_decay_export_lock.emit(i, p, False, True, dir_path, True)
                                lock.acquire()
                                while lock.locked():
                                    sleep(0.1)
                            else:
                                mainwindow.lifetime_controller.plot_decay(select_ind=i,
                                                                          particle=p,
                                                                          for_export=True,
                                                                          export_path=dir_path)
                    if p.has_groups:
                        for i in range(p.num_groups):
                            i_g = i + p.num_levels
                            if signals:
                                signals.plot_decay_export_lock.emit(i_g, p, False, True, dir_path,
                                                                    True)
                                lock.acquire()
                                while lock.locked():
                                    sleep(0.1)
                            else:
                                mainwindow.lifetime_controller.plot_decay(select_ind=i_g,
                                                                          particle=p,
                                                                          for_export=True,
                                                                          export_path=dir_path)

            if ex_plot_lifetimes and ex_plot_with_fit:
                if signals:
                    signals.plot_decay_lock.emit(-1, p, False, True, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                    signals.plot_convd_export_lock.emit(-1, p, False, True, f_dir, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                else:
                    mainwindow.lifetime_controller.plot_decay(select_ind=-1,
                                                              particle=p, for_export=True)
                    mainwindow.lifetime_controller.plot_convd(select_ind=-1,
                                                              particle=p, for_export=True,
                                                              export_path=f_dir)
                dir_path = os.path.join(f_dir, p.name + ' hists')
                try:
                    os.mkdir(dir_path)
                except FileExistsError:
                    pass
                if p.has_levels:
                    signals.plot_decay_convd_export_lock.emit(p, dir_path, p.has_groups,
                                                              ex_plot_lifetimes_only_groups, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)

            if ex_plot_and_residuals:
                if signals:
                    signals.plot_decay_lock.emit(-1, p, False, True, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                    signals.plot_convd_lock.emit(-1, p, False, True, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                    signals.plot_residuals_export_lock.emit(-1, p, True, f_dir, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                else:
                    mainwindow.lifetime_controller.plot_decay(select_ind=-1,
                                                              particle=p, for_export=True)
                    mainwindow.lifetime_controller.plot_convd(select_ind=-1,
                                                              particle=p, for_export=True,
                                                              export_path=f_dir)
                dir_path = os.path.join(f_dir, p.name + ' hists')
                try:
                    os.mkdir(dir_path)
                except FileExistsError:
                    pass
                if p.has_levels:
                    signals.plot_decay_convd_residuals_export_lock\
                        .emit(p, dir_path, p.has_groups, ex_plot_lifetimes_only_groups, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)

            if ex_spectra_2d:
                spectra_2d_path = os.path.join(f_dir, p.name + ' spectra-2D.csv')
                with open_file(spectra_2d_path) as f:
                    f.write("First row:,Wavelength (nm)\n")
                    f.write("First column:,Time (s)\n")
                    f.write("Values:,Intensity (counts/s)\n\n")

                    rows = list()
                    rows.append([''] + p.spectra.wavelengths.tolist())
                    for num, spec_row in enumerate(p.spectra.data[:]):
                        this_row = list()
                        this_row.append(str(p.spectra.series_times[num]))
                        for single_val in spec_row:
                            this_row.append(str(single_val))
                        rows.append(this_row)

                    writer = csv.writer(f, dialect=csv.excel)
                    writer.writerows(rows)

            if ex_plot_spectra:
                if signals:
                    signals.plot_spectra_export_lock.emit(p, True, f_dir, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                else:
                    mainwindow.spectra_controller.plot_spectra(particle=p, for_export=True,
                                                               export_path=f_dir)

            logger.info('Exporting Finished')
            if signals:
                signals.progress.emit()
            p.has_exported = True

        # if ex_raster_scan_2d:
        #     dataset = mainwindow.current_dataset
        #     for raster_scan_index in raster_scans_use:
        #         raster_scan = dataset.all_raster_scans[raster_scan_index]
        #         if signals:
        #             signals.progress.emit()

    if ex_raster_scan_2d or ex_plot_raster_scans:
        dataset = mainwindow.current_dataset
        for raster_scan_index in raster_scans_use:
            raster_scan = dataset.all_raster_scans[raster_scan_index]
            rs_part_ind = raster_scan.particle_indexes[0]
            p = dataset.particles[rs_part_ind]
            if ex_raster_scan_2d:
                raster_scan_2d_path = \
                    os.path.join(f_dir, f"Raster Scan {raster_scan.dataset_index + 1} data.csv")
                top_row = [np.NaN, *raster_scan.x_axis_pos]
                y_and_data = np.column_stack((raster_scan.y_axis_pos, raster_scan.dataset[:]))
                x_y_data = np.insert(y_and_data, 0, top_row, axis=0)
                with open_file(raster_scan_2d_path) as f:
                    f.write('Rows = X-Axis (um)')
                    f.write('Columns = Y-Axis (um)')
                    f.write('')
                    writer = csv.writer(f, dialect=csv.excel)
                    writer.writerows(x_y_data)

            if ex_plot_raster_scans:
                if signals:
                    # with lock:
                    signals.plot_raster_scan_export_lock.emit(p, raster_scan, True, f_dir, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                mainwindow.raster_scan_controller.plot_raster_scan(
                    raster_scan=raster_scan, for_export=True,
                    export_path=f_dir)

            if signals:
                signals.progress.emit()

    ## DataFrame compilation and writing
    if any([ex_df_levels, ex_df_grouped_levels, ex_df_grouping_info]):
        any_has_lifetime = any([p.has_fit_a_lifetime for p in particles])
        if any_has_lifetime:
            max_numexp = max([p.numexp for p in particles])
            tau_cols = [f'tau_{i + 1}' for i in range(max_numexp)]
            amp_cols = [f'amp_{i + 1}' for i in range(max_numexp)]
            life_cols_add = ['num_photons_in_lifetime_fit', *tau_cols, *amp_cols,
                             'irf_shift', 'decay_bg', 'irf_bg',
                             'chi_squared', 'dw', 'dw_5', 'dw_1', 'dw_03', 'dw_01']
        else:
            life_cols_add = ['']
            max_numexp = None
        if ex_df_levels or ex_df_grouped_levels:
            levels_cols = ['particle', 'level', 'start', 'end', 'dwell', 'dwell_frac', 'int',
                           'num_photons']
            grouped_levels_cols = levels_cols.copy()
            # grouped_levels_cols[1] = 'grouped_level'
            grouped_levels_cols.insert(2, 'group_index')
            if any_has_lifetime:
                if ex_df_levels_lifetimes:
                    levels_cols.extend(life_cols_add)
                if ex_df_grouped_levels_lifetimes:
                    grouped_levels_cols.extend(life_cols_add)
            levels_cols.append('is_in_roi')
            grouped_levels_cols.append('is_in_roi')

            data_levels = list()
            if ex_df_grouped_levels:
                data_grouped_levels = list()

        if ex_df_grouping_info:
            grouping_info_cols = ['particle', 'group', 'total_dwell', 'int', 'num_levels',
                                  'num_photons', 'num_steps', 'is_best_step']
            data_grouping_info = list()

        for p in particles:
            # print(p.name)
            roi_first_level_ind = p.first_level_ind_in_roi
            roi_last_level_ind = p.last_level_ind_in_roi
            if ex_df_levels:
                for l_num, l in enumerate(p.cpts.levels):
                    level_in_roi = roi_first_level_ind <= l_num <= roi_last_level_ind
                    row = [p.name,
                           l_num + 1,
                           *get_level_data(l, p.dwell_time,
                                           incl_lifetimes=all([ex_df_levels_lifetimes,
                                                               p.has_fit_a_lifetime]),
                                           max_numexp=max_numexp),
                           level_in_roi]
                    data_levels.append(row)

            if ex_df_grouped_levels:
                roi_first_group_level_ind = p.first_group_level_ind_in_roi
                roi_last_group_level_ind = p.last_group_level_ind_in_roi
                for g_l_num, g_l in enumerate(p.group_levels):
                    group_level_in_roi = roi_first_group_level_ind <= g_l_num <= roi_last_group_level_ind
                    row = [p.name, g_l_num + 1, g_l.group_ind + 1,
                           *get_level_data(g_l,
                                           p.dwell_time,
                                           incl_lifetimes=all([ex_df_grouped_levels_lifetimes,
                                                               p.has_fit_a_lifetime]),
                                           max_numexp=max_numexp),
                           group_level_in_roi]
                    data_grouped_levels.append(row)

            if ex_df_grouping_info:
                if p.has_groups:
                    for g_num, g in enumerate(p.ahca.selected_step.groups):
                        row = [p.name, g_num + 1, g.int_p_s, g.dwell_time_s, len(g.lvls),
                               g.num_photons, p.ahca.num_steps,
                               p.ahca.selected_step == p.ahca.best_step_ind]
                        data_grouping_info.append(row)
                else:
                    row = [p.name]
                    row.extend([np.NaN]*7)
                    data_grouping_info.append(row)

        if ex_df_levels:
            df_levels = pd.DataFrame(data=data_levels, columns=levels_cols)
            df_levels['particle'] = df_levels['particle'].astype('string')
            # levels_df_path = os.path.join(f_dir, 'levels.df')
            # feather.write_feather(df=df_levels, dest=levels_df_path)
            write_dataframe_to_file(dataframe=df_levels, path=f_dir, filename='levels',
                                    file_type=ex_df_format)
            if signals:
                signals.progress.emit()

        if ex_df_grouped_levels:
            df_grouped_levels = pd.DataFrame(data=data_grouped_levels, columns=grouped_levels_cols)
            df_grouped_levels['particle'] = df_grouped_levels.particle.astype('string')
            # grouped_levels_df_path = os.path.join(f_dir, 'grouped_levels.df')
            # feather.write_feather(df=df_grouped_levels, dest=grouped_levels_df_path)
            write_dataframe_to_file(dataframe=df_grouped_levels, path=f_dir,
                                    filename='grouped_levels',
                                    file_type=ex_df_format)
            if signals:
                signals.progress.emit()

        if ex_df_grouping_info:
            df_grouping_info = pd.DataFrame(data=data_grouping_info, columns=grouping_info_cols)
            # grouping_info_df_path = os.path.join(f_dir, 'grouping_info.df')
            # feather.write_feather(df=df_grouping_info, dest=grouping_info_df_path)
            write_dataframe_to_file(dataframe=df_grouping_info, path=f_dir, filename='grouping_info',
                                    file_type=ex_df_format)
            if signals:
                signals.progress.emit()

    if signals:
        signals.end_progress.emit()
        signals.status_message.emit("Done")


def write_dataframe_to_file(dataframe: pd.DataFrame, path: str, filename: str, file_type: dict):
    # DATAFRAME_FORMATS = {'Parquet (*.parquet)': 0,
    #                      'Feather (*.ftr)': 1,
    #                      'Feather (*.df)': 2,
    #                      'Pickle (*.pkl)': 3,
    #                      'HDF (*.h5)': 4,
    #                      'Excel (*.xlsx)': 5,
    #                      'CSV (*.csv)': 6}
    if file_type == 0:  # Parquet
        file_path = os.path.join(path, filename + '.parquet')
        dataframe.to_parquet(path=file_path)
    elif file_type == 1 or file_type == 2:  # Feather
        if file_type == 1:  # with .ftr
            file_path = os.path.join(path, filename + '.ftr')
        else:
            file_path = os.path.join(path, filename + '.df')
        feather.write_feather(df=dataframe, dest=file_path)
    elif file_type == 3:  # Pickle
        file_path = os.path.join(path, filename + '.pkl')
        dataframe.to_pickle(path=file_path)
    elif file_type == 4:  # HDF
        file_path = os.path.join(path, filename + '.h5')
        dataframe.to_hdf(path_or_buf=file_path, key=filename, format='table')
    elif file_type == 5:  # Excel
        file_path = os.path.join(path, filename + '.xlsx')
        dataframe.to_excel(file_path)
    elif file_type == 6:  # CSV
        file_path = os.path.join(path, filename + '.csv')
        dataframe.to_csv(file_path)
    else:
        logger.error("File type not configured yet")
    pass


def get_level_data(level: Level, total_dwelltime: float,
                   incl_lifetimes: bool = False, max_numexp: int = 3) -> List:
    data = [*level.times_s, level.dwell_time_s, level.dwell_time_s/total_dwelltime, level.int_p_s,
            level.num_photons]
    if incl_lifetimes:
        h = level.histogram
        if h.fitted:
            data.append(h.num_photons_used)
            if h.numexp == 1:
                taus = [h.tau] if type(h.tau) is not list else h.tau
                amps = [h.amp]
            else:
                taus = list(h.tau)
                amps = list(h.amp)

            taus.extend([np.NaN] * (max_numexp - h.numexp))
            data.extend(taus)

            amps.extend([np.NaN] * (max_numexp - h.numexp))
            data.extend(amps)

            if h.dw_bound is None:
                h.dw_bound = [None, None, None, None]
            data.extend([h.shift, h.bg, h.irfbg, h.chisq, h.dw, h.dw_bound[0],
                        h.dw_bound[1], h.dw_bound[2], h.dw_bound[3]])
        else:
            data.extend([np.NaN]*(9 + max_numexp))

    return data
