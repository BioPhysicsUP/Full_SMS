from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, List

from PyQt5.QtWidgets import QFileDialog
import numpy as np
import pandas as pd
from pyarrow import feather

from my_logger import setup_logger
from PyQt5.QtCore import QRunnable, pyqtSlot
from signals import WorkerSignals
from multiprocessing import Lock

if TYPE_CHECKING:
    from main import MainWindow
    from change_point import Level
    from grouping import Group

logger = setup_logger(__name__)


class ExportWorker(QRunnable):
    def __init__(self, mainwindow: MainWindow, mode: str = None):
        super(ExportWorker, self).__init__()
        self.main_window = mainwindow
        self.mode = mode
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            export_data(mainwindow=self.main_window, mode=self.mode, signals=self.signals)
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.fitting_finished.emit(self.mode)


def export_data(mainwindow: MainWindow, mode: str = None, signals: WorkerSignals = None):
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

    lock = Lock()

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
    ex_spectra_2d = mainwindow.chbEx_Spectra_2D.isChecked()
    ex_plot_spectra = mainwindow.chbEx_Plot_Spectra.isChecked()
    ex_raster_scan_2d = mainwindow.chbEx_Raster_Scan_2D.isChecked()
    ex_plot_raster_scans = mainwindow.chbEx_Plot_Raster_Scans.isChecked()

    ex_df_levels = mainwindow.chbEx_DF_Levels.isChecked()
    ex_df_levels_lifetimes = mainwindow.chbEx_DF_Levels_Lifetimes.isChecked()
    ex_df_grouped_levels = mainwindow.chbEx_DF_Grouped_Levels.isChecked()
    ex_df_grouped_levels_lifetimes = mainwindow.chbEx_DF_Grouped_Levels_Lifetimes.isChecked()
    ex_df_grouping_info = mainwindow.chbEx_DF_Grouping_Info.isChecked()

    any_text_plot = any([ex_traces, ex_levels, ex_plot_intensities, ex_grouped_levels,
                         ex_grouping_info, ex_grouping_results, ex_plot_grouping_bics,
                         ex_lifetime, ex_hist, ex_plot_lifetimes, ex_plot_spectra,
                         ex_plot_spectra])
    if signals:
        prog_num = 0
        if any_text_plot:
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

    any_text_plot = any([any_text_plot, ex_raster_scan_2d, ex_plot_raster_scans])

    def open_file(path: str):
        return open(path, 'w', newline='')

    # Export fits of whole traces
    all_fitted = [part.histogram.fitted for part in particles]
    if ex_lifetime and any(all_fitted):
        p = particles[0]
        if p.numexp == 1:
            taucol = ['Lifetime (ns)']
            ampcol = ['Amp']
        elif p.numexp == 2:
            taucol = ['Lifetime 1 (ns)', 'Lifetime 2 (ns)']
            ampcol = ['Amp 1', 'Amp 2']
        elif p.numexp == 3:
            taucol = ['Lifetime 1 (ns)', 'Lifetime 2 (ns)', 'Lifetime 3 (ns)']
            ampcol = ['Amp 1', 'Amp 2', 'Amp 3']
        lifetime_path = os.path.join(f_dir, 'Whole trace lifetimes.csv')
        rows = list()
        rows.append(['Particle #'] + taucol + ampcol +
                    ['Av. Lifetime (ns)', 'IRF Shift (ns)', 'Decay BG', 'IRF BG',
                     'Chi Squared'])
        for i, p in enumerate(particles):
            if p.histogram.fitted:
                if p.histogram.tau is None or p.histogram.amp is None:  # Problem with fitting the level
                    tauexp = ['0' for i in range(p.numexp)]
                    ampexp = ['0' for i in range(p.numexp)]
                    other_exp = ['0', '0', '0', '0']
                else:
                    if p.numexp == 1:
                        tauexp = [str(p.histogram.tau)]
                        ampexp = [str(p.histogram.amp)]
                    else:
                        tauexp = [str(tau) for tau in p.histogram.tau]
                        ampexp = [str(amp) for amp in p.histogram.amp]
                    other_exp = [str(p.histogram.avtau), str(p.histogram.shift),
                                 str(p.histogram.bg),
                                 str(p.histogram.irfbg), str(p.histogram.chisq)]

                rows.append([str(i + 1)] + tauexp + ampexp + other_exp)

        with open_file(lifetime_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    # Export data for levels
    if any_text_plot:
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

            if ex_plot_intensities and ex_plot_int_only:
                if signals:
                    signals.plot_trace_export_lock.emit(p, True, f_dir, lock)
                    lock.acquire()
                else:
                    mainwindow.int_controller.plot_trace(particle=p,
                                                         for_export=True,
                                                         export_path=f_dir)

            if ex_levels:
                if p.has_levels:
                    lvl_tr_path = os.path.join(f_dir, p.name + ' levels-plot.csv')
                    ints, times = p.levels2data(use_grouped=False)
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

            if ex_plot_intensities and ex_plot_with_levels:
                if p.has_levels:
                    if signals:
                        signals.plot_trace_lock.emit(p, True, lock)
                        lock.acquire()
                        signals.plot_levels_export_lock.emit(p, True, f_dir, lock)
                        lock.acquire()
                    else:
                        mainwindow.int_controller.plot_trace(particle=p, for_export=True)
                        mainwindow.int_controller.plot_levels(particle=p,
                                                              for_export=True,
                                                              export_path=f_dir)

            if ex_grouped_levels:
                if p.has_groups:
                    grp_lvl_tr_path = os.path.join(f_dir, p.name + ' levels-grouped-plot.csv')
                    ints, times = p.levels2data(use_grouped=True)
                    rows = list()
                    rows.append(['Grouped Level #', 'Time (s)', 'Int (counts/s)'])
                    for i in range(len(ints)):
                        rows.append([str((i // 2) + 1), str(times[i]), str(ints[i])])

                    with open_file(grp_lvl_tr_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

                    grp_lvl_path = os.path.join(f_dir, p.name + ' levels-grouped.csv')
                    rows = list()
                    rows.append(['Grouped Level #', 'Start Time (s)', 'End Time (s)',
                                 'Dwell Time (/s)', 'Int (counts/s)', 'Num of Photons',
                                 'Group Index'])
                    for i, l in enumerate(p.ahca.selected_step.group_levels):
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
                        signals.plot_trace_lock.emit(p, True, lock)
                        lock.acquire()
                        signals.plot_levels_lock.emit(p, True, lock)
                        lock.acquire()
                        signals.plot_group_bounds_export_lock.emit(p, True, f_dir, lock)
                        lock.acquire()
                    else:
                        mainwindow.int_controller.plot_trace(particle=p, for_export=True)
                        mainwindow.int_controller.plot_levels(particle=p, for_export=True)
                        mainwindow.int_controller.plot_group_bounds(particle=p,
                                                                    for_export=True,
                                                                    export_path=f_dir)

            if ex_grouping_info:
                if p.has_groups:
                    group_info_path = os.path.join(f_dir, p.name + ' groups-info.csv')
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
                    group_info_path = os.path.join(f_dir, p.name + ' grouping-results.csv')
                    with open_file(group_info_path) as f:
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
                    signals.plot_grouping_bic_export_lock.emit(p, True, f_dir, lock)
                    lock.acquire()
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

                all_fitted_lvls = [lvl.histogram.fitted for lvl in p.levels]
                if p.has_levels and any(all_fitted_lvls):
                    lvl_path = os.path.join(f_dir, p.name + ' levels-lifetimes.csv')
                    rows = list()
                    rows.append(['Level #', 'Start Time (s)', 'End Time (s)',
                                 'Dwell Time (/s)', 'Int (counts/s)',
                                 'Num of Photons'] + taucol + ampcol +
                                ['Av. Lifetime (ns)', 'IRF Shift (ns)', 'Decay BG',
                                 'IRF BG', 'Chi Squared'])
                    for i, l in enumerate(p.levels):
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
                                other_exp = [str(l.histogram.avtau), str(l.histogram.shift),
                                             str(l.histogram.bg),
                                             str(l.histogram.irfbg), str(l.histogram.chisq)]

                            rows.append([str(i + 1), str(l.times_s[0]), str(l.times_s[1]),
                                         str(l.dwell_time_s), str(l.int_p_s),
                                         str(l.num_photons)] + tauexp + ampexp + other_exp)

                    with open_file(lvl_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

                    all_fitted_grps = [grp.histogram.fitted for grp in p.groups]
                    if p.has_groups and any(all_fitted_grps):
                        group_path = os.path.join(f_dir, p.name + ' groups-lifetimes.csv')
                        rows = list()
                        rows.append(['Group #', 'Dwell Time (/s)',
                                     'Int (counts/s)', 'Num of Photons'] + taucol + ampcol +
                                    ['Av. Lifetime (ns)', 'IRF Shift (ns)', 'Decay BG', 'IRF BG',
                                     'Chi Squared'])
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
                                                 str(g.histogram.bg),
                                                 str(g.histogram.irfbg), str(g.histogram.chisq)]

                                rows.append(
                                    [str(i + 1), str(g.dwell_time_s), str(g.int_p_s),
                                     str(g.num_photons)] + tauexp + ampexp + other_exp)

                    with open_file(group_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)

            if ex_hist:
                tr_path = os.path.join(f_dir, p.name + ' hist.csv')
                if not p.histogram.fitted:
                    decay = p.histogram.decay
                    times = p.histogram.t
                    rows = ['Times (ns)', 'Decay']
                    for i, time in enumerate(times):
                        rows.append([str(time), str(decay[i])])
                    with open_file(tr_path) as f:
                        writer = csv.writer(f, dialect=csv.excel)
                        writer.writerows(rows)
                else:
                    times = p.histogram.convd_t
                    if times is not None:
                        decay = p.histogram.fit_decay
                        convd = p.histogram.convd
                        residual = p.histogram.residuals
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
                    for i, l in enumerate(p.levels):
                        hist_path = os.path.join(dir_path,
                                                 'level ' + str(i + 1) + ' hist.csv')
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

            if ex_plot_lifetimes and ex_plot_hist_only:
                if signals:
                    signals.plot_decay_export_lock.emit(-1, p, False, True, f_dir, lock)
                    lock.acquire()
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
                    for i in range(p.num_levels):
                        if signals:
                            signals.plot_decay_export_lock.emit(i, p, False, True, dir_path, lock)
                            lock.acquire()
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
                                                                    lock)
                                lock.acquire()
                            else:
                                mainwindow.lifetime_controller.plot_decay(select_ind=i_g,
                                                                          particle=p,
                                                                          for_export=True,
                                                                          export_path=dir_path)

            if ex_plot_and_residuals or (ex_plot_lifetimes and ex_plot_with_fit):
                if signals:
                    signals.plot_decay_lock.emit(-1, p, False, True, lock)
                    lock.acquire()
                    signals.plot_convd_export_lock.emit(-1, p, False, True, f_dir, lock)
                    lock.acquire()
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
                    signals.plot_decay_convd_export_lock.emit(p, dir_path, p.has_groups, lock)
                    lock.acquire()

            if ex_plot_and_residuals:
                was_showing = mainwindow.chbShow_Residuals.isChecked()
                if not was_showing:
                    if signals:
                        signals.show_residual_widget_lock.emit(True, lock)
                        lock.acquire()
                    else:
                        mainwindow.lifetime_controller.residual_widget.show()
                if signals:
                    signals.plot_residuals_export_lock.emit(-1, p, True, f_dir, lock)
                    lock.acquire()
                else:
                    mainwindow.lifetime_controller.plot_residuals(select_ind=-1, particle=p,
                                                                  for_export=True,
                                                                  export_path=f_dir)
                dir_path = os.path.join(f_dir, p.name + ' hists')
                try:
                    os.mkdir(dir_path)
                except FileExistsError:
                    pass
                if p.has_levels:
                    for i in range(p.num_levels):
                        if signals:
                            signals.plot_residuals_export_lock.emit(i, p, True, dir_path, lock)
                            lock.acquire()
                        else:
                            mainwindow.lifetime_controller.plot_residuals(select_ind=i,
                                                                          particle=p,
                                                                          for_export=True,
                                                                          export_path=dir_path)
                    if p.has_groups:
                        for i in range(p.num_groups):
                            i_g = i + p.num_levels
                            if signals:
                                signals.plot_residuals_export_lock.emit(i_g, p, True, dir_path,
                                                                        lock)
                                lock.acquire()
                            else:
                                mainwindow.lifetime_controller.plot_residuals(select_ind=i_g,
                                                                              particle=p,
                                                                              for_export=True,
                                                                              export_path=dir_path)
                if not was_showing:
                    if signals:
                        signals.show_residual_widget.emit(False)
                    mainwindow.lifetime_controller.residual_widget.hide()

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
                    signals.plot_spectra_export_lock.emit(p, True, f_dir, lock)
                    lock.acquire()
                else:
                    mainwindow.spectra_controller.plot_spectra(particle=p, for_export=True,
                                                               export_path=f_dir)

            logger.info('Exporting Finished')
            if signals:
                signals.progress.emit()
            p.has_exported = True

        if ex_raster_scan_2d:
            dataset = mainwindow.current_dataset
            for raster_scan_index in raster_scans_use:
                raster_scan = dataset.all_raster_scans[raster_scan_index]
                if signals:
                    signals.progress.emit()

        if ex_raster_scan_2d or ex_plot_raster_scans:
            dataset = mainwindow.current_dataset
            for raster_scan_index in raster_scans_use:
                raster_scan = dataset.all_raster_scans[raster_scan_index]
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
                        signals.plot_raster_scan_export_lock.emit(p, raster_scan, True, f_dir, lock)
                        lock.acquire()
                    mainwindow.raster_scan_controller.plot_raster_scan(
                        raster_scan=raster_scan, for_export=True,
                        export_path=f_dir)

                if signals:
                    signals.progress.emit()

    ## DataFrame compilation and writing to Feather files
    if any([ex_df_levels, ex_df_grouped_levels, ex_df_grouping_info]):
        life_cols_add = ['tau_1_ns', 'tau_2_ns', 'tau_3_ns', 'amp_1', 'amp_2', 'amp_3',
                         'irf_shift_ns', 'decay_bg', 'irf_bg',
                         'chi_squared']
        if ex_df_levels or ex_df_grouped_levels:
            levels_cols = ['particle', 'level', 'start_s', 'end_s', 'dwell_s', 'int_cps',
                           'num_photons']
            grouped_levels_cols = levels_cols.copy()
            grouped_levels_cols[1] = 'grouped_level'
            grouped_levels_cols.insert(2, 'group_index')
            if ex_df_levels_lifetimes:
                levels_cols.extend(life_cols_add)
            if ex_df_grouped_levels_lifetimes:
                grouped_levels_cols.extend(life_cols_add)

            data_levels = list()
            if ex_df_grouped_levels:
                data_grouped_levels = list()

        if ex_df_grouping_info:
            grouping_info_cols = ['particle', 'group', 'total_dwell_s', 'int_cps', 'num_levels',
                                  'num_photons', 'num_steps', 'is_best_step']
            data_grouping_info = list()

        for p in particles:
            # print(p.name)
            if ex_df_levels:
                for l_num, l in enumerate(p.cpts.levels):
                    row = [p.name, l_num + 1,
                           *get_level_data(l, incl_lifetimes=ex_df_levels_lifetimes)]
                    data_levels.append(row)

            if ex_df_grouped_levels:
                for g_l_num, g_l in enumerate(p.ahca.selected_step.group_levels):
                    row = [p.name, g_l_num + 1, g_l.group_ind + 1,
                           *get_level_data(g_l, incl_lifetimes=ex_df_grouped_levels_lifetimes)]
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
            levels_df_path = os.path.join(f_dir, 'levels.df')
            feather.write_feather(df=df_levels, dest=levels_df_path)
            if signals:
                signals.progress.emit()

        if ex_df_grouped_levels:
            df_grouped_levels = pd.DataFrame(data=data_grouped_levels, columns=grouped_levels_cols)
            grouped_levels_df_path = os.path.join(f_dir, 'grouped levels.df')
            feather.write_feather(df=df_grouped_levels, dest=grouped_levels_df_path)
            if signals:
                signals.progress.emit()

        if ex_df_grouping_info:
            df_grouping_info = pd.DataFrame(data=data_grouping_info, columns=grouping_info_cols)
            grouping_info_df_path = os.path.join(f_dir, 'grouping info.df')
            feather.write_feather(df=df_grouping_info, dest=grouping_info_df_path)
            if signals:
                signals.progress.emit()


    if signals:
        signals.end_progress.emit()
        signals.status_message.emit("Done")


def get_level_data(level: Level, incl_lifetimes: bool = False) -> List:
    data = [*level.times_s, level.dwell_time_s, level.int_p_s, level.num_photons]
    if incl_lifetimes:
        h = level.histogram
        if h.fitted:
            taus = list(h.tau)
            taus.extend([np.NaN] * (3 - h.numexp))
            data.extend(taus)

            amps = list(h.amp)
            amps.extend([np.NaN] * (3 - h.numexp))
            data.extend(amps)

            data.extend([h.shift, h.bg, h.irfbg, h.chisq])
        else:
            data.extend([np.NaN]*10)

    return data
