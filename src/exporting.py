from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QFileDialog
import numpy as np

from my_logger import setup_logger
from PyQt5.QtCore import QRunnable, pyqtSlot
from signals import WorkerSignals

if TYPE_CHECKING:
    from main import MainWindow

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
        if signals:
            signals.start_progress.emit(len(particles))
            signals.status_message.emit(f"Exporting data for {mode} particles...")

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
    ex_plot_raster_scans = mainwindow.chbEx_Plot_Raster_Scans.isChecked()

    def open_file(path: str):
        return open(path, 'w', newline='')

    # Export fits of whole traces
    # all_fitted = [p.histogram.fitted]
    # if p.has_levels and all([lvl.histogram is not None for lvl in p.levels]):
    #     all_fitted.extend([lvl.histogram.fitted for lvl in p.levels])
    #     if p.has_groups and all([grp.histogram is not None for grp in p.groups]):
    #         all_fitted.extend([grp.histogram.fitted for grp in p.groups])
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
                signals.plot_trace_export.emit(p, True, f_dir)
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
                    signals.plot_trace.emit(p, True)
                    signals.plot_levels_export.emit(p, True, f_dir)
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
                    signals.plot_trace.emit(p, True)
                    signals.plot_levels.emit(p, True)
                    signals.plot_group_bounds_export.emit(p, True, f_dir)
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
                        answer = 'Yes'
                    else:
                        answer = 'No'
                    f.write(f"Selected solution highest BIC value? {answer}\n\n")

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
                signals.plot_grouping_bic_export.emit(p, True, f_dir)
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
                                [str(i + 1), str(g.dwell_time_s), str(g.int_p_s), str(g.num_photons)]
                                + tauexp + ampexp + other_exp)

                with open_file(group_path) as f:
                    writer = csv.writer(f, dialect=csv.excel)
                    writer.writerows(rows)

        # TODO: make a function for the repeated code
        if ex_hist:
            tr_path = os.path.join(f_dir, p.name + ' hist.csv')
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
                            rows.append([str(time), str(decay[j]), str(convd[j]), str(residual[j])])

                        with open_file(hist_path) as f:
                            writer = csv.writer(f, dialect=csv.excel)
                            writer.writerows(rows)

        if ex_plot_lifetimes and ex_plot_hist_only:
            if signals:
                signals.plot_decay_export.emit(-1, p, True, f_dir)
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
                        signals.plot_decay_export.emit(i, p, True, dir_path)
                    else:
                        mainwindow.lifetime_controller.plot_decay(select_ind=i,
                                                                  particle=p,
                                                                  for_export=True,
                                                                  export_path=dir_path)
                if p.has_groups:
                    for i in range(p.num_groups):
                        i_g = i + p.num_levels
                        if signals:
                            signals.plot_decay_export.emit(i_g, p, True, dir_path)
                        else:
                            mainwindow.lifetime_controller.plot_decay(select_ind=i_g,
                                                                      particle=p,
                                                                      for_export=True,
                                                                      export_path=dir_path)

        if ex_plot_and_residuals or (ex_plot_lifetimes and ex_plot_with_fit):
            if signals:
                signals.plot_decay.emit(-1, p, True)
                signals.plot_convd_export.emit(-1, p, False, True, f_dir)
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
                for i in range(p.num_levels):
                    if signals:
                        signals.plot_decay.emit(i, p, True)
                        signals.plot_convd_export.emit(i, p, False, True, dir_path)
                    else:
                        mainwindow.lifetime_controller.plot_decay(select_ind=i,
                                                                  particle=p,
                                                                  for_export=True)
                        mainwindow.lifetime_controller.plot_convd(select_ind=i,
                                                                  particle=p,
                                                                  for_export=True,
                                                                  export_path=dir_path)
                if p.has_groups:
                    for i in range(p.num_groups):
                        i_g = i + p.num_levels
                        if signals:
                            signals.plot_decay.emit(i_g, p, True)
                            signals.plot_convd_export.emit(i_g, p, False, True, dir_path)
                        else:
                            mainwindow.lifetime_controller.plot_decay(select_ind=i_g,
                                                                      particle=p,
                                                                      for_export=True)
                            mainwindow.lifetime_controller.plot_convd(select_ind=i_g,
                                                                      particle=p,
                                                                      for_export=True,
                                                                      export_path=dir_path)

        if ex_plot_and_residuals:
            was_showing = mainwindow.chbShow_Residuals.isChecked()
            if not was_showing:
                if signals:
                    signals.show_residual_widget.emit(True)
                else:
                    mainwindow.lifetime_controller.residual_widget.show()
            if signals:
                signals.plot_residuals_export.emit(-1, p, True, f_dir)
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
                        signals.plot_residuals_export.emit(i, p, True, dir_path)
                    else:
                        mainwindow.lifetime_controller.plot_residuals(select_ind=i,
                                                                      particle=p,
                                                                      for_export=True,
                                                                      export_path=dir_path)
                if p.has_groups:
                    for i in range(p.num_groups):
                        i_g = i + p.num_levels
                        if signals:
                            signals.plot_residuals_export.emit(i_g, p, True, dir_path)
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
                signals.plot_spectra_export.emit(p, True, f_dir)
            else:
                mainwindow.spectra_controller.plot_spectra(particle=p, for_export=True,
                                                           export_path=f_dir)

        logger.info('Exporting Finished')
        p.has_exported = True

    if ex_plot_raster_scans:
        # all_raster_scan_particles = list()
        # for part in particles:
        #     all_raster_scan_particles.extend(part.raster_scan.particle_indexes)
        # all_part_with_raster_scans = np.unique(all_raster_scan_particles).tolist()
        dataset = mainwindow.current_dataset
        raster_scans_use = [part.raster_scan.dataset_index for part in particles]
        raster_scans_use = np.unique(raster_scans_use).tolist()
        for raster_scan_index in raster_scans_use:
            raster_scan = dataset.all_raster_scans[raster_scan_index]
            if signals:
                signals.plot_raster_scan_export.emit(p, raster_scan, True, f_dir)
            mainwindow.raster_scan_controller.plot_raster_scan(
                raster_scan=raster_scan, for_export=True,
                export_path=f_dir)

    if signals:
        signals.progress.emit()
        signals.end_progress.emit()
        signals.status_message.emit("Done")
