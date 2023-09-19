from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, List, Union, Tuple
from time import sleep
import re

import numpy as np
import pandas as pd
from pyarrow import feather

import smsh5
from my_logger import setup_logger
from PyQt5.QtCore import QRunnable, pyqtSlot
from signals import WorkerSignals
from threading import Lock

import matplotlib

matplotlib.use("Agg")

if TYPE_CHECKING:
    from main import MainWindow
    from change_point import Level
    from grouping import Group, GlobalLevel

logger = setup_logger(__name__)

DATAFRAME_FORMATS = {
    "Parquet (*.parquet)": 0,
    "Feather (*.ftr)": 1,
    "Feather (*.df)": 2,
    "Pickle (*.pkl)": 3,
    "HDF (*.h5)": 4,
    "Excel (*.xlsx)": 5,
    "CSV (*.csv)": 6,
}


class ExportWorker(QRunnable):
    def __init__(
        self,
        main_window: MainWindow,
        mode: str = None,
        lock: Lock = None,
        f_dir: str = None,
    ):
        super(ExportWorker, self).__init__()
        self.main_window = main_window
        self.mode = mode
        self.lock = lock
        self.f_dir = f_dir
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            exporter = Exporter(
                main_window=self.main_window,
                mode=self.mode,
                signals=self.signals,
                lock=self.lock,
                f_dir=self.f_dir,
            )
            exporter.run_export()
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.fitting_finished.emit(self.mode)


class ExporterOptions:
    def __init__(self, main_window: MainWindow):
        self._main_window = main_window

    ############################################################################
    #       ROI
    ############################################################################

    @property
    def use_roi(self):
        return self._main_window.chbEx_Use_ROI.isChecked()

    @property
    def ex_traces(self):
        return self._main_window.chbEx_Trace.isChecked()

    @property
    def ex_levels(self):
        return self._main_window.chbEx_Levels.isChecked()

    @property
    def ex_grouped_levels(self):
        return self._main_window.chbEx_Grouped_Levels.isChecked()

    @property
    def ex_global_grouped_levels(self):
        return self._main_window.chbEx_Global_Grouped_Levels.isChecked()

    @property
    def ex_grouping_info(self):
        return self._main_window.chbEx_Grouping_Info.isChecked()

    @property
    def ex_grouping_results(self):
        return self._main_window.chbEx_Grouping_Results.isChecked()

    @property
    def ex_lifetime(self):
        return self._main_window.chbEx_Lifetimes.isChecked()

    @property
    def ex_hist(self):
        return self._main_window.chbEx_Hist.isChecked()

    @property
    def ex_spectra_2d(self):
        return self._main_window.chbEx_Spectra_2D.isChecked()

    @property
    def ex_raster_scan_2d(self):
        return self._main_window.chbEx_Raster_Scan_2D.isChecked()

    @property
    def ex_corr_hists(self):
        return self._main_window.chbEx_Corr.isChecked()

    ############################################################################
    #       DataFrames
    ############################################################################

    @property
    def ex_df_levels(self):
        return self._main_window.chbEx_DF_Levels.isChecked()

    @property
    def ex_df_levels_lifetimes(self):
        return self._main_window.chbEx_DF_Levels_Lifetimes.isChecked()

    @property
    def ex_df_grouped_levels(self):
        return self._main_window.chbEx_DF_Grouped_Levels.isChecked()

    @property
    def ex_df_grouped_levels_lifetimes(self):
        return self._main_window.chbEx_DF_Grouped_Levels_Lifetimes.isChecked()

    @property
    def ex_df_grouping_info(self):
        return self._main_window.chbEx_DF_Grouping_Info.isChecked()

    @property
    def ex_df_format(self):
        return self._main_window.cmbEx_DataFrame_Format.currentIndex()

    ############################################################################
    #       Plots
    ############################################################################

    @property
    def ex_plot_intensities(self):
        return self._main_window.chbEx_Plot_Intensity.isChecked()

    @property
    def ex_plot_intensities_only(self):
        return self.ex_plot_intensities and self._main_window.rdbInt_Only.isChecked()

    @property
    def ex_plot_intensities_with_levels(self):
        return self.ex_plot_intensities and self._main_window.rdbWith_Levels.isChecked()

    @property
    def ex_plot_intensities_with_levels_and_groups(self):
        return self.ex_plot_intensities and self._main_window.rdbAnd_Groups.isChecked()

    @property
    def ex_plot_lifetimes(self):
        return self._main_window.chbEx_Plot_Lifetimes.isChecked()

    @property
    def ex_plot_lifetimes_hist_only(self):
        return self.ex_plot_lifetimes and self._main_window.mwdbHist_Only.isChecked()

    @property
    def ex_plot_lifetimes_with_fit(self):
        return self.ex_plot_lifetimes and self._main_window.rdbWith_Fit.isChecked()

    @property
    def ex_plot_lifetimes_fit_and_residuals(self):
        return self.ex_plot_lifetimes and self._main_window.rdbAnd_Residuals.isChecked()

    @property
    def ex_plot_lifetimes_only_groups(self):
        return (
            self.ex_plot_lifetimes
            and self._main_window.chbEx_Plot_Lifetimes_Only_Groups.isChecked()
        )

    @property
    def ex_plot_grouping_bics(self):
        return self._main_window.chbEx_Plot_Group_BIC.isChecked()

    @property
    def ex_plot_raster_scans(self):
        return self._main_window.chbEx_Plot_Raster_Scans.isChecked()

    @property
    def ex_plot_spectra(self):
        return self._main_window.chbEx_Plot_Spectra.isChecked()

    @property
    def ex_plot_corr_hists(self):
        return self._main_window.chbEx_Plot_Corr.isChecked()

    @property
    def any_particle_text_plot(self):
        any_particle_text_plot = any(
            [
                self.ex_traces,
                self.ex_levels,
                self.ex_plot_intensities,
                self.ex_grouped_levels,
                self.ex_global_grouped_levels,
                self.ex_grouping_info,
                self.ex_grouping_results,
                self.ex_plot_grouping_bics,
                self.ex_lifetime,
                self.ex_hist,
                self.ex_plot_lifetimes,
                self.ex_spectra_2d,
                self.ex_plot_spectra,
                self.ex_corr_hists,
                self.ex_plot_corr_hists,
            ]
        )
        return any_particle_text_plot


class Exporter:
    def __init__(
        self,
        main_window: MainWindow,
        mode: str = None,
        signals: WorkerSignals = None,
        lock: Lock = None,
        f_dir: str = None,
    ):
        self._main_window = main_window
        self.options = ExporterOptions(main_window=main_window)
        self.mode = mode
        self.signals = signals
        self.lock = lock
        self.f_dir = f_dir
        if f_dir is not None:
            self.f_dir = os.path.abspath(self.f_dir)

    @property
    def mw(self) -> MainWindow:
        return self._main_window

    @property
    def main_window(self) -> MainWindow:
        return self._main_window

    @staticmethod
    def _open_file(path: str):
        return open(path, "w", newline="")

    def run_export(self):
        assert self.mode in [
            "current",
            "selected",
            "all",
        ], "MainWindow\tThe mode parameter is invalid"

        if self.mode == "current":
            particles = [self.mw.current_particle]
        elif self.mode == "selected":
            particles = self.mw.get_checked_particles()
        else:
            particles = self.mw.current_dataset.particles

        if self.f_dir is None:
            return
        else:
            try:
                raster_scans_use = [
                    part.raster_scan.h5dataset_index for part in particles
                ]
                raster_scans_use = np.unique(raster_scans_use).tolist()
            except AttributeError:
                raster_scans_use = []

        if self.lock is None:
            self.lock = Lock()

        if self.signals is not None:
            prog_num = 0
            if self.options.any_particle_text_plot:
                prog_num = prog_num + len(particles)
            if self.options.ex_raster_scan_2d or self.options.ex_plot_raster_scans:
                prog_num = prog_num + len(raster_scans_use)
            if self.options.ex_df_levels:
                prog_num = prog_num + 1
            if self.options.ex_df_grouped_levels:
                prog_num = prog_num + 1
            if self.options.ex_df_grouping_info:
                prog_num = prog_num + 1

            self.signals.start_progress.emit(prog_num)
            self.signals.status_message.emit(
                f"Exporting data for {self.mode} particles..."
            )

        # Export fits of whole traces
        all_fitted = [part._histogram.fitted for part in particles]
        if self.options.ex_lifetime and any(all_fitted):
            self.export_lifetimes(particles=particles, whole_trace=True)

        if self.options.ex_lifetime and any(all_fitted):
            self.export_lifetimes(particles=particles, whole_trace=True)

        # Export data for levels
        if self.options.any_particle_text_plot:
            for num, p in enumerate(particles):
                if self.options.ex_traces:
                    self.export_trace(particle=p)

                if self.options.ex_levels:
                    self.export_levels(particle=p)

                if self.options.ex_corr_hists:
                    self.export_corr_hists(particle=p)

                if self.options.ex_plot_lifetimes and p.has_levels:
                    self.plot_lifetimes(particle=p)

                if self.options.ex_grouped_levels and p.has_groups:
                    self.export_levels_grouped_plot(particle=p)

                if self.options.ex_global_grouped_levels and p.has_global_grouping:
                    self.export_levels_global_grouped_plot(particle=p)

                if self.options.ex_grouping_info and p.has_groups:
                    self.export_grouping_info(particle=p)

                if self.options.ex_grouping_results and p.has_groups:
                    self.export_grouping_results(particle=p)

                if self.options.ex_plot_grouping_bics:
                    self.plot_grouping_bic(particle=p)

                if self.options.ex_lifetime:
                    self.export_lifetimes(particles=p)

                if self.options.ex_hist:
                    self.export_hists(particle=p)

                if self.options.ex_plot_intensities_only:
                    self.plot_intensities(particle=p)
                elif self.options.ex_plot_intensities_with_levels:
                    if p.has_levels:
                        self.plot_levels(particle=p)
                elif self.options.ex_plot_intensities_with_levels_and_groups:
                    if p.has_groups:
                        self.plot_levels(particle=p, plot_groups=True)

                if self.options.ex_plot_lifetimes:
                    # handles with_fit and only_groups options internally
                    self.plot_lifetimes(particle=p)
                elif self.options.ex_plot_lifetimes_fit_and_residuals:
                    self.plot_lifetime_fit_residuals(particle=p)

                if self.options.ex_spectra_2d:
                    self.export_spectra_2d(particle=p)

                if self.options.ex_plot_spectra:
                    self.plot_spectra(particle=p)

                if self.options.ex_plot_corr_hists:
                    self.plot_corr_hists(particle=p)

                if self.signals:
                    self.signals.progress.emit()
                p.has_exported = True

        if self.options.ex_raster_scan_2d or self.options.ex_plot_raster_scans:
            dataset = self.mw.current_dataset
            for raster_scan_index in raster_scans_use:
                raster_scan = dataset.all_raster_scans[raster_scan_index]
                rs_part_ind = raster_scan.particle_indexes[0]
                p = dataset.particles[rs_part_ind]
                if self.options.ex_raster_scan_2d:
                    self.export_raster_scan_2d(raster_scan=raster_scan)

                if self.options.ex_plot_raster_scans:
                    self.plot_raster_scan(p=p, raster_scan=raster_scan)

                if self.signals:
                    self.signals.progress.emit()

        # DataFrame compilation and writing
        if any(
            [
                self.options.ex_df_levels,
                self.options.ex_df_grouped_levels,
                self.options.ex_df_grouping_info,
            ]
        ):
            self.export_dataframes(particles=particles)

        if self.signals:
            self.signals.end_progress.emit()
            self.signals.status_message.emit("Done")

        logger.info("Export finished")

    ##############################################################
    # pandas DataFrame exports
    ##############################################################

    @staticmethod
    def write_dataframe_to_file(
        dataframe: pd.DataFrame, path: str, filename: str, file_type: dict
    ):
        if file_type == 0:  # Parquet
            file_path = os.path.join(path, filename + ".parquet")
            dataframe.to_parquet(path=file_path)
        elif file_type == 1 or file_type == 2:  # Feather
            if file_type == 1:  # with .ftr
                file_path = os.path.join(path, filename + ".ftr")
            else:
                file_path = os.path.join(path, filename + ".df")
            feather.write_feather(df=dataframe, dest=file_path)
        elif file_type == 3:  # Pickle
            file_path = os.path.join(path, filename + ".pkl")
            dataframe.to_pickle(path=file_path)
        elif file_type == 4:  # HDF
            file_path = os.path.join(path, filename + ".h5")
            dataframe.to_hdf(path_or_buf=file_path, key=filename, format="table")
        elif file_type == 5:  # Excel
            file_path = os.path.join(path, filename + ".xlsx")
            dataframe.to_excel(file_path)
        elif file_type == 6:  # CSV
            file_path = os.path.join(path, filename + ".csv")
            dataframe.to_csv(file_path)
        else:
            logger.error("File type not configured yet")
        pass

    def export_dataframes(self, particles):
        any_has_lifetime = any([p.has_fit_a_lifetime for p in particles])
        if not any_has_lifetime:
            if self.signals:
                self.signals.progress.emit()
                self.signals.progress.emit()
                self.signals.progress.emit()
            return

        max_exp_num = np.max(
            [
                *[p.histogram.numexp for p in particles if p.histogram.fitted],
                *[
                    l.histogram.numexp
                    for p in particles
                    for l in p.levels
                    if l.histogram.fitted
                ],
                *[
                    g.histogram.numexp
                    for p in particles
                    for g in p.groups
                    if g.histogram.fitted
                ],
            ]
        )
        if self.options.ex_df_levels:
            all_levels = [l for p in particles for l in p.levels]
            df_levels = self.levels_to_df(max_exp_num=max_exp_num, levels=all_levels)
            self.write_dataframe_to_file(
                dataframe=df_levels,
                path=self.f_dir,
                filename="levels",
                file_type=self.options.ex_df_format,
            )
            if self.signals:
                self.signals.progress.emit()

        if self.options.ex_df_grouped_levels:
            all_group_levels = [g_l for p in particles for g_l in p.group_levels]
            df_group_levels = self.levels_to_df(
                max_exp_num=max_exp_num, levels=all_group_levels
            )
            self.write_dataframe_to_file(
                dataframe=df_group_levels,
                path=self.f_dir,
                filename="group_levels",
                file_type=self.options.ex_df_format,
            )
            if self.signals:
                self.signals.progress.emit()

        if self.options.ex_df_grouping_info:
            all_groups = [group for p in particles for group in p.groups]
            df_groups = self.groups_to_df(groups=all_groups)
            self.write_dataframe_to_file(
                dataframe=df_groups,
                path=self.f_dir,
                filename="groups",
                file_type=self.options.ex_df_format,
            )
            if self.signals:
                self.signals.progress.emit()

    @staticmethod
    def groups_to_df(groups: List[Group]):
        s = dict()
        s["particle"] = pd.Series([g.lvls[0]._particle.unique_name for g in groups])
        s["group"] = pd.Series([g.group_ind + 1 for g in groups])
        s["total_dwell_time"] = pd.Series([g.dwell_time_s for g in groups])
        s["int"] = pd.Series([g.int_p_s for g in groups])
        s["num_levels"] = pd.Series([len(g.lvls) for g in groups])
        s["num_photons"] = pd.Series([g.num_photons for g in groups])
        s["num_steps"] = pd.Series([g.lvls[0]._particle.ahca.num_steps for g in groups])
        s["is_best_step"] = pd.Series(
            [
                g.lvls[0]._particle.ahca.selected_step_ind
                == g.lvls[0]._particle.ahca.best_step_ind
                for g in groups
            ]
        )
        s["is_primary_particle"] = pd.Series(
            [not g.lvls[0]._particle.is_secondary_part for g in groups]
        )
        s["tcspc_card"] = pd.Series([g.lvls[0]._particle.tcspc_card for g in groups])

        return pd.DataFrame(s)

    @staticmethod
    def levels_to_df(max_exp_num: int, levels: List[Union[Level, GlobalLevel]]):
        s = dict()
        s["particle"] = pd.Series([l._particle.unique_name for l in levels])
        s["level"] = pd.Series([l.particle_ind + 1 for l in levels])
        s["group_index"] = pd.Series(
            [
                l.group_ind + 1
                if l._particle.has_groups and l.group_ind is not None
                else None
                for l in levels
            ]
        )
        s["start"] = pd.Series([l.times_s[0] for l in levels])
        s["end"] = pd.Series([l.times_s[-1] for l in levels])
        s["dwell"] = pd.Series([l.dwell_time_s for l in levels])
        s["dwell_frac"] = pd.Series(
            [l.dwell_time_s / l._particle.dwell_time_s for l in levels]
        )
        s["int"] = pd.Series([l.int_p_s for l in levels])
        s["num_photons"] = pd.Series([l.num_photons for l in levels])
        s["num_photons_in_lifetime_fit"] = pd.Series(
            [
                l.histogram.num_photons_used if l.histogram.fitted else None
                for l in levels
            ]
        )
        for exp_num in range(1, max_exp_num + 1):
            if exp_num == 1:
                av_taus, av_tau_stds = list(
                    zip(
                        *[
                            (l.histogram.avtau, l.histogram.avtaustd)
                            if l.histogram.fitted
                            else (None, None)
                            for l in levels
                        ]
                    )
                )
                s["av_tau"] = pd.Series(av_taus)
                s["av_tau_std"] = pd.Series(av_tau_stds)

            taus, tau_stds, amps, amp_stds = list(
                zip(
                    *[
                        (
                            l.histogram.tau[exp_num - 1],
                            l.histogram.stds[exp_num - 1],
                            l.histogram.amp[exp_num - 1],
                            l.histogram.stds[exp_num - 1 + l.histogram.numexp],
                        )
                        if l.histogram.fitted and l.histogram.numexp <= max_exp_num
                        else (None, None, None, None)
                        for l in levels
                    ]
                )
            )
            s[f"tau_{exp_num}"] = pd.Series(taus)
            s[f"tau_std_{exp_num}"] = pd.Series(tau_stds)
            s[f"amp_{exp_num}"] = pd.Series(amps)
            s[f"amp_std_{exp_num}"] = pd.Series(amp_stds)

        s["irf_shift"] = pd.Series(
            [l.histogram.shift if l.histogram.fitted else None for l in levels]
        )
        s["decay_bg"] = pd.Series(
            [
                l.histogram.stds[2 * l.histogram.numexp] if l.histogram.fitted else None
                for l in levels
            ]
        )
        s["irf_bg"] = pd.Series(
            [l.histogram.irfbg if l.histogram.fitted else None for l in levels]
        )
        s["chisq"] = pd.Series(
            [l.histogram.chisq if l.histogram.fitted else None for l in levels]
        )
        s["dw"] = pd.Series(
            [l.histogram.dw if l.histogram.fitted else None for l in levels]
        )
        s["dw_5"] = pd.Series(
            [
                l.histogram.dw_bound[0]
                if l.histogram.fitted and l.histogram.dw_bound is not None
                else None
                for l in levels
            ]
        )
        s["dw_1"] = pd.Series(
            [
                l.histogram.dw_bound[1]
                if l.histogram.fitted and l.histogram.dw_bound is not None
                else None
                for l in levels
            ]
        )
        s["dw_03"] = pd.Series(
            [
                l.histogram.dw_bound[2]
                if l.histogram.fitted and l.histogram.dw_bound is not None
                else None
                for l in levels
            ]
        )
        s["dw_01"] = pd.Series(
            [
                l.histogram.dw_bound[3]
                if l.histogram.fitted and l.histogram.dw_bound is not None
                else None
                for l in levels
            ]
        )
        s["chisq"] = pd.Series(
            [l.histogram.chisq if l.histogram.fitted else None for l in levels]
        )
        s["is_in_roi"] = pd.Series(
            [
                l._particle.first_level_ind_in_roi
                <= l.particle_ind
                <= l._particle.last_level_ind_in_roi
                for l in levels
            ]
        )

        return pd.DataFrame(s)

    ##############################################################
    # NON pandas DataFrame exports
    ##############################################################

    def export_raster_scan_2d(self, raster_scan):
        raster_scan_2d_path = os.path.join(
            self.f_dir, f"Raster Scan {raster_scan.h5dataset_index + 1} data.csv"
        )
        top_row = [np.NaN, *raster_scan.x_axis_pos]
        y_and_data = np.column_stack((raster_scan.y_axis_pos, raster_scan.dataset[:]))
        x_y_data = np.insert(y_and_data, 0, top_row, axis=0)
        with self._open_file(raster_scan_2d_path) as f:
            f.write("Rows = X-Axis (um)")
            f.write("Columns = Y-Axis (um)")
            f.write("")
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(x_y_data)

    def export_spectra_2d(self, particle: smsh5.Particle):
        p_name = particle.unique_name
        spectra_2d_path = os.path.join(self.f_dir, p_name + " spectra-2D.csv")
        with self._open_file(spectra_2d_path) as f:
            f.write("First row:,Wavelength (nm)\n")
            f.write("First column:,Time (s)\n")
            f.write("Values:,Intensity (counts/s)\n\n")

            rows = list()
            rows.append([""] + particle.spectra.wavelengths.tolist())
            for num, spec_row in enumerate(particle.spectra.data[:]):
                this_row = list()
                this_row.append(str(particle.spectra.series_times[num]))
                for single_val in spec_row:
                    this_row.append(str(single_val))
                rows.append(this_row)

            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    def export_hists(self, particle: smsh5.Particle):
        def _export_hist(f_path: str, histogram: smsh5.Histogram):
            rows = list()
            if histogram.fitted:
                times = histogram.convd_t
                if times is not None:
                    decay = histogram.fit_decay
                    convd = histogram.convd
                    residual = histogram.residuals
                    rows.append(["Time (ns)", "Decay", "Fitted", "Residual"])
                    for i, time in enumerate(times):
                        rows.append(
                            [str(time), str(decay[i]), str(convd[i]), str(residual[i])]
                        )
            else:
                decay = histogram.decay
                times = histogram.t
                rows.append(["Times (ns)", "Decay"])
                for i, time in enumerate(times):
                    rows.append([str(time), str(decay[i])])
            with self._open_file(f_path) as f:
                writer = csv.writer(f, dialect=csv.excel)
                writer.writerows(rows)

        p_name = particle.unique_name
        use_roi = particle.use_roi_for_histogram
        post_fix_roi = " hist (roi).csv" if use_roi else " hist.csv"
        tr_path = os.path.join(self.f_dir, p_name + post_fix_roi)
        _export_hist(f_path=tr_path, histogram=particle.histogram)

        dir_path = os.path.join(self.f_dir, p_name + " hists\\")
        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass

        hists_with_name: List[Tuple[str, smsh5.Histogram]] = []
        if particle.has_levels:
            levels = particle.levels if not use_roi else particle.levels_roi
            hists_with_name.extend(
                [(f"level {i+1}", l.histogram) for i, l in enumerate(levels)]
            )
        if particle.has_groups:
            g_levels = (
                particle.group_levels if not use_roi else particle.group_levels_roi
            )
            hists_with_name.extend(
                [
                    (f"group_level {i+1}", g_l.histogram)
                    for i, g_l in enumerate(g_levels)
                ]
            )
            hists_with_name.extend(
                [(f"group {i+1}", g.histogram) for i, g in enumerate(particle.groups)]
            )

        for name, hist in hists_with_name:
            f_path = (
                dir_path
                + name
                + (" (roi)" if use_roi else "")
                + (" (fitted)" if hist.fitted else "")
                + ".csv"
            )
            _export_hist(f_path=f_path, histogram=hist)

    def export_lifetimes(
        self,
        particles: Union[smsh5.Particle, List[smsh5.Particle]],
        whole_trace: bool = False,
    ) -> None:
        def _export_lifetimes(
            lifetime_path: str,
            particles: List[smsh5.Particle],
            levels=False,
        ):
            if type(particles) is smsh5.Particle:
                particles = [particles]

            # TODO: better handling of the cases that produce the error instead of the try.
            try:
                max_exp_number = np.max(
                    [
                        *[p.histogram.numexp for p in particles if p.histogram.fitted],
                        *[
                            l.histogram.numexp
                            for p in particles
                            for l in p.levels
                            if l.histogram.fitted
                        ],
                        *[
                            g.histogram.numexp
                            for p in particles
                            for g in p.groups
                            if g.histogram.fitted
                        ],
                    ]
                )
            except TypeError:  # no levels and/or no groups:
                max_exp_number = np.max([p.histogram.numexp for p in particles if p.histogram.fitted])

            tau_cols = [f"Lifetime {i} (ns)" for i in range(max_exp_number)]
            tau_std_cols = [f"Lifetime {i} std (ns)" for i in range(max_exp_number)]
            amp_cols = [f"Amp {i}" for i in range(max_exp_number)]
            amp_std_cols = [f"Amp {i} std" for i in range(max_exp_number)]

            rows = list()
            if levels:
                part_level = ["Level #"]
            else:
                part_level = ["Particle #", "Primary?"]
            rows.append(
                part_level
                + tau_cols
                + tau_std_cols
                + amp_cols
                + amp_std_cols
                + [
                    "Av. Lifetime (ns)",
                    "Av. Lifetime std (ns)",
                    "IRF Shift (ns)",
                    "IRF Shift std (ns)",
                    "Decay BG",
                    "IRF BG",
                    "Chi Squared",
                    "Sim. IRF FWHM (ns)",
                    "Sim. IRF FWHM std (ns)",
                    "DW",
                    "DW 0.05",
                    "DW 0.01",
                    "DW 0.003",
                    "DW 0.001",
                    "ROI applied",
                ]
            )

            def pad(list_to_pad: list, total_len: int = 3) -> list:
                l_padded = list_to_pad.copy()
                if len(list_to_pad) < total_len:
                    l_padded.extend([None] * (total_len - len(list_to_pad)))
                return l_padded

            for i, p in enumerate(particles):
                p_name = p.unique_name
                tau_exp_std = None
                amp_std_exp = None
                histograms: List[smsh5.Histogram] = list()
                if levels and not self.options.use_roi:
                    histograms = [l.histogram for l in p.levels]
                elif levels and self.options.use_roi:
                    histograms = [l.histogram for l in p.levels_roi]
                else:
                    histograms = [p.histogram]
                for h in histograms:
                    if h.fitted:
                        if (
                            h.tau is None or h.amp is None
                        ):  # Problem with fitting the level
                            tau_exp = [""] * max_exp_number
                            amp_exp = [""] * max_exp_number
                            other_exp = ["0", "0", "0", "0"]
                        else:
                            tau_exp = [
                                str(tau) if tau is not None else ""
                                for tau in pad(h.tau, total_len=max_exp_number)
                            ]
                            tau_exp_std = [
                                str(tau_std) if tau_std is not None else ""
                                for tau_std in pad(
                                    h.stds[: h.numexp], total_len=max_exp_number
                                )
                            ]
                            amp_exp = [
                                str(amp) if amp is not None else ""
                                for amp in pad(h.amp, total_len=max_exp_number)
                            ]
                            amp_exp_std = [
                                str(amp_std) if amp_std is not None else ""
                                for amp_std in pad(
                                    h.stds[h.numexp : h.numexp * 2],
                                    total_len=max_exp_number,
                                )
                            ]
                            if hasattr(h, "fwhm") and h.fwhm is not None:
                                sim_irf_fwhm = str(h.fwhm)
                                sim_irf_fwhm_std = str(h.stds[2 * h.numexp + 1])
                            else:
                                sim_irf_fwhm = ""
                                sim_irf_fwhm_std = ""
                            other_exp = [
                                str(
                                    h.avtau[0]
                                    if type(h.avtau) in [list, np.ndarray]
                                    else h.avtau
                                ),
                                str(h.avtaustd),
                                str(h.shift),
                                str(h.stds[2 * h.numexp]),
                                str(h.bg),
                                str(h.irfbg),
                                str(h.chisq),
                                sim_irf_fwhm,
                                sim_irf_fwhm_std,
                                str(h.dw),
                                str(h.dw_bound[0]),
                                str(h.dw_bound[1]),
                                str(h.dw_bound[2]),
                                str(h.dw_bound[3]),
                                str(h.is_for_roi),
                            ]
                        if levels:
                            p_num = [str(i + 1)]
                        else:  # get number from particle name
                            p_num = re.findall(r"\d+", p_name) + [
                                str(int(not p.is_secondary_part))
                            ]
                        rows.append(
                            p_num
                            + tau_exp
                            + tau_exp_std
                            + amp_exp
                            + amp_exp_std
                            + other_exp
                        )
            with self._open_file(lifetime_path) as f:
                writer = csv.writer(f, dialect=csv.excel)
                writer.writerows(rows)

        if type(particles) is smsh5.Particle:
            particles = [particles]

        if whole_trace:
            lifetime_path = os.path.join(self.f_dir, "Whole trace lifetimes.csv")
            all_fitted = [p for p in particles if p.histogram.fitted]
            if len(all_fitted) > 0:
                _export_lifetimes(lifetime_path=lifetime_path, particles=all_fitted)

        else:
            # TODO: fix so try is not needed, see line 783.
            try:
                max_exp_number = np.max(
                    [
                        *[p.histogram.numexp for p in particles if p.histogram.fitted],
                        *[
                            l.histogram.numexp
                            for p in particles
                            for l in p.levels
                            if l.histogram.fitted
                        ],
                        *[
                            g.histogram.numexp
                            for p in particles
                            for g in p.groups
                            if g.histogram.fitted
                        ],
                    ]
                )
            except TypeError:  # no levels and/or no groups:
                max_exp_number = np.max([p.histogram.numexp for p in particles if p.histogram.fitted])

            df_levels = None
            # TODO: fix so try is not needed, see line 783.
            try:
                all_levels = [l for p in particles for l in p.levels if l.histogram.fitted]
            except TypeError:
                all_levels = []
            if len(all_levels):
                df_levels = self.levels_to_df(
                    max_exp_num=max_exp_number,
                    levels=all_levels,
                )
                df_levels.drop(
                    columns=[
                        "group_index",
                        "start",
                        "end",
                        "dwell",
                        "dwell_frac",
                        "int",
                    ],
                    inplace=True,
                )

            df_group_levels = None
            # TODO: fix so try is not needed, see line 783.
            try:
                all_group_levels = [
                    l for p in particles for l in p.group_levels if l.histogram.fitted
                ]
            except TypeError:
                all_group_levels = []
            if len(all_group_levels):
                df_group_levels = self.levels_to_df(
                    max_exp_num=max_exp_number,
                    levels=all_group_levels,
                )
                df_group_levels.drop(
                    columns=[
                        "group_index",
                        "start",
                        "end",
                        "dwell",
                        "dwell_frac",
                        "int",
                    ],
                    inplace=True,
                )

            for p in particles:
                p_name = p.unique_name
                if df_levels is not None:
                    lvls_file = os.path.join(
                        self.f_dir, p_name + " levels-lifetimes.csv"
                    )
                    df = df_levels.loc[df_levels["particle"] == p_name]
                    df.to_csv(lvls_file)

                if df_group_levels is not None:
                    g_lvls_file = os.path.join(
                        self.f_dir, p_name + " group-level-lifetimes.csv"
                    )
                    df = df_group_levels.loc[df_group_levels["particle"] == p_name]
                    df.to_csv(g_lvls_file)

    def export_grouping_results(self, particle: smsh5.Particle):
        pname = particle.unique_name
        grouping_results_path = os.path.join(self.f_dir, pname + " grouping-results")
        if not particle.grouped_with_roi:
            grouping_results_path += ".csv"
        else:
            grouping_results_path += " (ROI).csv"
        with self._open_file(grouping_results_path) as f:
            f.write(f"# of Steps:,{particle.ahca.num_steps}\n")
            f.write(f"Step with highest BIC value:,{particle.ahca.best_step.bic}\n")
            f.write(f"Step selected:,{particle.ahca.selected_step_ind}\n\n")

            rows = list()
            rows.append(["Step #", "# of Groups", "BIC value"])
            for num, step in enumerate(particle.ahca.steps):
                rows.append([str(num + 1), str(step.num_groups), str(step.bic)])

            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    def export_grouping_info(self, particle: smsh5.Particle):
        pname = particle.unique_name
        group_info_path = os.path.join(self.f_dir, pname + " groups-info")
        if not particle.grouped_with_roi:
            group_info_path += ".csv"
        else:
            group_info_path += " (ROI).csv"
        with self._open_file(group_info_path) as f:
            f.write(f"# of Groups:,{particle.ahca.best_step.num_groups}\n")
            if particle.ahca.best_step_ind == particle.ahca.selected_step_ind:
                answer = "TRUE"
            else:
                answer = "FALSE"
            f.write(f"Selected solution highest BIC value?,{answer}\n\n")

            rows = list()
            rows.append(
                [
                    "Group #",
                    "Int (counts/s)",
                    "Total Dwell Time (s)",
                    "# of Levels",
                    "# of Photons",
                ]
            )
            for num, group in enumerate(particle.ahca.selected_step.groups):
                rows.append(
                    [
                        str(num + 1),
                        str(group.int_p_s),
                        str(group.dwell_time_s),
                        str(len(group.lvls)),
                        str(group.num_photons),
                    ]
                )
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    def export_levels_global_grouped_plot(self, particle: smsh5.Particle):
        pname = particle.unique_name
        grp_lvl_path = os.path.join(self.f_dir, pname + " levels-global-grouped.csv")
        rows = list()
        rows.append(
            [
                "Global Grouped Level #",
                "Start Time (s)",
                "End Time (s)",
                "Dwell Time (/s)",
                "Int (counts/s)",
                "Num of Photons",
                "Global Group Index",
            ]
        )
        for i, l in enumerate(particle.global_group_levels):
            rows.append(
                [
                    str(i + 1),
                    str(l.times_s[0]),
                    str(l.times_s[1]),
                    str(l.dwell_time_s),
                    str(l.int_p_s),
                    str(l.num_photons),
                    str(l.group_ind + 1),
                ]
            )
        with self._open_file(grp_lvl_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    def export_levels_grouped_plot(self, particle: smsh5.Particle):
        pname = particle.unique_name
        grp_lvl_tr_path = os.path.join(self.f_dir, pname + " levels-grouped-plot")
        if not particle.grouped_with_roi:
            grp_lvl_tr_path += ".csv"
        else:
            grp_lvl_tr_path += " (ROI).csv"
        ints, times = particle.levels2data(use_grouped=True)
        rows = list()
        rows.append(["Grouped Level #", "Time (s)", "Int (counts/s)"])
        for i in range(len(ints)):
            rows.append([str((i // 2) + 1), str(times[i]), str(ints[i])])
        with self._open_file(grp_lvl_tr_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)
        grp_lvl_path = os.path.join(self.f_dir, pname + " levels-grouped")
        if not particle.grouped_with_roi:
            grp_lvl_path += ".csv"
        else:
            grp_lvl_path += " (ROI).csv"
        rows = list()
        rows.append(
            [
                "Grouped Level #",
                "Start Time (s)",
                "End Time (s)",
                "Dwell Time (/s)",
                "Int (counts/s)",
                "Num of Photons",
                "Group Index",
            ]
        )
        for i, l in enumerate(particle.group_levels):
            rows.append(
                [
                    str(i + 1),
                    str(l.times_s[0]),
                    str(l.times_s[1]),
                    str(l.dwell_time_s),
                    str(l.int_p_s),
                    str(l.num_photons),
                    str(l.group_ind + 1),
                ]
            )
        with self._open_file(grp_lvl_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    def _export_level_plot(
        self,
        ints: Union[list, np.ndarray],
        lvl_tr_path: str,
        times: Union[list, np.ndarray],
    ):
        rows = list()
        rows.append(["Level #", "Time (s)", "Int (counts/s)"])
        for i in range(len(ints)):
            rows.append([str((i // 2) + 1), str(times[i]), str(ints[i])])
        with self._open_file(lvl_tr_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    def _export_levels(
        self, lvl_path: str, particle: smsh5.Particle, roi: bool = False
    ):
        rows = list()
        rows.append(
            [
                "Level #",
                "Start Time (s)",
                "End Time (s)",
                "Dwell Time (/s)",
                "Int (counts/s)",
                "Num of Photons",
            ]
        )
        if roi:
            levels = particle.levels_roi
        else:
            levels = particle.cpts.levels
        for i, l in enumerate(levels):
            rows.append(
                [
                    str(i + 1),
                    str(l.times_s[0]),
                    str(l.times_s[1]),
                    str(l.dwell_time_s),
                    str(l.int_p_s),
                    str(l.num_photons),
                ]
            )
        with self._open_file(lvl_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    def export_levels(self, particle: smsh5.Particle):
        p_name = particle.unique_name
        if not self.options.use_roi:
            lvl_tr_path = os.path.join(self.f_dir, p_name + " levels-plot.csv")
            ints, times = particle.levels2data(use_grouped=False, use_roi=False)
            self._export_level_plot(ints=ints, lvl_tr_path=lvl_tr_path, times=times)

            lvl_path = os.path.join(self.f_dir, p_name + " levels.csv")
            self._export_levels(lvl_path=lvl_path, particle=particle)
        else:
            lvl_tr_path = os.path.join(self.f_dir, p_name + " levels-plot (ROI).csv")
            ints, times = particle.levels2data(
                use_grouped=False, use_roi=self.options.use_roi
            )
            self._export_level_plot(ints=ints, lvl_tr_path=lvl_tr_path, times=times)
            lvl_path = os.path.join(self.f_dir, p_name + " levels (ROI).csv")
            self._export_levels(lvl_path=lvl_path, particle=particle, roi=True)

    def export_corr_hists(self, particle: smsh5.Particle):
        pname = particle.unique_name
        tr_path = os.path.join(self.f_dir, pname + " corr.csv")
        self.export_corr(tr_path=tr_path, particle=particle)
        if self.options.use_roi:
            return

    def _export_trace(
        self,
        ints: Union[list, np.ndarray],
        particle: smsh5.Particle,
        times: Union[list, np.ndarray],
        tr_path: str,
    ):
        rows = list()
        rows.append(
            ["Bin #", "Bin Time (s)", f"Bin Int (counts/{particle.bin_size}ms)"]
        )
        for i in range(len(ints)):
            rows.append([str(i + 1), str(times[i]), str(ints[i])])
        with self._open_file(tr_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    def export_trace(self, particle: smsh5.Particle):
        p_name = particle.unique_name
        tr_path = os.path.join(self.f_dir, p_name + " trace.csv")
        ints = particle.binnedtrace.intdata
        times = particle.binnedtrace.inttimes / 1e3
        self._export_trace(ints=ints, particle=particle, times=times, tr_path=tr_path)

        if self.options.use_roi:
            tr_path = os.path.join(self.f_dir, p_name + " trace (ROI).csv")
            roi_filter = (particle.roi_region[0] > times) ^ (
                times <= particle.roi_region[1]
            )
            roi_ints = ints[roi_filter]
            roi_times = times[roi_filter]
            self._export_trace(
                ints=roi_ints, particle=particle, times=roi_times, tr_path=tr_path
            )

    def export_corr(self, tr_path: str, particle: smsh5.Particle):
        bins = particle.ab_analysis.corr_bins
        hist = particle.ab_analysis.corr_hist / 1e3
        rows = list()
        rows.append(["Bin #", "Bin Time (ns)", f"Correlation (counts/bin)"])
        for i in range(len(bins)):
            rows.append([str(i + 1), str(bins[i]), str(hist[i])])
        with self._open_file(tr_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    ##############################################################
    # Plots
    ##############################################################

    def plot_lifetimes(
        self,
        particle: smsh5.Particle,
    ) -> None:
        with_fit = self.options.ex_plot_lifetimes_with_fit
        only_groups = self.options.ex_plot_lifetimes_only_groups

        p_name = particle.unique_name
        hist_path = os.path.join(self.f_dir, p_name + " hists")

        try:
            os.mkdir(hist_path)
        except FileExistsError:
            pass
        if particle.has_levels:
            # None -> Particle Histogram
            levels_and_groups: List[Union[None, Level, GlobalLevel, Group]] = [None]
            if not only_groups:
                if particle.has_levels:
                    levels_and_groups.extend(particle.cpts.levels)
                if particle.has_groups:
                    levels_and_groups.extend(particle.ahca.selected_step.group_levels)
            if particle.has_groups:
                levels_and_groups.extend(particle.groups)
            for l_or_g in levels_and_groups:
                path = self.f_dir if l_or_g is None else hist_path
                if self.signals:
                    self.signals.plot_decay_export_lock.emit(
                        l_or_g, particle, False, True, hist_path, True
                    )
                    self.lock.acquire()
                    while self.lock.locked():
                        sleep(0.1)
                else:
                    self.mw.lifetime_controller.plot_decay(
                        selected_level_or_group=l_or_g,
                        particle=particle,
                        for_export=True,
                        export_path=path if not with_fit else None,
                    )
                    if with_fit:
                        self.mw.lifetime_controller.plot_convd(
                            selected_level_or_group=l_or_g,
                            particle=particle,
                            for_export=True,
                            export_path=path,
                        )

    def plot_raster_scan(self, p, raster_scan):
        if self.signals:
            # with lock:
            self.signals.plot_raster_scan_export_lock.emit(
                p, raster_scan, True, self.f_dir, True
            )
            self.lock.acquire()
            while self.lock.locked():
                sleep(0.1)
        self.mw.raster_scan_controller.plot_raster_scan(
            raster_scan=raster_scan, for_export=True, export_path=self.f_dir
        )

    def plot_corr_hists(
        self,
        particle: smsh5.Particle,
    ):
        if self.signals:
            self.signals.plot_corr_export_lock.emit(particle, True, self.f_dir, True)
            self.lock.acquire()
            while self.lock.locked():
                sleep(0.1)
        else:
            self.mw.antibunch_controller.plot_corr_hists(
                particle=particle, for_export=True, export_path=self.f_dir
            )

    def plot_spectra(
        self,
        particle: smsh5.Particle,
    ):
        if self.signals:
            self.signals.plot_spectra_export_lock.emit(particle, True, self.f_dir, True)
            self.lock.acquire()
            while self.lock.locked():
                sleep(0.1)
        else:
            self.mw.spectra_controller.plot_spectra(
                particle=particle, for_export=True, export_path=self.f_dir
            )

    def plot_lifetime_fit_residuals(
        self,
        particle: smsh5.Particle,
        only_groups: bool = False,
    ):
        p_name = particle.unique_name
        hist_path = os.path.join(self.f_dir, p_name + " hists")
        try:
            os.mkdir(hist_path)
        except FileExistsError:
            pass
        if particle.has_levels:
            # None -> Particle Histogram
            levels_and_groups: List[Union[None, Level, GlobalLevel, Group]] = []
            if particle.histogram.fitted:
                levels_and_groups.append(None)
            if not only_groups:
                if particle.has_levels:
                    levels_and_groups.extend(
                        [l for l in particle.cpts.levels if l.histogram.fitted]
                    )
                if particle.has_groups:
                    levels_and_groups.extend(
                        [
                            gl
                            for gl in particle.ahca.selected_step.group_levels
                            if gl.histogram.fitted
                        ]
                    )
            if particle.has_groups:
                levels_and_groups.extend(
                    [g for g in particle.groups if g.histogram.fitted]
                )
            for l_or_g in levels_and_groups:
                path = self.f_dir if l_or_g is None else hist_path
                if self.signals:
                    self.signals.plot_residuals_export_lock.emit(
                        l_or_g, particle, True, path, True
                    )
                    self.lock.acquire()
                    while self.lock.locked():
                        sleep(0.1)
                else:
                    self.mw.lifetime_controller.plot_decay(
                        selected_level_or_group=l_or_g,
                        particle=particle,
                        for_export=True,
                        export_path=path,
                    )

    def plot_grouping_bic(
        self,
        particle: smsh5.Particle,
    ):
        if self.signals:
            self.signals.plot_grouping_bic_export_lock.emit(
                particle, True, self.f_dir, True
            )
            self.lock.acquire()
            while self.lock.locked():
                sleep(0.1)
        else:
            self.mw.grouping_controller.plot_group_bic(
                particle=particle, for_export=True, export_path=self.f_dir
            )

    def plot_levels(
        self,
        particle: smsh5.Particle,
        plot_groups: bool = False,
    ):
        if self.signals:
            self.signals.plot_trace_lock.emit(particle, True, True)
            self.lock.acquire()
            while self.lock.locked():
                sleep(0.1)
            self.signals.plot_levels_lock.emit(particle, True, True)
            self.lock.acquire()
            while self.lock.locked():
                sleep(0.1)
            if plot_groups:
                self.signals.plot_group_bounds_export_lock.emit(
                    particle, True, self.f_dir, True
                )
                self.lock.acquire()
                while self.lock.locked():
                    sleep(0.1)
        else:
            self.mw.intensity_controller.plot_trace(particle=particle, for_export=True)
            self.mw.intensity_controller.plot_levels(particle=particle, for_export=True)
            if plot_groups:
                self.mw.intensity_controller.plot_group_bounds(
                    particle=particle, for_export=True, export_path=self.f_dir
                )

    def plot_intensities(
        self,
        particle: smsh5.Particle,
    ):
        if self.signals:
            self.signals.plot_trace_export_lock.emit(particle, True, self.f_dir, True)
            self.lock.acquire()
            while self.lock.locked():
                sleep(0.1)
        else:
            self.mw.intensity_controller.plot_trace(
                particle=particle, for_export=True, export_path=self.f_dir
            )
