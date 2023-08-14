from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, List, Union
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
            export_data(
                main_window=self.main_window,
                mode=self.mode,
                signals=self.signals,
                lock=self.lock,
                f_dir=self.f_dir,
            )
        except Exception as err:
            self.signals.error.emit(err)
        finally:
            self.signals.fitting_finished.emit(self.mode)


def open_file(path: str):
    return open(path, "w", newline="")


def export_data(
    main_window: MainWindow,
    mode: str = None,
    signals: WorkerSignals = None,
    lock: Lock = None,
    f_dir: str = None,
):
    assert mode in [
        "current",
        "selected",
        "all",
    ], "MainWindow\tThe mode parameter is invalid"

    if mode == "current":
        particles = [main_window.current_particle]
    elif mode == "selected":
        particles = main_window.get_checked_particles()
    else:
        particles = main_window.current_dataset.particles

    f_dir = os.path.abspath(f_dir)

    if not f_dir:
        return
    else:
        try:
            raster_scans_use = [part.raster_scan.dataset_index for part in particles]
            raster_scans_use = np.unique(raster_scans_use).tolist()
        except AttributeError:
            raster_scans_use = []

    if not lock:
        lock = Lock()

    options = get_options(main_window=main_window)

    if signals:
        prog_num = 0
        if options["any_particle_text_plot"]:
            prog_num = prog_num + len(particles)
        if options["ex_raster_scan_2d"] or options["ex_plot_raster_scans"]:
            prog_num = prog_num + len(raster_scans_use)
        if options["ex_df_levels"]:
            prog_num = prog_num + 1
        if options["ex_df_grouped_levels"]:
            prog_num = prog_num + 1
        if options["ex_df_grouping_info"]:
            prog_num = prog_num + 1

        signals.start_progress.emit(prog_num)
        signals.status_message.emit(f"Exporting data for {mode} particles...")

    logger.info("Export finished")

    # any_particle_text_plot = any([any_particle_text_plot, ex_raster_scan_2d, ex_plot_raster_scans])

    # Export fits of whole traces
    lifetime_path = os.path.join(f_dir, "Whole trace lifetimes.csv")
    all_fitted = [part._histogram.fitted for part in particles]
    if options["ex_lifetime"] and any(all_fitted):
        export_lifetimes(f_dir=lifetime_path, particle_s=particles, whole_trace=True)

    lifetime_path = os.path.join(f_dir, "Whole trace lifetimes (ROI).csv")
    all_fitted = [part._histogram_roi.fitted for part in particles]
    if options["ex_lifetime"] and any(all_fitted):
        export_lifetimes(
            f_dir=lifetime_path, particle_s=particles, use_roi=True, whole_trace=True
        )

    # Export data for levels
    if options["any_particle_text_plot"]:
        for num, p in enumerate(particles):
            pname = p.unique_name
            if options["ex_traces"]:
                export_trace(f_dir=f_dir, particle=p, use_roi=options["use_roi"])

            if options["ex_corr_hists"]:
                export_corr_hists(f_dir=f_dir, particle=p, use_roi=options["use_roi"])

            if options["ex_plot_intensities"] and options["ex_plot_int_only"]:
                plot_intensities(
                    f_dir=f_dir,
                    lock=lock,
                    main_window=main_window,
                    particle=p,
                    signals=signals,
                )

            if options["ex_levels"] and p.has_levels:
                export_levels(f_dir=f_dir, particle=p, use_roi=options["use_roi"])

            if options["ex_plot_intensities"] and options["ex_plot_with_levels"]:
                if p.has_levels:
                    plot_levels(
                        f_dir=f_dir,
                        lock=lock,
                        main_window=main_window,
                        particle=p,
                        signals=signals,
                    )

            if options["ex_grouped_levels"] and p.has_groups:
                export_levels_grouped_plot(f_dir=f_dir, particle=p)

            if options["ex_global_grouped_levels"] and p.has_global_grouping:
                export_levels_global_grouped_plot(f_dir=f_dir, particle=p)

            if (
                options["ex_plot_intensities"]
                and options["ex_plot_and_groups"]
                and p.has_groups
            ):
                plot_levels(
                    f_dir=f_dir,
                    lock=lock,
                    main_window=main_window,
                    particle=p,
                    signals=signals,
                    plot_groups=True,
                )

            if options["ex_grouping_info"] and p.has_groups:
                export_grouping_info(f_dir=f_dir, particle=p)

            if options["ex_grouping_results"] and p.has_groups:
                export_grouping_results(f_dir=f_dir, particle=p)

            if options["ex_plot_grouping_bics"]:
                plot_grouping_bic(
                    f_dir=f_dir,
                    lock=lock,
                    main_window=main_window,
                    particle=p,
                    signals=signals,
                )

            if options["ex_lifetime"]:
                export_lifetimes(f_dir=f_dir, particle_s=p, use_roi=options["use_roi"])

            if options["ex_hist"]:
                export_hists(f_dir=f_dir, particle=p, use_roi=options["use_roi"])

            # TODO: Fix problems
            # Current problems
            # 1. When use_roi is off, only levels in roi print
            # 2. Hist only looks good, but all other options go into E-14 range in y. Looks bad
            # 3. Some plots are empty when fit is included
            if options["ex_plot_lifetimes"]:
                plot_lifetimes(
                    f_dir=f_dir,
                    lock=lock,
                    main_window=main_window,
                    particle=p,
                    signals=signals,
                    with_fit=options["ex_plot_with_fit"],
                    only_groups=options["ex_plot_lifetimes_only_groups"],
                )

            if options["ex_plot_and_residuals"]:
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
                    main_window.lifetime_controller.plot_decay(
                        selected_level_or_group=-1, particle=p, for_export=True
                    )
                    main_window.lifetime_controller.plot_convd(
                        selected_level_or_group=-1,
                        particle=p,
                        for_export=True,
                        export_path=f_dir,
                    )
                write_hists(
                    options["ex_plot_lifetimes_only_groups"],
                    f_dir,
                    lock,
                    p,
                    signals,
                    residuals=True,
                )

            if options["ex_spectra_2d"]:
                spectra_2d_path = os.path.join(f_dir, pname + " spectra-2D.csv")
                with open_file(spectra_2d_path) as f:
                    f.write("First row:,Wavelength (nm)\n")
                    f.write("First column:,Time (s)\n")
                    f.write("Values:,Intensity (counts/s)\n\n")

                    rows = list()
                    rows.append([""] + p.spectra.wavelengths.tolist())
                    for num, spec_row in enumerate(p.spectra.data[:]):
                        this_row = list()
                        this_row.append(str(p.spectra.series_times[num]))
                        for single_val in spec_row:
                            this_row.append(str(single_val))
                        rows.append(this_row)

                    writer = csv.writer(f, dialect=csv.excel)
                    writer.writerows(rows)

            if options["ex_plot_spectra"]:
                if signals:
                    signals.plot_spectra_export_lock.emit(p, True, f_dir, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                else:
                    main_window.spectra_controller.plot_spectra(
                        particle=p, for_export=True, export_path=f_dir
                    )
            if options["ex_plot_corr_hists"]:
                if signals:
                    signals.plot_corr_export_lock.emit(p, True, f_dir, True)
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                else:
                    main_window.antibunch_controller.plot_corr(
                        particle=p, for_export=True, export_path=f_dir
                    )

            logger.info("Exporting Finished")
            if signals:
                signals.progress.emit()
            p.has_exported = True

        # if ex_raster_scan_2d:
        #     dataset = mainwindow.current_dataset
        #     for raster_scan_index in raster_scans_use:
        #         raster_scan = dataset.all_raster_scans[raster_scan_index]
        #         if signals:
        #             signals.progress.emit()

    if options["ex_raster_scan_2d"] or options["ex_plot_raster_scans"]:
        dataset = main_window.current_dataset
        for raster_scan_index in raster_scans_use:
            raster_scan = dataset.all_raster_scans[raster_scan_index]
            rs_part_ind = raster_scan.particle_indexes[0]
            p = dataset.particles[rs_part_ind]
            if options["ex_raster_scan_2d"]:
                raster_scan_2d_path = os.path.join(
                    f_dir, f"Raster Scan {raster_scan.dataset_index + 1} data.csv"
                )
                top_row = [np.NaN, *raster_scan.x_axis_pos]
                y_and_data = np.column_stack(
                    (raster_scan.y_axis_pos, raster_scan.dataset[:])
                )
                x_y_data = np.insert(y_and_data, 0, top_row, axis=0)
                with open_file(raster_scan_2d_path) as f:
                    f.write("Rows = X-Axis (um)")
                    f.write("Columns = Y-Axis (um)")
                    f.write("")
                    writer = csv.writer(f, dialect=csv.excel)
                    writer.writerows(x_y_data)

            if options["ex_plot_raster_scans"]:
                if signals:
                    # with lock:
                    signals.plot_raster_scan_export_lock.emit(
                        p, raster_scan, True, f_dir, True
                    )
                    lock.acquire()
                    while lock.locked():
                        sleep(0.1)
                main_window.raster_scan_controller.plot_raster_scan(
                    raster_scan=raster_scan, for_export=True, export_path=f_dir
                )

            if signals:
                signals.progress.emit()

    ## DataFrame compilation and writing
    if any(
        [
            options["ex_df_levels"],
            options["ex_df_grouped_levels"],
            options["ex_df_grouping_info"],
        ]
    ):
        any_has_lifetime = any([p.has_fit_a_lifetime for p in particles])
        if any_has_lifetime:
            max_numexp = max([p.numexp for p in particles if p.numexp is not None])
            tau_cols = [f"tau_{i + 1}" for i in range(max_numexp)]
            taustd_cols = [f"tau_std_{i + 1}" for i in range(max_numexp)]
            amp_cols = [f"amp_{i + 1}" for i in range(max_numexp)]
            ampstd_cols = [f"amp_std_{i + 1}" for i in range(max_numexp)]
            life_cols_add = [
                "num_photons_in_lifetime_fit",
                *tau_cols,
                *taustd_cols,
                *amp_cols,
                *ampstd_cols,
                "irf_shift",
                "irf_shift_std",
                "decay_bg",
                "irf_bg",
                "chi_squared",
                "dw",
                "dw_5",
                "dw_1",
                "dw_03",
                "dw_01",
            ]
        else:
            life_cols_add = [""]
            max_numexp = None
        if options["ex_df_levels"] or options["ex_df_grouped_levels"]:
            levels_cols = [
                "particle",
                "is_primary_part",
                "tcspc_card",
                "level",
                "start",
                "end",
                "dwell",
                "dwell_frac",
                "int",
                "num_photons",
            ]
            grouped_levels_cols = levels_cols.copy()
            # grouped_levels_cols[1] = 'grouped_level'
            grouped_levels_cols.insert(2, "group_index")
            if any_has_lifetime:
                if options["ex_df_levels_lifetimes"]:
                    levels_cols.extend(life_cols_add)
                if options["ex_df_grouped_levels_lifetimes"]:
                    grouped_levels_cols.extend(life_cols_add)
            levels_cols.append("is_in_roi")
            grouped_levels_cols.append("is_in_roi")

            data_levels = list()
            if options["ex_df_grouped_levels"]:
                data_grouped_levels = list()

        if options["ex_df_grouping_info"]:
            grouping_info_cols = [
                "particle",
                "is_primary_part",
                "tcspc_card",
                "group",
                "total_dwell",
                "int",
                "num_levels",
                "num_photons",
                "num_steps",
                "is_best_step",
            ]
            data_grouping_info = list()

        for p in particles:
            if not p.has_levels:
                continue
            roi_first_level_ind = p.first_level_ind_in_roi
            roi_last_level_ind = p.last_level_ind_in_roi
            pname = p.unique_name
            if options["ex_df_levels"]:
                for l_num, l in enumerate(p.cpts.levels):
                    level_in_roi = roi_first_level_ind <= l_num <= roi_last_level_ind
                    row = [
                        pname,
                        not p.is_secondary_part,
                        p.tcspc_card,
                        l_num + 1,
                        *get_level_data(
                            l,
                            p.dwell_time,
                            incl_lifetimes=all(
                                [
                                    options["ex_df_levels_lifetimes"],
                                    p.has_fit_a_lifetime,
                                ]
                            ),
                            max_numexp=max_numexp,
                        ),
                        level_in_roi,
                    ]
                    data_levels.append(row)

            if options["ex_df_grouped_levels"]:
                roi_first_group_level_ind = p.first_group_level_ind_in_roi
                roi_last_group_level_ind = p.last_group_level_ind_in_roi
                for g_l_num, g_l in enumerate(p.group_levels):
                    group_level_in_roi = (
                        roi_first_group_level_ind <= g_l_num <= roi_last_group_level_ind
                    )
                    row = [
                        pname,
                        not p.is_secondary_part,
                        p.tcspc_card,
                        g_l_num + 1,
                        g_l.group_ind + 1,
                        *get_level_data(
                            g_l,
                            p.dwell_time,
                            incl_lifetimes=all(
                                [
                                    options["ex_df_grouped_levels_lifetimes"],
                                    p.has_fit_a_lifetime,
                                ]
                            ),
                            max_numexp=max_numexp,
                        ),
                        group_level_in_roi,
                    ]
                    data_grouped_levels.append(row)

            if options["ex_df_grouping_info"]:
                if p.has_groups:
                    for g_num, g in enumerate(p.ahca.selected_step.groups):
                        row = [
                            pname,
                            not p.is_secondary_part,
                            p.tcspc_card,
                            g_num + 1,
                            g.int_p_s,
                            g.dwell_time_s,
                            len(g.lvls),
                            g.num_photons,
                            p.ahca.num_steps,
                            p.ahca.selected_step == p.ahca.best_step_ind,
                        ]
                        data_grouping_info.append(row)
                else:
                    row = [pname]
                    row.extend([np.NaN] * 7)
                    data_grouping_info.append(row)

        if options["ex_df_levels"]:
            df_levels = pd.DataFrame(data=data_levels, columns=levels_cols)
            df_levels["particle"] = df_levels["particle"].astype("string")
            # levels_df_path = os.path.join(f_dir, 'levels.df')
            # feather.write_feather(df=df_levels, dest=levels_df_path)
            write_dataframe_to_file(
                dataframe=df_levels,
                path=f_dir,
                filename="levels",
                file_type=options["ex_df_format"],
            )
            if signals:
                signals.progress.emit()

        if options["ex_df_grouped_levels"]:
            df_grouped_levels = pd.DataFrame(
                data=data_grouped_levels, columns=grouped_levels_cols
            )
            df_grouped_levels["particle"] = df_grouped_levels.particle.astype("string")
            # grouped_levels_df_path = os.path.join(f_dir, 'grouped_levels.df')
            # feather.write_feather(df=df_grouped_levels, dest=grouped_levels_df_path)
            write_dataframe_to_file(
                dataframe=df_grouped_levels,
                path=f_dir,
                filename="grouped_levels",
                file_type=options["ex_df_format"],
            )
            if signals:
                signals.progress.emit()

        if options["ex_df_grouping_info"]:
            df_grouping_info = pd.DataFrame(
                data=data_grouping_info, columns=grouping_info_cols
            )
            # grouping_info_df_path = os.path.join(f_dir, 'grouping_info.df')
            # feather.write_feather(df=df_grouping_info, dest=grouping_info_df_path)
            write_dataframe_to_file(
                dataframe=df_grouping_info,
                path=f_dir,
                filename="grouping_info",
                file_type=options["ex_df_format"],
            )
            if signals:
                signals.progress.emit()

    if signals:
        signals.end_progress.emit()
        signals.status_message.emit("Done")


def get_options(main_window: MainWindow) -> dict:
    options = dict()
    options["use_roi"] = main_window.chbEx_Use_ROI.isChecked()

    options["ex_traces"] = main_window.chbEx_Trace.isChecked()
    options["ex_levels"] = main_window.chbEx_Levels.isChecked()
    options["ex_plot_intensities"] = main_window.chbEx_Plot_Intensity.isChecked()
    options["ex_plot_with_levels"] = False
    options["ex_plot_and_groups"] = False
    options["ex_plot_int_only"] = False
    if options["ex_plot_intensities"]:
        options["ex_plot_int_only"] = main_window.rdbInt_Only.isChecked()
        if not options["ex_plot_int_only"]:
            if main_window.rdbWith_Levels.isChecked():
                options["ex_plot_with_levels"] = True
            else:
                options["ex_plot_and_groups"] = True
    options["ex_grouped_levels"] = main_window.chbEx_Grouped_Levels.isChecked()
    options[
        "ex_global_grouped_levels"
    ] = main_window.chbEx_Global_Grouped_Levels.isChecked()
    options["ex_grouping_info"] = main_window.chbEx_Grouping_Info.isChecked()
    options["ex_grouping_results"] = main_window.chbEx_Grouping_Results.isChecked()
    options["ex_plot_grouping_bics"] = main_window.chbEx_Plot_Group_BIC.isChecked()
    options["ex_lifetime"] = main_window.chbEx_Lifetimes.isChecked()
    options["ex_hist"] = main_window.chbEx_Hist.isChecked()
    options["ex_plot_lifetimes"] = main_window.chbEx_Plot_Lifetimes.isChecked()
    options["ex_plot_with_fit"] = False
    options["ex_plot_and_residuals"] = False
    options["ex_plot_hist_only"] = False
    options["ex_plot_lifetimes_only_groups"] = False
    if options["ex_plot_lifetimes"]:
        options["ex_plot_hist_only"] = main_window.rdbHist_Only.isChecked()
        if not options["ex_plot_hist_only"]:
            if main_window.rdbWith_Fit.isChecked():
                options["ex_plot_with_fit"] = True
            else:
                options["ex_plot_and_residuals"] = True
        options[
            "ex_plot_lifetimes_only_groups"
        ] = main_window.chbEx_Plot_Lifetimes_Only_Groups.isChecked()
    options["ex_spectra_2d"] = main_window.chbEx_Spectra_2D.isChecked()
    options["ex_plot_spectra"] = main_window.chbEx_Plot_Spectra.isChecked()
    options["ex_raster_scan_2d"] = main_window.chbEx_Raster_Scan_2D.isChecked()
    options["ex_plot_raster_scans"] = main_window.chbEx_Plot_Raster_Scans.isChecked()
    options["ex_corr_hists"] = main_window.chbEx_Corr.isChecked()
    options["ex_plot_corr_hists"] = main_window.chbEx_Plot_Corr.isChecked()

    options["ex_df_levels"] = main_window.chbEx_DF_Levels.isChecked()
    options[
        "ex_df_levels_lifetimes"
    ] = main_window.chbEx_DF_Levels_Lifetimes.isChecked()
    options["ex_df_grouped_levels"] = main_window.chbEx_DF_Grouped_Levels.isChecked()
    options[
        "ex_df_grouped_levels_lifetimes"
    ] = main_window.chbEx_DF_Grouped_Levels_Lifetimes.isChecked()
    options["ex_df_grouping_info"] = main_window.chbEx_DF_Grouping_Info.isChecked()

    options["ex_df_format"] = main_window.cmbEx_DataFrame_Format.currentIndex()

    options["any_particle_text_plot"] = any(
        [
            options["ex_traces"],
            options["ex_levels"],
            options["ex_plot_intensities"],
            options["ex_grouped_levels"],
            options["ex_global_grouped_levels"],
            options["ex_grouping_info"],
            options["ex_grouping_results"],
            options["ex_plot_grouping_bics"],
            options["ex_lifetime"],
            options["ex_hist"],
            options["ex_plot_lifetimes"],
            options["ex_spectra_2d"],
            options["ex_plot_spectra"],
            options["ex_corr_hists"],
            options["ex_plot_corr_hists"],
        ]
    )

    return options


# def export_lifetime_plots(
#     f_dir: str,
#     particle: smsh5.Particle,
#     lock: Lock,
#     main_window: MainWindow,
#     signals: WorkerSignals = None,
#     with_fit: bool = False,
#     only_groups: bool = False,
# ):
#     if signals:
#         signals.plot_decay_lock.emit(-1, particle, False, True, True)
#         lock.acquire()
#         while lock.locked():
#             sleep(0.1)
#         signals.plot_convd_export_lock.emit(-1, particle, False, True, f_dir, True)
#         lock.acquire()
#         while lock.locked():
#             sleep(0.1)
#     else:
#         main_window.lifetime_controller.plot_decay(
#             selected_level_or_group=-1, particle=particle, for_export=True
#         )
#         main_window.lifetime_controller.plot_convd(
#             selected_level_or_group=-1,
#             particle=particle,
#             for_export=True,
#             export_path=f_dir,
#         )
#     write_hists(only_groups, f_dir, lock, particle, signals)


def plot_lifetimes(
    f_dir: str,
    lock: Lock,
    main_window: MainWindow,
    particle: smsh5.Particle,
    signals: WorkerSignals = None,
    with_fit: bool = False,
    only_groups: bool = False,
) -> None:
    p_name = particle.unique_name
    hist_path = os.path.join(f_dir, p_name + " hists")
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
            path = f_dir if l_or_g is None else hist_path
            if signals:
                signals.plot_decay_export_lock.emit(
                    l_or_g, particle, False, True, hist_path, True
                )
                lock.acquire()
                while lock.locked():
                    sleep(0.1)
            else:
                main_window.lifetime_controller.plot_decay(
                    selected_level_or_group=l_or_g,
                    particle=particle,
                    for_export=True,
                    export_path=path if not with_fit else None,
                )
                if with_fit:
                    main_window.lifetime_controller.plot_convd(
                        selected_level_or_group=l_or_g,
                        particle=particle,
                        for_export=True,
                        export_path=path,
                    )


def export_hists(f_dir: str, particle: smsh5.Particle, use_roi: bool = False):
    def _export_hist(tr_path: str, particle: smsh5.Particle, roi: bool = False):
        rows = list()
        if roi:
            histogram = particle._histogram_roi
        else:
            histogram = particle._histogram
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
        with open_file(tr_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    p_name = particle.unique_name
    tr_path = os.path.join(f_dir, p_name + " hist.csv")
    _export_hist(open_file, particle, tr_path)
    if use_roi:
        tr_path = os.path.join(f_dir, p_name + " hist (ROI).csv")
        _export_hist(open_file, particle, tr_path, roi=True)
    if particle.has_levels:
        dir_path = os.path.join(f_dir, p_name + " hists")
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
        roi_start_ind = particle.first_level_ind_in_roi
        roi_end_ind = particle.last_level_ind_in_roi
        for i, l in enumerate(particle.cpts.levels):
            roi_tag = " (ROI)" if use_roi and roi_start_ind <= i <= roi_end_ind else ""
            hist_path = os.path.join(
                dir_path, "level " + str(i + 1) + roi_tag + " hist.csv"
            )
            _export_hist(open_file, l, hist_path, level=True)

        if particle.has_groups:
            roi_start_ind = particle.first_group_level_ind_in_roi
            roi_end_ind = particle.last_group_level_ind_in_roi
            for i, g in enumerate(particle.group_levels):
                roi_tag = (
                    " (ROI)" if use_roi and roi_start_ind <= i <= roi_end_ind else ""
                )
                hist_path = os.path.join(
                    dir_path,
                    "group level " + str(i + 1) + roi_tag + " hist.csv",
                )
                _export_hist(open_file, g, hist_path, level=True)

            for i, g in enumerate(particle.groups):
                hist_path = os.path.join(dir_path, "group " + str(i + 1) + " hist.csv")
                _export_hist(open_file, g, hist_path, level=True)


def export_lifetimes(
    f_dir: str,
    particle_s: Union[smsh5.Particle, List[smsh5.Particle]],
    use_roi: bool = False,
    whole_trace: bool = False,
) -> None:
    def _export_lifetimes(
        lifetime_path: str, particles: List[smsh5.Particle], roi=False, levels=False
    ):
        max_exp_number = np.max(
            [
                np.max([p.histogram.numexp for p in particles]),
                np.max([[l.histogram.numexp for l in p.levels] for p in particles]),
                np.max([[g.histogram.numexp for g in p.groups] for p in particles]),
            ]
        )

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
            ]
        )

        def pad(list_to_pad: list, total_len: int = 3) -> list:
            l_padded = list_to_pad.copy()
            if len(list_to_pad) < total_len:
                l_padded.extend([None] * (total_len - len(list_to_pad)))
            return l_padded

        for i, p in enumerate(particles):
            p_name = p.unique_name
            tau_std_exp = None
            amp_std_exp = None
            if levels:
                histogram = p.histogram
            elif roi:
                histogram = p._histogram_roi
            else:
                histogram = p._histogram
            if histogram.fitted:
                if (
                    histogram.tau is None or histogram.amp is None
                ):  # Problem with fitting the level
                    tau_exp = pad(["0" for i in range(histogram.numexp)])
                    amp_exp = pad(["0" for i in range(histogram.numexp)])
                    other_exp = ["0", "0", "0", "0"]
                else:
                    num_exp = np.size(histogram.tau)
                    if num_exp == 1:
                        tau_exp = pad([str(histogram.tau)])
                        tau_std_exp = pad([str(histogram.stds[0])])
                        amp_exp = pad([str(histogram.amp)])
                        amp_std_exp = pad([str(0)])
                    else:
                        tau_exp = pad([str(tau) for tau in histogram.tau])
                        tau_std_exp = pad(
                            [str(std) for std in histogram.stds[:num_exp]]
                        )
                        amp_exp = pad([str(amp) for amp in histogram.amp])
                        amp_std_exp = pad(
                            [str(std) for std in histogram.stds[num_exp : 2 * num_exp]]
                        )
                    if hasattr(histogram, "fwhm") and histogram.fwhm is not None:
                        sim_irf_fwhm = str(histogram.fwhm)
                        sim_irf_fwhm_std = str(histogram.stds[2 * num_exp + 1])
                    else:
                        sim_irf_fwhm = ""
                        sim_irf_fwhm_std = ""
                    other_exp = [
                        str(histogram.avtau),
                        str(histogram.avtaustd),
                        str(histogram.shift),
                        str(histogram.stds[2 * num_exp]),
                        str(histogram.bg),
                        str(histogram.irfbg),
                        str(histogram.chisq),
                        sim_irf_fwhm,
                        sim_irf_fwhm_std,
                        str(histogram.dw),
                        str(histogram.dw_bound[0]),
                        str(histogram.dw_bound[1]),
                        str(histogram.dw_bound[2]),
                        str(histogram.dw_bound[3]),
                    ]
                if levels:
                    p_num = [str(i + 1)]
                else:  # get number from particle name
                    p_num = re.findall(r"\d+", p_name) + [
                        str(int(not p.is_secondary_part))
                    ]
                rows.append(
                    p_num + tau_exp + tau_std_exp + amp_exp + amp_std_exp + other_exp
                )
        with open_file(lifetime_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    if type(particle_s) is smsh5.Particle:
        particle_s = [particle_s]

    if whole_trace:
        p = particle_s[0]
        lifetime_path = os.path.join(f_dir, "Whole trace lifetimes.csv")
        all_fitted = [part._histogram.fitted for part in particle_s]
        if any(all_fitted):
            _export_lifetimes(lifetime_path=lifetime_path, particles=p)

        lifetime_path = os.path.join(f_dir, "Whole trace lifetimes (ROI).csv")
        all_fitted = [part._histogram_roi.fitted for part in particle_s]
        if any(all_fitted):
            _export_lifetimes(lifetime_path=lifetime_path, particles=p, roi=True)

    else:
        for p in particle_s:
            p_name = p.unique_name
            all_fitted_lvls = [lvl.histogram.fitted for lvl in p.cpts.levels]
            if p.has_levels and any(all_fitted_lvls):
                lvl_path = os.path.join(f_dir, p_name + " levels-lifetimes.csv")
                _export_lifetimes(lvl_path, p.cpts.levels, levels=True)

                all_fitted_grps = [grp.histogram.fitted for grp in p.groups]
                if p.has_groups and any(all_fitted_grps):
                    group_path = os.path.join(f_dir, p_name + " groups-lifetimes")
                    if not p.grouped_with_roi:
                        group_path += ".csv"
                    else:
                        group_path += " (ROI).csv"
                    _export_lifetimes(group_path, p.groups, levels=True)
            if use_roi:
                all_fitted_lvls_roi = [lvl.histogram.fitted for lvl in p.levels_roi]
                if p.has_levels and any(all_fitted_lvls_roi):
                    lvl_path = os.path.join(
                        f_dir, p_name + " levels-lifetimes (ROI).csv"
                    )
                    _export_lifetimes(lvl_path, p.levels_roi, levels=True)


def plot_grouping_bic(
    f_dir: str,
    lock: Lock,
    main_window: MainWindow,
    particle: smsh5.Particle,
    signals: WorkerSignals = None,
):
    if signals:
        signals.plot_grouping_bic_export_lock.emit(particle, True, f_dir, True)
        lock.acquire()
        while lock.locked():
            sleep(0.1)
    else:
        main_window.grouping_controller.plot_group_bic(
            particle=particle, for_export=True, export_path=f_dir
        )


def export_grouping_results(f_dir: str, particle: smsh5.Particle):
    pname = particle.unique_name
    grouping_results_path = os.path.join(f_dir, pname + " grouping-results")
    if not particle.grouped_with_roi:
        grouping_results_path += ".csv"
    else:
        grouping_results_path += " (ROI).csv"
    with open_file(grouping_results_path) as f:
        f.write(f"# of Steps:,{particle.ahca.num_steps}\n")
        f.write(f"Step with highest BIC value:,{particle.ahca.best_step.bic}\n")
        f.write(f"Step selected:,{particle.ahca.selected_step_ind}\n\n")

        rows = list()
        rows.append(["Step #", "# of Groups", "BIC value"])
        for num, step in enumerate(particle.ahca.steps):
            rows.append([str(num + 1), str(step.num_groups), str(step.bic)])

        writer = csv.writer(f, dialect=csv.excel)
        writer.writerows(rows)


def export_grouping_info(f_dir: str, particle: smsh5.Particle):
    pname = particle.unique_name
    group_info_path = os.path.join(f_dir, pname + " groups-info")
    if not particle.grouped_with_roi:
        group_info_path += ".csv"
    else:
        group_info_path += " (ROI).csv"
    with open_file(group_info_path) as f:
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


def export_levels_global_grouped_plot(f_dir: str, particle: smsh5.Particle):
    pname = particle.unique_name
    grp_lvl_path = os.path.join(f_dir, pname + " levels-global-grouped.csv")
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
    with open_file(grp_lvl_path) as f:
        writer = csv.writer(f, dialect=csv.excel)
        writer.writerows(rows)


def export_levels_grouped_plot(f_dir: str, particle: smsh5.Particle):
    pname = particle.unique_name
    grp_lvl_tr_path = os.path.join(f_dir, pname + " levels-grouped-plot")
    if not particle.grouped_with_roi:
        grp_lvl_tr_path += ".csv"
    else:
        grp_lvl_tr_path += " (ROI).csv"
    ints, times = particle.levels2data(use_grouped=True)
    rows = list()
    rows.append(["Grouped Level #", "Time (s)", "Int (counts/s)"])
    for i in range(len(ints)):
        rows.append([str((i // 2) + 1), str(times[i]), str(ints[i])])
    with open_file(grp_lvl_tr_path) as f:
        writer = csv.writer(f, dialect=csv.excel)
        writer.writerows(rows)
    grp_lvl_path = os.path.join(f_dir, pname + " levels-grouped")
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
    with open_file(grp_lvl_path) as f:
        writer = csv.writer(f, dialect=csv.excel)
        writer.writerows(rows)


def plot_levels(
    f_dir: str,
    lock: Lock,
    main_window: MainWindow,
    particle: smsh5.Particle,
    signals: WorkerSignals = None,
    plot_groups: bool = False,
):
    if signals:
        signals.plot_trace_lock.emit(particle, True, True)
        lock.acquire()
        while lock.locked():
            sleep(0.1)
        signals.plot_levels_lock.emit(particle, True, True)
        lock.acquire()
        while lock.locked():
            sleep(0.1)
        if plot_groups:
            signals.plot_group_bounds_export_lock.emit(particle, True, f_dir, True)
            lock.acquire()
            while lock.locked():
                sleep(0.1)
    else:
        main_window.intensity_controller.plot_trace(particle=particle, for_export=True)
        main_window.intensity_controller.plot_levels(particle=particle, for_export=True)
        if plot_groups:
            main_window.intensity_controller.plot_group_bounds(
                particle=particle, for_export=True, export_path=f_dir
            )


def export_levels(f_dir: str, particle: smsh5.Particle, use_roi: bool):
    def _export_level_plot(
        ints: Union[list, np.ndarray], lvl_tr_path: str, times: Union[list, np.ndarray]
    ):
        rows = list()
        rows.append(["Level #", "Time (s)", "Int (counts/s)"])
        for i in range(len(ints)):
            rows.append([str((i // 2) + 1), str(times[i]), str(ints[i])])
        with open_file(lvl_tr_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    def _export_levels(lvl_path: str, particle: smsh5.Particle, roi: bool = False):
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
        with open_file(lvl_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    pname = particle.unique_name
    lvl_tr_path = os.path.join(f_dir, pname + " levels-plot.csv")
    ints, times = particle.levels2data(use_grouped=False, use_roi=False)
    _export_level_plot(ints=ints, lvl_tr_path=lvl_tr_path, times=times)
    if use_roi:
        lvl_tr_path = os.path.join(f_dir, pname + " levels-plot (ROI).csv")
        ints, times = particle.levels2data(use_grouped=False, use_roi=use_roi)
        _export_level_plot(ints=ints, lvl_tr_path=lvl_tr_path, times=times)
    lvl_path = os.path.join(f_dir, pname + " levels.csv")
    _export_levels(lvl_path=lvl_path, particle=particle)
    if use_roi:
        lvl_path = os.path.join(f_dir, pname + " levels (ROI).csv")
        _export_levels(lvl_path=lvl_path, particle=particle, roi=True)


def plot_intensities(
    f_dir: str,
    lock: Lock,
    main_window: MainWindow,
    particle: smsh5.Particle,
    signals: WorkerSignals = None,
):
    if signals:
        signals.plot_trace_export_lock.emit(particle, True, f_dir, True)
        lock.acquire()
        while lock.locked():
            sleep(0.1)
    else:
        main_window.intensity_controller.plot_trace(
            particle=particle, for_export=True, export_path=f_dir
        )


def export_corr_hists(f_dir: str, particle: smsh5.Particle, use_roi: bool = False):
    pname = particle.unique_name
    tr_path = os.path.join(f_dir, pname + " corr.csv")
    export_corr(tr_path=tr_path, particle=particle)
    if use_roi:
        pass


def export_trace(f_dir: str, particle: smsh5.Particle, use_roi: bool):
    def _export_trace(
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
        with open_file(tr_path) as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerows(rows)

    pname = particle.unique_name
    tr_path = os.path.join(f_dir, pname + " trace.csv")
    ints = particle.binnedtrace.intdata
    times = particle.binnedtrace.inttimes / 1e3
    _export_trace(ints=ints, particle=particle, times=times, tr_path=tr_path)

    if use_roi:
        tr_path = os.path.join(f_dir, pname + " trace (ROI).csv")
        roi_filter = (particle.roi_region[0] > times) ^ (
            times <= particle.roi_region[1]
        )
        roi_ints = ints[roi_filter]
        roi_times = times[roi_filter]
        _export_trace(
            ints=roi_ints, particle=particle, times=roi_times, tr_path=tr_path
        )


# def write_hists(
#     f_dir: str,
#     lock: Lock,
#     p: smsh5.Particle,
#     signals: WorkerSignals,
#     only_groups: bool = False,
#     residuals=False,
# ):
#     pname = p.unique_name
#     dir_path = os.path.join(f_dir, pname + " hists")
#     try:
#         os.mkdir(dir_path)
#     except FileExistsError:
#         pass
#     if p.has_levels:
#         args = p, dir_path, p.has_groups, only_groups, True
#         if not residuals:
#             signals.plot_decay_convd_export_lock.emit(*args)
#         else:
#             signals.plot_decay_convd_residuals_export_lock.emit(*args)
#         lock.acquire()
#         while lock.locked():
#             sleep(0.1)


def export_corr(tr_path: str, particle: smsh5.Particle):
    bins = particle.ab_analysis.corr_bins
    hist = particle.ab_analysis.corr_hist / 1e3
    rows = list()
    rows.append(["Bin #", "Bin Time (ns)", f"Correlation (counts/bin)"])
    for i in range(len(bins)):
        rows.append([str(i + 1), str(bins[i]), str(hist[i])])
    with open_file(tr_path) as f:
        writer = csv.writer(f, dialect=csv.excel)
        writer.writerows(rows)


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


def get_level_data(
    level: Level,
    total_dwelltime: float,
    incl_lifetimes: bool = False,
    max_numexp: int = 3,
) -> List:
    data = [
        *level.times_s,
        level.dwell_time_s,
        level.dwell_time_s / total_dwelltime,
        level.int_p_s,
        level.num_photons,
    ]
    if incl_lifetimes:
        h = level.histogram
        if h.fitted:
            data.append(h.num_photons_used)
            if h.numexp == 1:
                taus = [h.tau] if type(h.tau) is not list else h.tau
                taustds = [h.stds[0]]
                amps = [h.amp]
                ampstds = [h.stds[0]]
            else:
                taus = list(h.tau)
                taustds = list(h.stds[: h.numexp])
                amps = list(h.amp)
                ampstds = list(h.stds[h.numexp : 2 * h.numexp])

            taus.extend([np.NaN] * (max_numexp - h.numexp))
            data.extend(taus)
            taustds.extend([np.NaN] * (max_numexp - h.numexp))
            data.extend(taustds)

            amps.extend([np.NaN] * (max_numexp - h.numexp))
            data.extend(amps)
            ampstds.extend([np.NaN] * (max_numexp - h.numexp))
            data.extend(ampstds)

            if h.dw_bound is None:
                h.dw_bound = [None, None, None, None]
            data.extend(
                [
                    h.shift,
                    h.stds[2 * h.numexp],
                    h.bg,
                    h.irfbg,
                    h.chisq,
                    h.dw,
                    h.dw_bound[0],
                    h.dw_bound[1],
                    h.dw_bound[2],
                    h.dw_bound[3],
                ]
            )
        else:
            data.extend([np.NaN] * (9 + max_numexp))

    return data
