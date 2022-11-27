from __future__ import annotations

__docformat__ = 'NumPy'

from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QDialogButtonBox

import smsh5
from my_logger import setup_logger
import file_manager as fm
import pyqtgraph as pg
from PyQt5.QtGui import QPalette, QPen, QColor
import numpy as np
from dataclasses import dataclass
from typing import Union, List
import numpy as np

logger = setup_logger(__name__)

filtering_normalization_dialog_file = fm.path(
    name="filtering_and_normalization_dialog.ui",
    file_type=fm.Type.UI
)
UI_Filtering_Normalization_Dialog, _ = uic.loadUiType(filtering_normalization_dialog_file)


@dataclass
class PlotFeature:
    PhotonNumber = 'Photon Number'
    Intensity = 'Intensity'
    Lifetime = 'Lifetime'
    DW = 'DW'
    ChiSquared = 'Chi-Squared'
    IRFShift = 'IRF Shift'

    @classmethod
    def get_list(cls) -> list:
        return [cls.PhotonNumber, cls.Intensity, cls.Lifetime, cls.DW, cls.ChiSquared, cls.IRFShift]

    @classmethod
    def get_dict(cls) -> dict:
        all_dict = {
            str(cls.PhotonNumber): cls.PhotonNumber,
            str(cls.Intensity): cls.Intensity,
            str(cls.Lifetime): cls.Lifetime,
            str(cls.DW): cls.DW,
            str(cls.ChiSquared): cls.ChiSquared,
            str(cls.IRFShift): cls.IRFShift
        }
        return all_dict


class FilteringNormalizationDialog(QDialog, UI_Filtering_Normalization_Dialog):

    def __init__(self, main_window):
        QDialog.__init__(self)
        UI_Filtering_Normalization_Dialog.__init__(self)
        self.setupUi(self)

        self.main_window = main_window
        self.parent = main_window

        self.plot_widget = self.pgPlotWidget
        self.plot = self.plot_widget.getPlotItem()
        self.plot_widget.setBackground(background=None)
        self.plot_widget.setAntialiasing(True)

        self.fit_results_text = self.lblFitResults

        self.option_linker = {
            'min_photons': (self.chbMinPhotons, self.spnMinPhotons),
            'min_intensity': (self.chbMinIntensity, self.dsbMinIntensity),
            'max_intensity': (self.chbMaxIntensity, self.dsbMaxIntensity),
            'min_lifetime': (self.chbMinLifetime, self.dsbMinLifetime),
            'max_lifetime': (self.chbMaxLifetime, self.dsbMaxLifetime),
            'use_dw': (self.chbUseDW, self.cmbDWTest),
            'min_chi_squared': (self.chbMinChiSquared, self.dsbMinChiSquared),
            'max_chi_squared': (self.chbMaxChiSquared, self.dsbMaxChiSquared),
            'min_irf_shift': (self.chbMinIRFShift, self.dsbMinIRFShift),
            'max_irf_shift': (self.chbMaxIRFShift, self.dsbMaxIRFShift),
            'force_through_origin': (self.chbForceOrigin, None),
        }

        for option, check_box_and_value_object in self.option_linker.items():
            check_box, value_object = check_box_and_value_object
            if value_object is not None:
                check_box.stateChanged.connect((lambda opt: lambda: self.filter_option_changed(option=opt))(option))

        self.btnPhotonNumberDistribution.clicked \
            .connect(lambda: self.plot_features(feature_x=PlotFeature.PhotonNumber))
        self.btnIntensityDistribution.clicked \
            .connect(lambda: self.plot_features(feature_x=PlotFeature.Intensity))
        self.btnLifetimeDistribution.clicked \
            .connect(lambda: self.plot_features(feature_x=PlotFeature.Lifetime))
        self.btnDWDistribution.clicked \
            .connect(lambda: self.plot_features(feature_x=PlotFeature.DW))
        self.btnChiSquaredDistribution.clicked \
            .connect(lambda: self.plot_features(feature_x=PlotFeature.ChiSquared))
        self.btnIRFShiftDistribution.clicked \
            .connect(lambda: self.plot_features(feature_x=PlotFeature.IRFShift))

        filter_nums = [self.spnMinPhotons, self.dsbMinIntensity, self.dsbMaxIntensity, self.dsbMinLifetime,
                       self.dsbMaxLifetime, self.dsbMinChiSquared, self.dsbMaxChiSquared, self.dsbMinIRFShift,
                       self.dsbMaxIRFShift]
        for num_control in filter_nums:
            num_control.valueChanged.connect(lambda: self.plot_features(use_current_plot=True))
        self.cmbDWTest.currentTextChanged.connect(lambda: self.plot_features(use_current_plot=True))

        self.current_plot_type = (None, None)  # (PlotFeature.Intensity, PlotFeature.Lifetime)
        self.current_particles_to_use = None
        self.current_data_points_to_use = None
        self.current_particles = list()
        self.levels_to_use = None
        self.mapping_level_to_use = None

        self.setup_plot(*self.current_plot_type, clear_plot=False, is_first_setup=True)

        self.cmbFeatureX.currentTextChanged.connect(lambda: self.two_features_changed(x_changed=True))
        self.cmbFeatureY.currentTextChanged.connect(lambda: self.two_features_changed(x_changed=False))
        self.btnPlotTwoFeatures.clicked.connect(lambda: self.plot_features(use_selected_two_features=True))
        self.btnFit.clicked.connect(self.fit_intensity_lifetime)

        self.chbAutoNumBins.stateChanged.connect(self.auto_num_bins_changed)
        self.spnNumBins.valueChanged.connect(lambda: self.plot_features(use_current_plot=True))

        self.cmbParticlesToUse.currentTextChanged.connect(lambda: self.plot_features(use_current_plot=True))
        self.cmbUseResolvedOrGrouped.currentTextChanged.connect(lambda: self.plot_features(use_current_plot=True))
        self.chbUseROI.stateChanged.connect(lambda: self.plot_features(use_current_plot=True))

        self.chbApplyAllFilters.stateChanged.connect(lambda: self.plot_features(use_current_plot=True))

        self.btnApplyFilters.clicked.connect(self.apply_filters)
        self.btnApplyNormalization.clicked.connect(self.apply_normalization)
        self.btnResetFilters.clicked.connect(self.reset_filters)
        self.btnResetAllFilters.clicked.connect(self.reset_dataset_filter)
        self.btnResetNormalization.clicked.connect(self.reset_normalization)
        self.btnClose.clicked.connect(self.rejected_callback)

        self.plot_pen = QPen()
        self.plot_pen.setCosmetic(True)
        self.plot_pen.setWidthF(2)
        self.plot_pen.setColor(QColor('black'))

        self.distribution_item = pg.PlotCurveItem(
            x=[0, 0],
            y=[0],
            stepMode=True,
            pen=self.plot_pen,
            fillLevel=0,
            brush=QColor('lightGray')
        )

        self.two_feature_item = pg.ScatterPlotItem(
            x=[0],
            y=[0],
            pen=self.plot_pen,
            size=3
        )

    @staticmethod
    def _get_label(feature: PlotFeature = None) -> tuple:
        label = 'Count'
        unit = None
        if feature is not None:
            label = str(feature)
            if feature == PlotFeature.Intensity:
                unit = 'counts/s'
            elif feature == PlotFeature.Lifetime or feature == PlotFeature.IRFShift:
                unit = 'ns'
            else:
                unit = None
        return label, unit

    def setup_plot(self,
                   feature_x: PlotFeature,
                   feature_y: PlotFeature = None,
                   clear_plot: bool = True,
                   is_first_setup: bool = False):
        left_axis = self.plot.getAxis('left')
        bottom_axis = self.plot.getAxis('bottom')
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
        self.spnNumBins.setEnabled(not self.chbAutoNumBins.isChecked())
        self.plot_features(*self.current_plot_type)

    def get_two_features(self) -> tuple:
        return self.cmbFeatureX.currentText(), self.cmbFeatureY.currentText()

    def change_plot_type(self, feature_x: Union[PlotFeature, str] = None, feature_y: Union[PlotFeature, str] = None):
        if feature_x is None and feature_y is None:
            feature_x, feature_y = self.get_two_features()
        self.setup_plot(feature_x=feature_x, feature_y=feature_y)

    def set_levels_to_use(self):
        particles_changed = False
        particles_to_use = self.cmbParticlesToUse.currentText()
        if particles_to_use != self.current_particles_to_use:
            particles = list()
            if particles_to_use == 'Current':
                particles = [self.main_window.current_particle]
            elif particles_to_use == 'Selected':
                particles = self.main_window.get_checked_particles()
            elif particles_to_use == 'All':
                particles = self.main_window.current_dataset.particles
            self.current_particles = particles

            self.current_particles_to_use = particles_to_use
            particles_changed = True

        data_points_to_use = self.cmbUseResolvedOrGrouped.currentText()
        if data_points_to_use != self.current_data_points_to_use or particles_changed:
            particles = self.current_particles
            if data_points_to_use == 'Grouped':
                if not self.chbUseROI.isChecked():
                    levels = [particle.group_levels for particle in particles if particle.has_groups]
                else:
                    levels = [particle.group_levels_roi for particle in particles if particle.has_groups]
            else:
                if not self.chbUseROI.isChecked():
                    levels = [particle.cpts.levels for particle in particles if particle.has_levels]
                else:
                    levels = [particle.cpts.levels_roi_forced for particle in particles if particle.has_levels]
            self.levels_to_use = np.concatenate(levels) if len(levels) > 0 else list()
            self.current_data_points_to_use = data_points_to_use

    def get_data(self) -> tuple:
        self.set_levels_to_use()

        ints = list()
        taus = list()
        for level in self.levels_to_use:
            if level.histogram.fitted:
                ints.append(level.int_p_s)
                taus.append(level.histogram.tau)
        else:
            ints, taus = None, None

        return ints, taus

    def two_features_changed(self, x_changed: bool = None):
        if x_changed is None:
            raise ValueError('No argument provided.')
        feature_changed_cmb = self.cmbFeatureX if x_changed else self.cmbFeatureY
        other_feature_cmb = self.cmbFeatureY if x_changed else self.cmbFeatureX

        feature_changed_cmb.blockSignals(True)
        other_feature_cmb.blockSignals(True)

        new_other_features = PlotFeature.get_dict()
        feature_selected = new_other_features.pop(feature_changed_cmb.currentText())

        new_changed_features = PlotFeature.get_dict()
        other_selected_feature = new_changed_features.pop(other_feature_cmb.currentText())

        feature_changed_cmb.clear()
        feature_changed_cmb.addItems(new_changed_features.keys())
        feature_changed_cmb.setCurrentText(feature_selected)

        other_feature_cmb.clear()
        other_feature_cmb.addItems(new_other_features.keys())
        other_feature_cmb.setCurrentText(other_selected_feature)

        feature_changed_cmb.blockSignals(False)
        other_feature_cmb.blockSignals(False)

        current_feature_x, current_feature_y = self.current_plot_type
        if current_feature_x is not None and current_feature_y is not None:  # True if current is two feature plot
            self.plot_features(use_current_plot=True)

    def plot_features(self,
                      feature_x: Union[PlotFeature, str] = None,
                      feature_y: Union[PlotFeature, str] = None,
                      use_current_plot: bool = False,
                      use_selected_two_features: bool = False):
        if use_current_plot:
            if use_selected_two_features:
                raise ValueError("Conflicting options")
            if feature_x is not None or feature_y is not None:
                raise ValueError("Use current plot option excludes provided features")
            feature_x, feature_y = self.current_plot_type
            use_selected_two_features = feature_x is not None and feature_y is not None

        if not use_current_plot or use_selected_two_features:
            if use_selected_two_features:
                if not use_current_plot and (feature_x is not None or feature_y is not None):
                    raise ValueError('Use selected two features excludes provided features')
                else:
                    feature_x = self.cmbFeatureX.currentText()
                    feature_y = self.cmbFeatureY.currentText()
            else:
                if feature_x is None and feature_y is not None:
                    raise ValueError('Can not provide only a plot feature for the Y-Axis')
                if feature_x is None and feature_y is None:
                    logger.warning('No feature(s) provided and no options selected')
                    return None

        if (feature_x, feature_y) != (None, None):
            if self.current_plot_type != (feature_x, feature_y):
                self.change_plot_type(feature_x=feature_x, feature_y=feature_y)

            self.set_levels_to_use()

            is_distribution = True if feature_y is None else False
            plot_item = self.distribution_item if is_distribution else self.two_feature_item
            if plot_item not in self.plot.items:
                for item in self.plot.items:
                    self.plot.removeItem(item)
                self.plot.addItem(plot_item)
            if is_distribution:
                self._plot_distribution()
            else:
                self._plot_two_features()

    def _filter_numeric_data(self,
                             feature_data: np.ndarray,
                             test_min: bool = False,
                             test_max: bool = False,
                             min_value=None,
                             max_value=None):
        levels = self.levels_to_use
        num_datapoints_filtered = None

        num_datapoints = len(feature_data)
        if test_min:
            feature_data = np.array([value if not np.isnan(value) and value >= min_value else np.NaN
                                     for value in feature_data])
        if test_max:
            feature_data = np.array([value if not np.isnan(value) and value <= max_value else np.NaN
                                     for value in feature_data])
        if test_min or test_max:
            num_datapoints_filtered = np.sum(~np.isnan(feature_data))

        return feature_data, num_datapoints_filtered

    def get_feature_data(self, feature: Union[PlotFeature, str]) -> tuple:
        if self.levels_to_use is None:
            self.set_levels_to_use()
        levels = self.levels_to_use
        histograms = [level.histogram for level in levels]
        feature_data = None
        num_datapoints = 0
        num_datapoints_filtered = None

        if feature == PlotFeature.PhotonNumber:
            feature_data = np.array([level.num_photons if level.num_photons is not None else np.NaN
                                     for level in levels])
            num_datapoints = np.sum(~np.isnan(feature_data))
            feature_data, num_datapoints_filtered = self._filter_numeric_data(
                feature_data=feature_data,
                test_min=self.chbMinPhotons.isChecked(),
                min_value=self.spnMinPhotons.value()
            )

        elif feature == PlotFeature.Intensity:
            feature_data = np.array([level.int_p_s if level.int_p_s is not None else np.NaN for level in levels])
            num_datapoints = np.sum(~np.isnan(feature_data))
            feature_data, num_datapoints_filtered, = self._filter_numeric_data(
                feature_data=feature_data,
                test_min=self.chbMinIntensity.isChecked(),
                test_max=self.chbMaxIntensity.isChecked(),
                min_value=self.dsbMinIntensity.value(),
                max_value=self.dsbMaxIntensity.value(),
            )

        elif feature == PlotFeature.Lifetime:
            feature_data = [histogram.tau if histogram.fitted and histogram.tau is not None else np.NaN
                            for histogram in histograms]
            feature_data = np.array([value[0] if type(value) is list and len(value) == 1 else value
                                     for value in feature_data])
            num_datapoints = np.sum(~np.isnan(feature_data))
            feature_data, num_datapoints_filtered, = self._filter_numeric_data(
                feature_data=feature_data,
                test_min=self.chbMinLifetime.isChecked(),
                test_max=self.chbMaxLifetime.isChecked(),
                min_value=self.dsbMinLifetime.value(),
                max_value=self.dsbMaxLifetime.value(),
            )

        elif feature == PlotFeature.DW:
            levels_used = np.array([level.histogram.fitted and level.histogram.dw is not None for level in levels])
            feature_data = np.array([level.histogram.dw if level.histogram.dw is not None else np.NaN
                                     for level in levels])
            num_datapoints = np.sum(~np.isnan(feature_data))
            if self.chbUseDW.isChecked():
                selected_dw_test = self.cmbDWTest.currentText()
                dw_ind = None
                if selected_dw_test == '5%':
                    dw_ind = 0
                elif selected_dw_test == '1%':
                    dw_ind = 1
                elif selected_dw_test == '0.3%':
                    dw_ind = 2
                elif selected_dw_test == '0.1%':
                    dw_ind = 3
                feature_data = np.array([value if not np.isnan(value) and histogram.dw >= histogram.dw_bound[dw_ind]
                                         else np.NaN for (value, histogram) in zip(feature_data, histograms)])
                num_datapoints_filtered = np.sum(~np.isnan(feature_data))

        elif feature == PlotFeature.IRFShift:
            feature_data = np.array([histogram.shift if histogram.fitted and histogram.shift is not None else np.NaN
                                     for histogram in histograms])
            num_datapoints = np.sum(~np.isnan(feature_data))
            feature_data, num_datapoints_filtered, = self._filter_numeric_data(
                feature_data=feature_data,
                test_min=self.chbMinIRFShift.isChecked(),
                test_max=self.chbMaxIRFShift.isChecked(),
                min_value=self.dsbMinIRFShift.value(),
                max_value=self.dsbMaxIRFShift.value(),
            )

        elif feature == PlotFeature.ChiSquared:
            feature_data = np.array([histogram.chisq if histogram.fitted and histogram.chisq is not None else np.NaN
                                     for histogram in histograms])
            num_datapoints = np.sum(~np.isnan(feature_data))
            feature_data, num_datapoints_filtered, = self._filter_numeric_data(
                feature_data=feature_data,
                test_min=self.chbMinChiSquared.isChecked(),
                test_max=self.chbMaxChiSquared.isChecked(),
                min_value=self.dsbMinChiSquared.value(),
                max_value=self.dsbMaxChiSquared.value(),
            )

        return feature_data, num_datapoints, num_datapoints_filtered

    def get_all_feature_data(self) -> dict:
        all_features = PlotFeature.get_dict()
        all_feature_data = dict()
        for key, item in all_features.items():
            feature_data, num_datapoints, num_datapoints_filtered = self.get_feature_data(feature=item)
            all_feature_data[key] = {
                'feature_data': feature_data,
                'num_datapoints': num_datapoints,
                'num_datapoints_filtered': num_datapoints_filtered
            }
        return all_feature_data

    def get_all_filter(self) -> np.ndarray:
        all_feature_data = self.get_all_feature_data()
        all_filter = None
        for _, feature_all_data in all_feature_data.items():
            if all_filter is None:
                all_filter = ~np.isnan(feature_all_data['feature_data'])
            else:
                all_filter &= ~np.isnan(feature_all_data['feature_data'])
        return all_filter

    def _plot_distribution(self):
        feature, _ = self.current_plot_type

        feature_data, num_datapoints, num_datapoints_filtered = self.get_feature_data(feature=feature)
        feature_data = feature_data[~np.isnan(feature_data)]

        if feature_data is not None:
            is_auto_num_bins = self.chbAutoNumBins.isChecked()
            if is_auto_num_bins:
                bin_edges = 'auto'
            else:
                num_bins = self.spnNumBins.value()
                bin_edges = np.histogram_bin_edges(feature_data, num_bins)

            bin_edges, hist_data = np.histogram(feature_data, bins=bin_edges, density=False)

            if is_auto_num_bins:
                self.spnNumBins.blockSignals(True)
                self.spnNumBins.setValue(len(bin_edges))
                self.spnNumBins.blockSignals(False)

            self.distribution_item.setData(x=hist_data, y=bin_edges)

            if num_datapoints_filtered is None:
                num_datapoints_text = f'# Datapoints: {num_datapoints}'
            else:
                num_datapoints_text = f'# Datapoints: {num_datapoints_filtered} ({num_datapoints} unfiltered)'
            self.lblNumDatapoints.setText(num_datapoints_text)
        else:
            logger.warning('No feature data found')

    def _plot_two_features(self):
        feature_x, feature_y = self.current_plot_type

        if feature_y == PlotFeature.IRFShift:
            self.plot.vb.setLimits(yMin=None)
        else:
            self.plot.vb.setLimits(yMin=0)

        featured_x_data, num_data_x, num_data_x_filt = self.get_feature_data(feature=feature_x)
        featured_y_data, num_data_y, num_data_y_filt = self.get_feature_data(feature=feature_y)

        not_nan_values = (~np.isnan(featured_x_data)) & (~np.isnan(featured_y_data))
        did_all_filter = False
        if self.chbApplyAllFilters.isChecked():
            all_filter = self.get_all_filter()
            did_all_filter = np.sum(all_filter) < np.sum(not_nan_values)
            not_nan_values &= all_filter

        featured_x_data = featured_x_data[not_nan_values]
        featured_y_data = featured_y_data[not_nan_values]

        self.two_feature_item.setData(x=featured_x_data, y=featured_y_data)

        num_data = np.min([num_data_y, num_data_y])
        if not did_all_filter and (num_data_x_filt is None and num_data_y_filt is None):
            num_datapoints_text = f'# Datapoints: {num_data}'
        else:
            num_data_filtered = np.sum(not_nan_values)
            num_datapoints_text = f'# Datapoints: {num_data_filtered} ({num_data} unfiltered)'
        self.lblNumDatapoints.setText(num_datapoints_text)

    def fit_intensity_lifetime(self):
        self.plot_features(feature_x=PlotFeature.Intensity, feature_y=PlotFeature.Lifetime)
        int_data, lifetime_data = self.two_feature_item.getData()

    def apply_filters(self):
        all_filters = self.get_all_filter()
        for level, level_filter in zip(self.levels_to_use, all_filters):
            level.is_filtered_out = not level_filter
        self.lblResults.setText('Filters Applied')

    def reset_filters(self):
        for level in self.levels_to_use:
            level.is_filtered_out = False
        self.lblResults.setText('Filters Reset')

    def reset_dataset_filter(self):
        for particle in self.main_window.current_dataset.particles:
            if particle.has_levels:
                for level in particle.cpts.levels:
                    level.is_filtered_out = False
                if particle.has_groups:
                    for step in particle.ahca.steps:
                        for group_level in step.group_levels:
                            group_level.is_filtered_out = False
        self.lblResults.setText('All Filters Reset')

    def reset_normalization(self):
        print('reset_normalization')

    def apply_normalization(self):
        print('apply_normalization')  # TODO: Return and apply Normalization

    def rejected_callback(self):
        self.close()
