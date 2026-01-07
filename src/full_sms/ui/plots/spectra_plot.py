"""Spectra heatmap plot widget.

Renders spectral time series data as a 2D heatmap using DearPyGui's ImPlot,
with time on the X axis, wavelength on the Y axis, and intensity as color.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import numpy as np
from numpy.typing import NDArray

from full_sms.models.particle import SpectraData

logger = logging.getLogger(__name__)


@dataclass
class SpectraPlotTags:
    """Tags for spectra plot elements."""

    container: str = "spectra_plot_container"
    plot: str = "spectra_plot"
    x_axis: str = "spectra_plot_x_axis"
    y_axis: str = "spectra_plot_y_axis"
    heat_series: str = "spectra_plot_heat_series"
    colormap_scale: str = "spectra_plot_colormap_scale"


SPECTRA_PLOT_TAGS = SpectraPlotTags()


class SpectraPlot:
    """Spectra heatmap plot widget.

    Displays spectral time series data as a 2D heatmap with:
    - X axis: Time (seconds)
    - Y axis: Wavelength (nm)
    - Color: Intensity (counts/s)
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the spectra plot.

        Args:
            parent: The parent container to build the plot in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._spectra_data: SpectraData | None = None

        # Display state
        self._colormap: int = dpg.mvPlotColormap_Plasma

        # Generate unique tags
        self._tags = SpectraPlotTags(
            container=f"{tag_prefix}spectra_plot_container",
            plot=f"{tag_prefix}spectra_plot",
            x_axis=f"{tag_prefix}spectra_plot_x_axis",
            y_axis=f"{tag_prefix}spectra_plot_y_axis",
            heat_series=f"{tag_prefix}spectra_plot_heat_series",
            colormap_scale=f"{tag_prefix}spectra_plot_colormap_scale",
        )

    @property
    def tags(self) -> SpectraPlotTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._spectra_data is not None

    def build(self) -> None:
        """Build the plot UI structure."""
        if self._is_built:
            return

        # Plot container
        with dpg.group(parent=self._parent, tag=self._tags.container, horizontal=True):
            # Create the plot
            with dpg.plot(
                tag=self._tags.plot,
                label="Spectral Trace",
                width=-100,  # Leave room for colormap scale
                height=-1,
                anti_aliased=True,
            ):
                # X axis (time in seconds)
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="Time (s)",
                    tag=self._tags.x_axis,
                )

                # Y axis (wavelength in nm)
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label="Wavelength (nm)",
                    tag=self._tags.y_axis,
                )

                # Add empty heat series (will be populated when data is set)
                dpg.add_heat_series(
                    [],
                    rows=1,
                    cols=1,
                    scale_min=0,
                    scale_max=1,
                    parent=self._tags.y_axis,
                    tag=self._tags.heat_series,
                    format="",  # Don't show values on hover
                )

            # Add colormap scale
            dpg.add_colormap_scale(
                tag=self._tags.colormap_scale,
                colormap=self._colormap,
                min_scale=0,
                max_scale=1,
                width=80,
                height=-1,
                label="Intensity",
            )

        # Bind colormap to the plot
        dpg.bind_colormap(self._tags.plot, self._colormap)

        self._is_built = True
        logger.debug("Spectra plot built")

    def set_data(self, spectra: SpectraData) -> None:
        """Set the spectra data and update the plot.

        Args:
            spectra: The SpectraData object containing the spectral time series.
        """
        self._spectra_data = spectra

        if spectra.data.size == 0:
            self.clear()
            return

        # The spectra data is (num_spectra × num_wavelengths)
        # For heat_series, we need to flatten in row-major order
        # and specify rows (wavelengths) and cols (time points)
        data = spectra.data

        # Transpose so we have wavelength on Y and time on X
        # Original: (num_spectra, num_wavelengths) -> (num_wavelengths, num_spectra)
        data_transposed = data.T

        rows = data_transposed.shape[0]  # wavelengths
        cols = data_transposed.shape[1]  # time points

        # Get bounds for the heat series
        t_min = spectra.time_min
        t_max = spectra.time_max
        wl_min = spectra.wavelength_min
        wl_max = spectra.wavelength_max

        # Calculate scale bounds
        scale_min = float(np.min(data))
        scale_max = float(np.max(data))

        # Handle case where all values are the same
        if scale_max <= scale_min:
            scale_max = scale_min + 1.0

        # Flatten the data in row-major order for heat_series
        flat_data = data_transposed.flatten().tolist()

        # Update the heat series
        if dpg.does_item_exist(self._tags.heat_series):
            dpg.configure_item(
                self._tags.heat_series,
                x=flat_data,
                rows=rows,
                cols=cols,
                bounds_min=(t_min, wl_min),
                bounds_max=(t_max, wl_max),
                scale_min=scale_min,
                scale_max=scale_max,
            )

        # Update colormap scale
        if dpg.does_item_exist(self._tags.colormap_scale):
            dpg.configure_item(
                self._tags.colormap_scale,
                min_scale=scale_min,
                max_scale=scale_max,
            )

        # Fit axes to data
        self._fit_axes()

        logger.debug(
            f"Spectra plot updated: {spectra.num_spectra} spectra × "
            f"{spectra.num_wavelengths} wavelengths"
        )

    def _fit_axes(self) -> None:
        """Auto-fit the axes to show all data."""
        if dpg.does_item_exist(self._tags.x_axis):
            dpg.fit_axis_data(self._tags.x_axis)
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.fit_axis_data(self._tags.y_axis)

    def clear(self) -> None:
        """Clear the plot data."""
        self._spectra_data = None

        if dpg.does_item_exist(self._tags.heat_series):
            dpg.configure_item(
                self._tags.heat_series,
                x=[],
                rows=1,
                cols=1,
                scale_min=0,
                scale_max=1,
            )

        if dpg.does_item_exist(self._tags.colormap_scale):
            dpg.configure_item(
                self._tags.colormap_scale,
                min_scale=0,
                max_scale=1,
            )

        logger.debug("Spectra plot cleared")

    def fit_view(self) -> None:
        """Fit the view to show all data (reset zoom/pan)."""
        self._fit_axes()

    def set_colormap(self, colormap: int) -> None:
        """Set the colormap for the heatmap.

        Args:
            colormap: DearPyGui colormap constant (e.g., dpg.mvPlotColormap_Plasma).
        """
        self._colormap = colormap

        if dpg.does_item_exist(self._tags.plot):
            dpg.bind_colormap(self._tags.plot, colormap)

        if dpg.does_item_exist(self._tags.colormap_scale):
            dpg.configure_item(self._tags.colormap_scale, colormap=colormap)

        logger.debug(f"Spectra plot colormap changed")

    def get_time_range(self) -> tuple[float, float] | None:
        """Get the current time range of the data.

        Returns:
            Tuple of (min_time, max_time) in seconds, or None if no data.
        """
        if self._spectra_data is None:
            return None
        return (self._spectra_data.time_min, self._spectra_data.time_max)

    def get_wavelength_range(self) -> tuple[float, float] | None:
        """Get the current wavelength range of the data.

        Returns:
            Tuple of (min_wavelength, max_wavelength) in nm, or None if no data.
        """
        if self._spectra_data is None:
            return None
        return (self._spectra_data.wavelength_min, self._spectra_data.wavelength_max)

    def get_intensity_range(self) -> tuple[float, float] | None:
        """Get the current intensity range of the data.

        Returns:
            Tuple of (min_intensity, max_intensity), or None if no data.
        """
        if self._spectra_data is None:
            return None
        data = self._spectra_data.data
        return (float(np.min(data)), float(np.max(data)))
