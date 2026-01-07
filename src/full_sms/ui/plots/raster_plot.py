"""Raster scan heatmap plot widget.

Renders raster scan image data as a 2D heatmap using DearPyGui's ImPlot,
with X position on the X axis, Y position on the Y axis, and intensity as color.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import numpy as np

from full_sms.models.particle import RasterScanData

logger = logging.getLogger(__name__)


@dataclass
class RasterPlotTags:
    """Tags for raster plot elements."""

    container: str = "raster_plot_container"
    plot: str = "raster_plot"
    x_axis: str = "raster_plot_x_axis"
    y_axis: str = "raster_plot_y_axis"
    heat_series: str = "raster_plot_heat_series"
    colormap_scale: str = "raster_plot_colormap_scale"


RASTER_PLOT_TAGS = RasterPlotTags()


class RasterPlot:
    """Raster scan heatmap plot widget.

    Displays raster scan image data as a 2D heatmap with:
    - X axis: X Position (um)
    - Y axis: Y Position (um)
    - Color: Intensity (counts)
    """

    def __init__(
        self,
        parent: int | str,
        tag_prefix: str = "",
    ) -> None:
        """Initialize the raster plot.

        Args:
            parent: The parent container to build the plot in.
            tag_prefix: Optional prefix for tags to allow multiple instances.
        """
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._is_built = False

        # Data state
        self._raster_data: RasterScanData | None = None

        # Display state
        self._colormap: int = dpg.mvPlotColormap_Plasma

        # Generate unique tags
        self._tags = RasterPlotTags(
            container=f"{tag_prefix}raster_plot_container",
            plot=f"{tag_prefix}raster_plot",
            x_axis=f"{tag_prefix}raster_plot_x_axis",
            y_axis=f"{tag_prefix}raster_plot_y_axis",
            heat_series=f"{tag_prefix}raster_plot_heat_series",
            colormap_scale=f"{tag_prefix}raster_plot_colormap_scale",
        )

    @property
    def tags(self) -> RasterPlotTags:
        """Get the tags for this plot instance."""
        return self._tags

    @property
    def has_data(self) -> bool:
        """Whether the plot has data loaded."""
        return self._raster_data is not None

    def build(self) -> None:
        """Build the plot UI structure."""
        if self._is_built:
            return

        # Plot container
        with dpg.group(parent=self._parent, tag=self._tags.container, horizontal=True):
            # Create the plot
            with dpg.plot(
                tag=self._tags.plot,
                label="Raster Scan",
                width=-100,  # Leave room for colormap scale
                height=-1,
                anti_aliased=True,
                equal_aspects=True,  # Keep aspect ratio square for spatial data
            ):
                # X axis (position in micrometers)
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="X Position (um)",
                    tag=self._tags.x_axis,
                )

                # Y axis (position in micrometers)
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label="Y Position (um)",
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
        logger.debug("Raster plot built")

    def set_data(self, raster: RasterScanData) -> None:
        """Set the raster scan data and update the plot.

        Args:
            raster: The RasterScanData object containing the 2D scan image.
        """
        self._raster_data = raster

        if raster.data.size == 0:
            self.clear()
            return

        # The raster data is (rows × cols) = (num_pixels_y × num_pixels_x)
        # For heat_series, we need rows and cols for the heat series
        data = raster.data

        rows = data.shape[0]  # Y pixels
        cols = data.shape[1]  # X pixels

        # Get bounds for the heat series (spatial coordinates in um)
        x_min = raster.x_min
        x_max = raster.x_max
        y_min = raster.y_min
        y_max = raster.y_max

        # Calculate scale bounds for colormap
        scale_min = raster.intensity_min
        scale_max = raster.intensity_max

        # Handle case where all values are the same
        if scale_max <= scale_min:
            scale_max = scale_min + 1.0

        # Flatten the data in row-major order for heat_series
        flat_data = data.flatten().tolist()

        # Update the heat series
        if dpg.does_item_exist(self._tags.heat_series):
            dpg.configure_item(
                self._tags.heat_series,
                x=flat_data,
                rows=rows,
                cols=cols,
                bounds_min=(x_min, y_min),
                bounds_max=(x_max, y_max),
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
            f"Raster plot updated: {raster.num_pixels_y}x{raster.num_pixels_x} pixels, "
            f"range {raster.scan_range:.1f} um"
        )

    def _fit_axes(self) -> None:
        """Auto-fit the axes to show all data."""
        if dpg.does_item_exist(self._tags.x_axis):
            dpg.fit_axis_data(self._tags.x_axis)
        if dpg.does_item_exist(self._tags.y_axis):
            dpg.fit_axis_data(self._tags.y_axis)

    def clear(self) -> None:
        """Clear the plot data."""
        self._raster_data = None

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

        logger.debug("Raster plot cleared")

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

        logger.debug("Raster plot colormap changed")

    def get_position_range(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Get the current position range of the data.

        Returns:
            Tuple of ((x_min, x_max), (y_min, y_max)) in um, or None if no data.
        """
        if self._raster_data is None:
            return None
        return (
            (self._raster_data.x_min, self._raster_data.x_max),
            (self._raster_data.y_min, self._raster_data.y_max),
        )

    def get_intensity_range(self) -> tuple[float, float] | None:
        """Get the current intensity range of the data.

        Returns:
            Tuple of (min_intensity, max_intensity), or None if no data.
        """
        if self._raster_data is None:
            return None
        return (self._raster_data.intensity_min, self._raster_data.intensity_max)
