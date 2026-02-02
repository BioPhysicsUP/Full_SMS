"""Data models for measurements and channels."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SpectraData:
    """Spectral data recorded as a time series of spectra.

    Spectra are recorded with a grating and CCD, with a certain integration time
    over which each individual spectrum is recorded.

    Attributes:
        data: 2D array of spectral intensities (num_spectra × num_wavelengths).
        wavelengths: Array of wavelength values in nanometers.
        series_times: Absolute times (in seconds) when each spectrum was recorded.
        exposure_time: Integration time per spectrum in seconds.
    """

    data: NDArray[np.float64]
    wavelengths: NDArray[np.float64]
    series_times: NDArray[np.float64]
    exposure_time: float

    @property
    def num_spectra(self) -> int:
        """Number of spectra in the time series."""
        return self.data.shape[0]

    @property
    def num_wavelengths(self) -> int:
        """Number of wavelength channels."""
        return len(self.wavelengths)

    @property
    def wavelength_min(self) -> float:
        """Minimum wavelength in nanometers."""
        return float(np.min(self.wavelengths))

    @property
    def wavelength_max(self) -> float:
        """Maximum wavelength in nanometers."""
        return float(np.max(self.wavelengths))

    @property
    def time_min(self) -> float:
        """Minimum time in seconds."""
        return float(np.min(self.series_times)) if len(self.series_times) > 0 else 0.0

    @property
    def time_max(self) -> float:
        """Maximum time in seconds."""
        return float(np.max(self.series_times)) if len(self.series_times) > 0 else 0.0

    def get_spectrum_at_index(self, idx: int) -> NDArray[np.float64]:
        """Get a single spectrum at a specific time index.

        Args:
            idx: Time index of the spectrum.

        Returns:
            Array of intensity values at each wavelength.
        """
        return self.data[idx, :]

    def get_averaged_spectrum(
        self, start_idx: int = 0, end_idx: int | None = None
    ) -> NDArray[np.float64]:
        """Get an averaged spectrum over a time range.

        Args:
            start_idx: Starting time index (inclusive).
            end_idx: Ending time index (exclusive). If None, use all remaining.

        Returns:
            Averaged intensity values at each wavelength.
        """
        if end_idx is None:
            end_idx = self.num_spectra
        return np.mean(self.data[start_idx:end_idx, :], axis=0)


@dataclass(frozen=True)
class RasterScanData:
    """Raster scan image data.

    A raster scan is a 2D intensity scan used to visualize measurements
    before measurement. The scan produces a 2D image where each pixel
    represents photon counts at that position.

    Attributes:
        data: 2D array of intensity values (pixels_y × pixels_x).
        x_start: Starting X position in micrometers.
        y_start: Starting Y position in micrometers.
        scan_range: Scan range in micrometers (assumes square scan).
        pixels_per_line: Number of pixels per line.
        integration_time: Integration time per pixel in ms/um.
    """

    data: NDArray[np.float64]
    x_start: float
    y_start: float
    scan_range: float
    pixels_per_line: int
    integration_time: float

    @property
    def num_pixels_x(self) -> int:
        """Number of pixels in X direction."""
        return self.data.shape[1] if self.data.ndim == 2 else 0

    @property
    def num_pixels_y(self) -> int:
        """Number of pixels in Y direction."""
        return self.data.shape[0] if self.data.ndim == 2 else 0

    @property
    def x_min(self) -> float:
        """Minimum X position in micrometers."""
        return self.x_start

    @property
    def x_max(self) -> float:
        """Maximum X position in micrometers."""
        return self.x_start + self.scan_range

    @property
    def y_min(self) -> float:
        """Minimum Y position in micrometers."""
        return self.y_start

    @property
    def y_max(self) -> float:
        """Maximum Y position in micrometers."""
        return self.y_start + self.scan_range

    @property
    def pixel_size(self) -> float:
        """Size of each pixel in micrometers."""
        if self.pixels_per_line == 0:
            return 0.0
        return self.scan_range / self.pixels_per_line

    @property
    def intensity_min(self) -> float:
        """Minimum intensity value."""
        return float(np.min(self.data)) if self.data.size > 0 else 0.0

    @property
    def intensity_max(self) -> float:
        """Maximum intensity value."""
        return float(np.max(self.data)) if self.data.size > 0 else 0.0


@dataclass(frozen=True)
class ChannelData:
    """Photon timing data from a single TCSPC channel.

    Attributes:
        abstimes: Absolute photon arrival times in nanoseconds.
        microtimes: TCSPC micro times (time within excitation cycle) in nanoseconds.
    """

    abstimes: NDArray[np.uint64]
    microtimes: NDArray[np.float64]

    @property
    def num_photons(self) -> int:
        """Number of photons in this channel."""
        return len(self.abstimes)

    @property
    def measurement_time_s(self) -> float:
        """Total measurement time in seconds."""
        if self.num_photons == 0:
            return 0.0
        return (self.abstimes[-1] - self.abstimes[0]) / 1e9

    def __post_init__(self) -> None:
        """Validate that arrays have the same length."""
        if len(self.abstimes) != len(self.microtimes):
            raise ValueError(
                f"abstimes and microtimes must have same length, "
                f"got {len(self.abstimes)} and {len(self.microtimes)}"
            )


@dataclass
class MeasurementData:
    """Data for a single molecule measurement.

    A measurement represents a single molecule measurement from an HDF5 file.
    It may have one or two TCSPC channels (for single or dual detector setups).

    Attributes:
        id: Unique identifier (typically the measurement number from the file).
        name: Display name (e.g., "Measurement 1").
        description: Optional description from the HDF5 file metadata.
        tcspc_card: TCSPC card identifier.
        channelwidth: TCSPC histogram channel width in nanoseconds.
        channel1: Primary TCSPC channel data.
        channel2: Optional secondary TCSPC channel data (for dual detector setups).
        spectra: Optional spectral data if recorded for this measurement.
        raster_scan: Optional raster scan image data.
        raster_scan_coord: Optional (x, y) coordinate in um where the measurement was taken.
    """

    id: int
    name: str
    tcspc_card: str
    channelwidth: float
    channel1: ChannelData
    channel2: Optional[ChannelData] = None
    description: str = ""
    spectra: Optional[SpectraData] = None
    raster_scan: Optional[RasterScanData] = None
    raster_scan_coord: Optional[tuple[float, float]] = None

    @property
    def num_photons(self) -> int:
        """Total number of photons across all channels."""
        total = self.channel1.num_photons
        if self.channel2 is not None:
            total += self.channel2.num_photons
        return total

    @property
    def measurement_time_s(self) -> float:
        """Total measurement time in seconds (from primary channel)."""
        return self.channel1.measurement_time_s

    @property
    def has_dual_channel(self) -> bool:
        """Whether this measurement has dual TCSPC channels."""
        return self.channel2 is not None

    @property
    def abstimes(self) -> NDArray[np.uint64]:
        """Absolute times from primary channel (convenience accessor)."""
        return self.channel1.abstimes

    @property
    def microtimes(self) -> NDArray[np.float64]:
        """Micro times from primary channel (convenience accessor)."""
        return self.channel1.microtimes

    @property
    def has_spectra(self) -> bool:
        """Whether this measurement has spectral data."""
        return self.spectra is not None

    @property
    def has_raster_scan(self) -> bool:
        """Whether this measurement has raster scan data."""
        return self.raster_scan is not None
