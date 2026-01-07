"""Data models for particles and channels."""

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
class ParticleData:
    """Data for a single particle measurement.

    A particle represents a single molecule measurement from an HDF5 file.
    It may have one or two TCSPC channels (for single or dual detector setups).

    Attributes:
        id: Unique identifier (typically the particle number from the file).
        name: Display name (e.g., "Particle 1").
        description: Optional description from the HDF5 file metadata.
        tcspc_card: TCSPC card identifier.
        channelwidth: TCSPC histogram channel width in nanoseconds.
        channel1: Primary TCSPC channel data.
        channel2: Optional secondary TCSPC channel data (for dual detector setups).
        spectra: Optional spectral data if recorded for this particle.
    """

    id: int
    name: str
    tcspc_card: str
    channelwidth: float
    channel1: ChannelData
    channel2: Optional[ChannelData] = None
    description: str = ""
    spectra: Optional[SpectraData] = None

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
        """Whether this particle has dual TCSPC channels."""
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
        """Whether this particle has spectral data."""
        return self.spectra is not None
