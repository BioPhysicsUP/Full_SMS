"""Data models for particles and channels."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


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
    """

    id: int
    name: str
    tcspc_card: str
    channelwidth: float
    channel1: ChannelData
    channel2: Optional[ChannelData] = None
    description: str = ""

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
