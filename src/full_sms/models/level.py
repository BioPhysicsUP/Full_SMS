"""Data models for intensity levels from change point analysis."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LevelData:
    """A single intensity level detected by change point analysis.

    A level represents a period of constant intensity within a particle's
    photon trace. Levels are the output of change point analysis (CPA).

    Attributes:
        start_index: First photon index (inclusive) in this level.
        end_index: Last photon index (inclusive) in this level.
        start_time_ns: Absolute time of the first photon in nanoseconds.
        end_time_ns: Absolute time of the last photon in nanoseconds.
        num_photons: Number of photons in this level.
        intensity_cps: Intensity in counts per second.
        group_id: Optional group assignment from clustering.
    """

    start_index: int
    end_index: int
    start_time_ns: int
    end_time_ns: int
    num_photons: int
    intensity_cps: float
    group_id: Optional[int] = None

    @property
    def indices(self) -> Tuple[int, int]:
        """Photon index range as (start, end) tuple."""
        return (self.start_index, self.end_index)

    @property
    def times_ns(self) -> Tuple[int, int]:
        """Time range in nanoseconds as (start, end) tuple."""
        return (self.start_time_ns, self.end_time_ns)

    @property
    def dwell_time_ns(self) -> int:
        """Duration of this level in nanoseconds."""
        return self.end_time_ns - self.start_time_ns

    @property
    def dwell_time_s(self) -> float:
        """Duration of this level in seconds."""
        return self.dwell_time_ns / 1e9

    @property
    def times_s(self) -> Tuple[float, float]:
        """Time range in seconds as (start, end) tuple."""
        return (self.start_time_ns / 1e9, self.end_time_ns / 1e9)

    def __post_init__(self) -> None:
        """Validate level data."""
        if self.start_index < 0:
            raise ValueError(f"start_index must be non-negative, got {self.start_index}")
        if self.end_index < self.start_index:
            raise ValueError(
                f"end_index ({self.end_index}) must be >= start_index ({self.start_index})"
            )
        if self.start_time_ns < 0:
            raise ValueError(
                f"start_time_ns must be non-negative, got {self.start_time_ns}"
            )
        if self.end_time_ns < self.start_time_ns:
            raise ValueError(
                f"end_time_ns ({self.end_time_ns}) must be >= start_time_ns ({self.start_time_ns})"
            )
        if self.num_photons < 0:
            raise ValueError(f"num_photons must be non-negative, got {self.num_photons}")
        if self.intensity_cps < 0:
            raise ValueError(f"intensity_cps must be non-negative, got {self.intensity_cps}")

    @classmethod
    def from_photon_indices(
        cls,
        abstimes: NDArray[np.uint64],
        start_index: int,
        end_index: int,
        group_id: Optional[int] = None,
    ) -> "LevelData":
        """Create a LevelData from photon absolute times and index range.

        Args:
            abstimes: Array of absolute photon arrival times in nanoseconds.
            start_index: First photon index (inclusive).
            end_index: Last photon index (inclusive).
            group_id: Optional group assignment.

        Returns:
            A new LevelData instance with computed properties.
        """
        start_time_ns = int(abstimes[start_index])
        end_time_ns = int(abstimes[end_index])
        num_photons = end_index - start_index + 1
        dwell_time_s = (end_time_ns - start_time_ns) / 1e9

        if dwell_time_s > 0:
            intensity_cps = num_photons / dwell_time_s
        else:
            intensity_cps = 0.0

        return cls(
            start_index=start_index,
            end_index=end_index,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            num_photons=num_photons,
            intensity_cps=intensity_cps,
            group_id=group_id,
        )
