"""Application state container for Full SMS."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from full_sms.models.fit import FitResult, FitResultData, IRFData
from full_sms.models.group import ClusteringResult
from full_sms.models.level import LevelData
from full_sms.models.particle import ParticleData


class ConfidenceLevel(Enum):
    """Confidence levels for change point analysis."""

    CONF_69 = 0.69
    CONF_90 = 0.90
    CONF_95 = 0.95
    CONF_99 = 0.99


class ActiveTab(Enum):
    """Available tabs in the application."""

    INTENSITY = "intensity"
    LIFETIME = "lifetime"
    GROUPING = "grouping"
    SPECTRA = "spectra"
    RASTER = "raster"
    CORRELATION = "correlation"
    EXPORT = "export"


@dataclass
class FileMetadata:
    """Metadata about the loaded HDF5 file.

    Attributes:
        path: Path to the HDF5 file.
        filename: Name of the file (without path).
        num_particles: Number of particles in the file.
        has_irf: Whether the file contains an IRF.
        has_spectra: Whether the file contains spectra data.
        has_raster: Whether the file contains raster scan data.
    """

    path: Path
    filename: str
    num_particles: int
    has_irf: bool = False
    has_spectra: bool = False
    has_raster: bool = False
    analysis_uuid: Optional[str] = None


@dataclass
class ChannelSelection:
    """Selection of a specific particle and channel.

    Attributes:
        particle_id: ID of the selected particle.
        channel: Channel number (1 or 2).
    """

    particle_id: int
    channel: int = 1

    def __post_init__(self) -> None:
        """Validate channel selection."""
        if self.channel not in (1, 2):
            raise ValueError(f"channel must be 1 or 2, got {self.channel}")


@dataclass
class UIState:
    """UI-related state.

    Attributes:
        bin_size_ms: Bin size for intensity histogram in milliseconds.
        confidence: Confidence level for change point analysis.
        active_tab: Currently active tab.
        show_levels: Whether to show levels on intensity plot.
        show_groups: Whether to show group colors on intensity plot.
        log_scale_decay: Whether to use log scale for decay plot.
        selected_level_indices: Dict mapping (particle_id, channel) to selected level index.
        use_lifetime_grouping: Whether to use lifetime in grouping analysis.
        global_grouping: Whether to apply grouping globally across all particles.
        export_bin_size_ms: Custom bin size for export (when not syncing with intensity tab).
        export_use_intensity_bin_size: Whether to sync export bin size with intensity tab.
    """

    bin_size_ms: float = 10.0
    confidence: ConfidenceLevel = ConfidenceLevel.CONF_95
    active_tab: ActiveTab = ActiveTab.INTENSITY
    show_levels: bool = True
    show_groups: bool = True
    log_scale_decay: bool = True
    selected_level_indices: Dict[tuple[int, int], int] = field(default_factory=dict)
    use_lifetime_grouping: bool = False
    global_grouping: bool = False
    export_bin_size_ms: float = 10.0
    export_use_intensity_bin_size: bool = True


@dataclass
class ProcessingState:
    """State for background processing.

    Attributes:
        is_busy: Whether a background task is running.
        progress: Progress value from 0.0 to 1.0.
        message: Status message to display.
        current_task: Description of the current task.
    """

    is_busy: bool = False
    progress: float = 0.0
    message: str = ""
    current_task: str = ""

    def start(self, task: str, message: str = "") -> None:
        """Start a new task."""
        self.is_busy = True
        self.progress = 0.0
        self.current_task = task
        self.message = message or f"Starting {task}..."

    def update(self, progress: float, message: str = "") -> None:
        """Update progress."""
        self.progress = max(0.0, min(1.0, progress))
        if message:
            self.message = message

    def finish(self, message: str = "") -> None:
        """Finish the current task."""
        self.is_busy = False
        self.progress = 1.0
        self.message = message or f"Completed {self.current_task}"
        self.current_task = ""


@dataclass
class SessionState:
    """Complete application state container.

    This is the single source of truth for the application. All analysis
    results, UI state, and file data are stored here.

    Attributes:
        file_metadata: Metadata about the loaded file.
        particles: List of all particles from the file.
        selected: List of currently selected particle/channel combinations.
        current_selection: The primary (most recently selected) particle/channel.
        levels: Dict mapping (particle_id, channel) to list of LevelData.
        clustering_results: Dict mapping (particle_id, channel) to ClusteringResult.
        particle_fits: Dict mapping (particle_id, channel) to FitResultData for full decay fits.
        level_fits: Dict mapping (particle_id, channel, level_index) to FitResultData.
        irf_data: Dict mapping (particle_id, channel) to IRFData for particle-level fits.
        level_irf_data: Dict mapping (particle_id, channel, level_index) to IRFData for level fits.
        export_directory: Directory to export data to (defaults to input file's directory).
        ui_state: UI-related state.
        processing: Background processing state.
    """

    file_metadata: Optional[FileMetadata] = None
    particles: List[ParticleData] = field(default_factory=list)
    selected: List[ChannelSelection] = field(default_factory=list)
    current_selection: Optional[ChannelSelection] = None
    levels: Dict[tuple[int, int], List[LevelData]] = field(default_factory=dict)
    clustering_results: Dict[tuple[int, int], ClusteringResult] = field(
        default_factory=dict
    )
    particle_fits: Dict[tuple[int, int], FitResultData] = field(default_factory=dict)
    level_fits: Dict[tuple[int, int, int], FitResultData] = field(default_factory=dict)
    irf_data: Dict[tuple[int, int], IRFData] = field(default_factory=dict)
    level_irf_data: Dict[tuple[int, int, int], IRFData] = field(default_factory=dict)
    export_directory: Optional[Path] = None
    ui_state: UIState = field(default_factory=UIState)
    processing: ProcessingState = field(default_factory=ProcessingState)

    @property
    def has_file(self) -> bool:
        """Whether a file is currently loaded."""
        return self.file_metadata is not None

    @property
    def num_particles(self) -> int:
        """Number of particles in the current session."""
        return len(self.particles)

    @property
    def has_selection(self) -> bool:
        """Whether any particle/channel is selected."""
        return self.current_selection is not None

    def get_particle(self, particle_id: int) -> Optional[ParticleData]:
        """Get a particle by ID.

        Args:
            particle_id: The particle ID to look up.

        Returns:
            The ParticleData if found, None otherwise.
        """
        for particle in self.particles:
            if particle.id == particle_id:
                return particle
        return None

    def get_current_particle(self) -> Optional[ParticleData]:
        """Get the currently selected particle.

        Returns:
            The currently selected ParticleData, or None if no selection.
        """
        if self.current_selection is None:
            return None
        return self.get_particle(self.current_selection.particle_id)

    def get_levels(
        self, particle_id: int, channel: int = 1
    ) -> Optional[List[LevelData]]:
        """Get levels for a particle/channel.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).

        Returns:
            List of LevelData if change point analysis has been run, None otherwise.
        """
        return self.levels.get((particle_id, channel))

    def get_current_levels(self) -> Optional[List[LevelData]]:
        """Get levels for the currently selected particle/channel.

        Returns:
            List of LevelData if available, None otherwise.
        """
        if self.current_selection is None:
            return None
        return self.get_levels(
            self.current_selection.particle_id, self.current_selection.channel
        )

    def set_levels(
        self, particle_id: int, channel: int, levels: List[LevelData]
    ) -> None:
        """Set levels for a particle/channel.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            levels: List of LevelData from change point analysis.
        """
        self.levels[(particle_id, channel)] = levels

    def get_clustering(
        self, particle_id: int, channel: int = 1
    ) -> Optional[ClusteringResult]:
        """Get clustering result for a particle/channel.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).

        Returns:
            ClusteringResult if clustering has been run, None otherwise.
        """
        return self.clustering_results.get((particle_id, channel))

    def get_current_clustering(self) -> Optional[ClusteringResult]:
        """Get clustering result for the currently selected particle/channel.

        Returns:
            ClusteringResult if available, None otherwise.
        """
        if self.current_selection is None:
            return None
        return self.get_clustering(
            self.current_selection.particle_id, self.current_selection.channel
        )

    def set_clustering(
        self, particle_id: int, channel: int, result: ClusteringResult
    ) -> None:
        """Set clustering result for a particle/channel.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            result: The ClusteringResult from AHCA.
        """
        self.clustering_results[(particle_id, channel)] = result

    def get_groups(
        self, particle_id: int, channel: int = 1
    ) -> Optional[List["GroupData"]]:
        """Get groups for a particle/channel.

        This is a convenience method that returns the groups from the
        clustering result.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).

        Returns:
            List of GroupData if clustering has been run, None otherwise.
        """
        result = self.get_clustering(particle_id, channel)
        if result is None:
            return None
        return list(result.groups)

    def get_current_groups(self) -> Optional[List["GroupData"]]:
        """Get groups for the currently selected particle/channel.

        Returns:
            List of GroupData if available, None otherwise.
        """
        if self.current_selection is None:
            return None
        return self.get_groups(
            self.current_selection.particle_id, self.current_selection.channel
        )

    def get_particle_fit(
        self, particle_id: int, channel: int
    ) -> Optional[FitResultData]:
        """Get particle (full decay) fit result.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).

        Returns:
            FitResultData if fitting has been run, None otherwise.
        """
        return self.particle_fits.get((particle_id, channel))

    def set_particle_fit(
        self,
        particle_id: int,
        channel: int,
        result: FitResultData,
    ) -> None:
        """Set particle (full decay) fit result.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            result: The FitResultData from lifetime fitting.
        """
        self.particle_fits[(particle_id, channel)] = result

    def get_level_fit(
        self, particle_id: int, channel: int, level_index: int
    ) -> Optional[FitResultData]:
        """Get level-specific fit result.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            level_index: The level index.

        Returns:
            FitResultData if fitting has been run, None otherwise.
        """
        return self.level_fits.get((particle_id, channel, level_index))

    def set_level_fit(
        self,
        particle_id: int,
        channel: int,
        level_index: int,
        result: FitResultData,
    ) -> None:
        """Set level-specific fit result.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            level_index: The level index.
            result: The FitResultData from lifetime fitting.
        """
        self.level_fits[(particle_id, channel, level_index)] = result

    def get_all_level_fits(
        self, particle_id: int, channel: int
    ) -> Dict[int, FitResultData]:
        """Get all level fits for a particle/channel.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).

        Returns:
            Dict mapping level_index to FitResultData.
        """
        result = {}
        for (pid, ch, level_idx), fit_data in self.level_fits.items():
            if pid == particle_id and ch == channel:
                result[level_idx] = fit_data
        return result

    def get_irf(self, particle_id: int, channel: int) -> Optional[IRFData]:
        """Get IRF data for a particle/channel fit.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).

        Returns:
            IRFData if available, None otherwise.
        """
        return self.irf_data.get((particle_id, channel))

    def set_irf(self, particle_id: int, channel: int, irf: IRFData) -> None:
        """Set IRF data for a particle/channel fit.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            irf: The IRFData to store.
        """
        self.irf_data[(particle_id, channel)] = irf

    def get_level_irf(
        self, particle_id: int, channel: int, level_index: int
    ) -> Optional[IRFData]:
        """Get IRF data for a level-specific fit.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            level_index: The level index.

        Returns:
            IRFData if available, None otherwise.
        """
        return self.level_irf_data.get((particle_id, channel, level_index))

    def set_level_irf(
        self, particle_id: int, channel: int, level_index: int, irf: IRFData
    ) -> None:
        """Set IRF data for a level-specific fit.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            level_index: The level index.
            irf: The IRFData to store.
        """
        self.level_irf_data[(particle_id, channel, level_index)] = irf

    def select(self, particle_id: int, channel: int = 1) -> None:
        """Select a particle/channel as the current selection.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
        """
        selection = ChannelSelection(particle_id=particle_id, channel=channel)
        self.current_selection = selection
        if selection not in self.selected:
            self.selected.append(selection)

    def clear_selection(self) -> None:
        """Clear all selections."""
        self.selected.clear()
        self.current_selection = None

    def get_selected_level_index(
        self, particle_id: int, channel: int = 1
    ) -> Optional[int]:
        """Get the selected level index for a particle/channel.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).

        Returns:
            The selected level index, or None if no level is selected.
        """
        return self.ui_state.selected_level_indices.get((particle_id, channel))

    def set_selected_level_index(
        self, particle_id: int, channel: int, level_index: Optional[int]
    ) -> None:
        """Set the selected level index for a particle/channel.

        Args:
            particle_id: The particle ID.
            channel: The channel number (1 or 2).
            level_index: The level index to select, or None to clear selection.
        """
        if level_index is None:
            self.ui_state.selected_level_indices.pop((particle_id, channel), None)
        else:
            self.ui_state.selected_level_indices[(particle_id, channel)] = level_index

    def get_current_selected_level_index(self) -> Optional[int]:
        """Get the selected level index for the current selection.

        Returns:
            The selected level index for the current particle/channel, or None.
        """
        if self.current_selection is None:
            return None
        return self.get_selected_level_index(
            self.current_selection.particle_id, self.current_selection.channel
        )

    def clear_analysis(self) -> None:
        """Clear all analysis results while keeping particles."""
        self.levels.clear()
        self.clustering_results.clear()
        self.particle_fits.clear()
        self.level_fits.clear()
        self.irf_data.clear()
        self.level_irf_data.clear()

    def reset(self) -> None:
        """Reset the session to initial empty state."""
        self.file_metadata = None
        self.particles.clear()
        self.selected.clear()
        self.current_selection = None
        self.levels.clear()
        self.clustering_results.clear()
        self.particle_fits.clear()
        self.level_fits.clear()
        self.irf_data.clear()
        self.level_irf_data.clear()
        self.ui_state = UIState()
        self.processing = ProcessingState()
