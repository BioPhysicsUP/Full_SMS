"""Session save and load functionality for Full SMS.

Sessions are saved as JSON files containing:
- Reference to the source HDF5 file
- Analysis results (levels, clustering, fits)
- UI state

The raw photon data is NOT stored - it's reloaded from the HDF5 file.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from full_sms.models.fit import FitResult
from full_sms.models.group import ClusteringResult, GroupData
from full_sms.models.level import LevelData
from full_sms.models.session import (
    ActiveTab,
    ChannelSelection,
    ConfidenceLevel,
    FileMetadata,
    SessionState,
    UIState,
)

# Current session format version
SESSION_VERSION = "1.0"


class SessionSerializationError(Exception):
    """Error during session serialization or deserialization."""

    pass


def _tuple_key_to_str(key: tuple) -> str:
    """Convert a tuple key to a string for JSON storage."""
    return ",".join(str(k) for k in key)


def _str_to_tuple_key(key: str, types: Tuple[type, ...]) -> tuple:
    """Convert a string key back to a tuple with the specified types."""
    parts = key.split(",")
    if len(parts) != len(types):
        raise SessionSerializationError(
            f"Key '{key}' has {len(parts)} parts, expected {len(types)}"
        )
    return tuple(t(p) for t, p in zip(types, parts))


def _level_to_dict(level: LevelData) -> Dict[str, Any]:
    """Convert a LevelData to a serializable dict."""
    return {
        "start_index": level.start_index,
        "end_index": level.end_index,
        "start_time_ns": level.start_time_ns,
        "end_time_ns": level.end_time_ns,
        "num_photons": level.num_photons,
        "intensity_cps": level.intensity_cps,
        "group_id": level.group_id,
    }


def _dict_to_level(data: Dict[str, Any]) -> LevelData:
    """Convert a dict back to a LevelData."""
    return LevelData(
        start_index=data["start_index"],
        end_index=data["end_index"],
        start_time_ns=data["start_time_ns"],
        end_time_ns=data["end_time_ns"],
        num_photons=data["num_photons"],
        intensity_cps=data["intensity_cps"],
        group_id=data.get("group_id"),
    )


def _group_to_dict(group: GroupData) -> Dict[str, Any]:
    """Convert a GroupData to a serializable dict."""
    return {
        "group_id": group.group_id,
        "level_indices": list(group.level_indices),
        "total_photons": group.total_photons,
        "total_dwell_time_s": group.total_dwell_time_s,
        "intensity_cps": group.intensity_cps,
    }


def _dict_to_group(data: Dict[str, Any]) -> GroupData:
    """Convert a dict back to a GroupData."""
    return GroupData(
        group_id=data["group_id"],
        level_indices=tuple(data["level_indices"]),
        total_photons=data["total_photons"],
        total_dwell_time_s=data["total_dwell_time_s"],
        intensity_cps=data["intensity_cps"],
    )


def _clustering_to_dict(result: ClusteringResult) -> Dict[str, Any]:
    """Convert a ClusteringResult to a serializable dict."""
    return {
        "groups": [_group_to_dict(g) for g in result.groups],
        "all_bic_values": list(result.all_bic_values),
        "optimal_step_index": result.optimal_step_index,
        "selected_step_index": result.selected_step_index,
        "num_original_levels": result.num_original_levels,
    }


def _dict_to_clustering(data: Dict[str, Any]) -> ClusteringResult:
    """Convert a dict back to a ClusteringResult."""
    return ClusteringResult(
        groups=tuple(_dict_to_group(g) for g in data["groups"]),
        all_bic_values=tuple(data["all_bic_values"]),
        optimal_step_index=data["optimal_step_index"],
        selected_step_index=data["selected_step_index"],
        num_original_levels=data["num_original_levels"],
    )


def _fit_result_to_dict(result: FitResult) -> Dict[str, Any]:
    """Convert a FitResult to a serializable dict."""
    return {
        "tau": list(result.tau),
        "tau_std": list(result.tau_std),
        "amplitude": list(result.amplitude),
        "amplitude_std": list(result.amplitude_std),
        "shift": result.shift,
        "shift_std": result.shift_std,
        "chi_squared": result.chi_squared,
        "durbin_watson": result.durbin_watson,
        "dw_bounds": list(result.dw_bounds) if result.dw_bounds else None,
        "residuals": result.residuals.tolist(),
        "fitted_curve": result.fitted_curve.tolist(),
        "fit_start_index": result.fit_start_index,
        "fit_end_index": result.fit_end_index,
        "background": result.background,
        "num_exponentials": result.num_exponentials,
        "average_lifetime": result.average_lifetime,
    }


def _dict_to_fit_result(data: Dict[str, Any]) -> FitResult:
    """Convert a dict back to a FitResult."""
    return FitResult(
        tau=tuple(data["tau"]),
        tau_std=tuple(data["tau_std"]),
        amplitude=tuple(data["amplitude"]),
        amplitude_std=tuple(data["amplitude_std"]),
        shift=data["shift"],
        shift_std=data["shift_std"],
        chi_squared=data["chi_squared"],
        durbin_watson=data["durbin_watson"],
        dw_bounds=tuple(data["dw_bounds"]) if data["dw_bounds"] else None,
        residuals=np.array(data["residuals"], dtype=np.float64),
        fitted_curve=np.array(data["fitted_curve"], dtype=np.float64),
        fit_start_index=data["fit_start_index"],
        fit_end_index=data["fit_end_index"],
        background=data["background"],
        num_exponentials=data["num_exponentials"],
        average_lifetime=data["average_lifetime"],
    )


def _ui_state_to_dict(ui: UIState) -> Dict[str, Any]:
    """Convert UIState to a serializable dict."""
    return {
        "bin_size_ms": ui.bin_size_ms,
        "confidence": ui.confidence.value,
        "active_tab": ui.active_tab.value,
        "show_levels": ui.show_levels,
        "show_groups": ui.show_groups,
        "log_scale_decay": ui.log_scale_decay,
    }


def _dict_to_ui_state(data: Dict[str, Any]) -> UIState:
    """Convert a dict back to UIState."""
    return UIState(
        bin_size_ms=data["bin_size_ms"],
        confidence=ConfidenceLevel(data["confidence"]),
        active_tab=ActiveTab(data["active_tab"]),
        show_levels=data["show_levels"],
        show_groups=data["show_groups"],
        log_scale_decay=data["log_scale_decay"],
    )


def _selection_to_dict(sel: ChannelSelection) -> Dict[str, Any]:
    """Convert a ChannelSelection to a serializable dict."""
    return {"particle_id": sel.particle_id, "channel": sel.channel}


def _dict_to_selection(data: Dict[str, Any]) -> ChannelSelection:
    """Convert a dict back to a ChannelSelection."""
    return ChannelSelection(particle_id=data["particle_id"], channel=data["channel"])


def save_session(state: SessionState, path: Path) -> None:
    """Save a session state to a JSON file.

    The session file stores:
    - Reference to the source HDF5 file (not the raw data)
    - Analysis results (levels, clustering, fit results)
    - UI state and selections

    Args:
        state: The SessionState to save.
        path: Path to write the session file.

    Raises:
        SessionSerializationError: If the state cannot be serialized.
        IOError: If the file cannot be written.
    """
    if state.file_metadata is None:
        raise SessionSerializationError("Cannot save session without loaded file")

    # Build the session data structure
    session_data: Dict[str, Any] = {
        "version": SESSION_VERSION,
        "file_metadata": {
            "path": str(state.file_metadata.path),
            "filename": state.file_metadata.filename,
            "num_particles": state.file_metadata.num_particles,
            "has_irf": state.file_metadata.has_irf,
            "has_spectra": state.file_metadata.has_spectra,
            "has_raster": state.file_metadata.has_raster,
        },
    }

    # Serialize levels (keyed by "particle_id,channel")
    levels_dict: Dict[str, List[Dict[str, Any]]] = {}
    for (pid, ch), level_list in state.levels.items():
        key = _tuple_key_to_str((pid, ch))
        levels_dict[key] = [_level_to_dict(lvl) for lvl in level_list]
    session_data["levels"] = levels_dict

    # Serialize clustering results
    clustering_dict: Dict[str, Dict[str, Any]] = {}
    for (pid, ch), result in state.clustering_results.items():
        key = _tuple_key_to_str((pid, ch))
        clustering_dict[key] = _clustering_to_dict(result)
    session_data["clustering_results"] = clustering_dict

    # Serialize fit results (keyed by "particle_id,channel,level_or_group_id")
    fit_dict: Dict[str, Dict[str, Any]] = {}
    for (pid, ch, lvl_id), result in state.fit_results.items():
        key = _tuple_key_to_str((pid, ch, lvl_id))
        fit_dict[key] = _fit_result_to_dict(result)
    session_data["fit_results"] = fit_dict

    # Serialize selections
    session_data["selected"] = [_selection_to_dict(s) for s in state.selected]
    session_data["current_selection"] = (
        _selection_to_dict(state.current_selection)
        if state.current_selection
        else None
    )

    # Serialize UI state
    session_data["ui_state"] = _ui_state_to_dict(state.ui_state)

    # Write to file
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)


def load_session(path: Path) -> Dict[str, Any]:
    """Load a session file and return the parsed data.

    This function returns a dictionary with the session data. The caller
    is responsible for:
    1. Reloading the HDF5 file referenced in file_metadata["path"]
    2. Applying the analysis results to the SessionState

    The returned dictionary contains:
    - "version": Session format version string
    - "file_metadata": Dict with path, filename, and feature flags
    - "levels": Dict mapping (particle_id, channel) to list of LevelData
    - "clustering_results": Dict mapping (particle_id, channel) to ClusteringResult
    - "fit_results": Dict mapping (particle_id, channel, level_id) to FitResult
    - "selected": List of ChannelSelection
    - "current_selection": Optional ChannelSelection
    - "ui_state": UIState

    Args:
        path: Path to the session file.

    Returns:
        Dictionary with parsed session data.

    Raises:
        SessionSerializationError: If the file format is invalid.
        FileNotFoundError: If the session file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Version check
    version = data.get("version", "unknown")
    if version != SESSION_VERSION:
        # For now, only support current version
        # Future: add migration logic here
        raise SessionSerializationError(
            f"Unsupported session version: {version} (expected {SESSION_VERSION})"
        )

    # Validate required fields
    if "file_metadata" not in data:
        raise SessionSerializationError("Session file missing file_metadata")

    result: Dict[str, Any] = {
        "version": version,
        "file_metadata": {
            "path": Path(data["file_metadata"]["path"]),
            "filename": data["file_metadata"]["filename"],
            "num_particles": data["file_metadata"]["num_particles"],
            "has_irf": data["file_metadata"].get("has_irf", False),
            "has_spectra": data["file_metadata"].get("has_spectra", False),
            "has_raster": data["file_metadata"].get("has_raster", False),
        },
    }

    # Deserialize levels
    levels: Dict[Tuple[int, int], List[LevelData]] = {}
    for key, level_list in data.get("levels", {}).items():
        pid, ch = _str_to_tuple_key(key, (int, int))
        levels[(pid, ch)] = [_dict_to_level(d) for d in level_list]
    result["levels"] = levels

    # Deserialize clustering results
    clustering: Dict[Tuple[int, int], ClusteringResult] = {}
    for key, clust_data in data.get("clustering_results", {}).items():
        pid, ch = _str_to_tuple_key(key, (int, int))
        clustering[(pid, ch)] = _dict_to_clustering(clust_data)
    result["clustering_results"] = clustering

    # Deserialize fit results
    fits: Dict[Tuple[int, int, int], FitResult] = {}
    for key, fit_data in data.get("fit_results", {}).items():
        pid, ch, lvl_id = _str_to_tuple_key(key, (int, int, int))
        fits[(pid, ch, lvl_id)] = _dict_to_fit_result(fit_data)
    result["fit_results"] = fits

    # Deserialize selections
    result["selected"] = [
        _dict_to_selection(s) for s in data.get("selected", [])
    ]
    result["current_selection"] = (
        _dict_to_selection(data["current_selection"])
        if data.get("current_selection")
        else None
    )

    # Deserialize UI state
    if "ui_state" in data:
        result["ui_state"] = _dict_to_ui_state(data["ui_state"])
    else:
        result["ui_state"] = UIState()

    return result


def apply_session_to_state(
    session_data: Dict[str, Any], state: SessionState
) -> None:
    """Apply loaded session data to a SessionState.

    This should be called after loading the HDF5 file referenced in the session.
    It applies the analysis results and UI state from the session.

    Args:
        session_data: Data returned from load_session().
        state: SessionState to update (should already have particles loaded).
    """
    # Apply levels
    state.levels.clear()
    state.levels.update(session_data["levels"])

    # Apply clustering results
    state.clustering_results.clear()
    state.clustering_results.update(session_data["clustering_results"])

    # Apply fit results
    state.fit_results.clear()
    state.fit_results.update(session_data["fit_results"])

    # Apply selections
    state.selected = session_data["selected"]
    state.current_selection = session_data["current_selection"]

    # Apply UI state
    state.ui_state = session_data["ui_state"]
