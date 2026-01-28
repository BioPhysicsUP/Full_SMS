# Full SMS UI Improvements Plan

This document tracks UI and behavioral improvements for the DearPyGui implementation. Each task is designed to be completable in a single Claude Code session.

---

## How to Use This Document

1. Find the task marked `[NEXT]`
2. Tell Claude: "Continue with the next task in the UI improvements plan"
3. Claude will:
   a. Read the task and research the codebase
   b. **Explain the implementation plan briefly** and ask for approval before coding
   c. Implement the changes after receiving go-ahead
   d. **Ask user to test** the changes (`uv run python -m full_sms.app`)
   e. **After user confirms it works**, update this document:
      - Change `[NEXT]` to `[DONE]` with completion date
      - Add implementation notes
      - Mark the following task as `[NEXT]`
      - Update the Progress Summary table
   f. Commit the changes
4. If testing reveals issues, fix them before updating the plan
5. Repeat until complete

---

## Status Legend

- `[DONE]` - Task completed (date in parentheses)
- `[NEXT]` - Current task to work on
- `[TODO]` - Pending task
- `[SKIP]` - Task skipped (reason noted)
- `[BLOCKED]` - Waiting on something
- `[FEEDBACK]` - Awaiting user feedback after implementation

---

## Phase 1: File Dialogs & Tree View

### Task 1.1: File dialog remembers last location `[DONE]` (2026-01-07)
**Objective**: Improve file dialog UX by remembering the last opened file location

**Actions**:
- Store the last opened file path in the application config
- When opening a file dialog, start at the last opened file's directory
- Ensure double-click on a file opens it (verify current behavior)

**User Testing**: After implementation, ask user to test opening multiple files from different locations

**Verification**: File dialog starts at previously used directory

**Implementation Notes**:
- Added `FileDialogSettings` dataclass in `config.py` with `last_open_directory`, `last_session_directory`, and `last_export_directory` fields
- `FileDialogs` class now loads paths from settings on init and saves them when files are selected
- Settings are persisted to `settings.json` in the platform-specific config location

---

### Task 1.2: Simplify tree view for single-channel files `[DONE]` (2026-01-07)
**Objective**: Remove unnecessary nesting for particles with only one channel

**Actions**:
- Modify particle tree to detect single-channel particles
- Display single-channel particles without channel nesting (show particle directly)
- Multi-channel particles retain the current nested structure

**User Testing**: Ask user to load both single and multi-channel files to verify

**Verification**: Single-channel files show particles without channel sub-items

**Implementation Notes**:
- Modified `_build_particle_node` to check `particle.has_dual_channel`
- Single-channel particles use new `_build_single_channel_item` method that creates a flat checkbox+selectable directly in the tree
- Dual-channel particles retain the existing tree node structure with nested channel items

---

### Task 1.3: Clean up tree view display `[DONE]` (2026-01-07)
**Objective**: Remove photon count and reduce tree width

**Actions**:
- Remove the number of photons display from tree view items
- Reduce the horizontal width of the tree view panel (approximately half current size)
- Ensure particle names are still readable (may need truncation with tooltip)

**User Testing**: Ask user if the tree is now appropriately sized

**Verification**: Tree view is narrower and cleaner without photon counts

**Implementation Notes**:
- Removed photon count from channel labels in `particle_tree.py` (now shows just "Channel 1" instead of "Channel 1 (12,345 photons)")
- Reduced `SIDEBAR_WIDTH` from 280 to 180 pixels in `layout.py`
- Adjusted sidebar button widths from 85 to 78 pixels to fit the narrower panel

---

## Phase 2: Layout & Scrolling Fixes

### Task 2.1: Fix status bar visibility `[DONE]` (2026-01-08)
**Objective**: Status bar should always be visible without scrolling

**Actions**:
- Adjust main layout so status bar is pinned to the bottom
- Ensure content area scrolls independently if needed
- Status bar should remain visible regardless of content size

**User Testing**: Ask user to load a file and check status bar visibility in all tabs

**Verification**: Status bar is always visible at window bottom

**Implementation Notes**:
- Wrapped the main horizontal group (sidebar + content area) in a `child_window` with `height=-STATUS_BAR_HEIGHT` (negative height leaves room at the bottom)
- Removed `autosize_y=True` from sidebar and content area child windows since they're now inside a fixed-height container
- Status bar is placed outside the main content wrapper, ensuring it's always at the bottom regardless of content size

---

### Task 2.2: Prevent horizontal text overflow `[DONE]` (2026-01-08)
**Objective**: Prevent long text from expanding the view horizontally

**Actions**:
- Identify text elements that cause horizontal expansion (info panels, labels)
- Apply text wrapping or truncation with tooltips for long values
- Ensure tab content fits within the window width without horizontal scrolling
- Focus on Intensity tab text (photon count, level count, average intensity)

**User Testing**: Ask user to load data and check if horizontal scrolling is needed

**Verification**: No horizontal scrollbar appears due to text length

**Implementation Notes**:
- Reorganized Intensity tab controls into three rows instead of one long horizontal row
- Row 1: Bin Size slider + label, Show Histogram, Show Levels, Colour by Group (UK spelling), Fit View
- Row 2: Confidence combo + Resolve buttons (Current, Selected, All)
- Row 3: Level info text + data statistics text
- This prevents the long info text from pushing other controls off-screen

---

## Phase 3: Spectra & Raster Tabs

### Task 3.1: Fix Spectra tab display `[DONE]` (2026-01-08)
**Objective**: Ensure Spectra tab shows data when available

**Actions**:
- Debug why spectra tab shows nothing
- Verify spectra data is being loaded from HDF5 files
- Ensure plot is created and data is rendered
- Show informative message when no spectra data exists

**User Testing**: Ask user to test with a file that has spectra data

**Verification**: Spectra plot displays when data is present

**Implementation Notes**:
- Root cause: `_on_selection_changed` in `app.py` was not updating the Spectra tab when a particle was selected
- Added calls to `set_spectra_data()` when particle has spectra, or `set_spectra_unavailable()` when it doesn't
- Added `clear_spectra_data()` call when selection is cleared
- The tab already had proper placeholder messages for "no data loaded" and "no spectra data" states

---

### Task 3.2: Fix Raster tab display `[DONE]` (2026-01-08)
**Objective**: Ensure Raster tab shows data when available

**Actions**:
- Debug why raster tab shows nothing
- Verify raster scan data is being loaded from HDF5 files
- Ensure 2D plot/image is created and rendered
- Show informative message when no raster data exists

**User Testing**: Ask user to test with a file that has raster scan data

**Verification**: Raster image displays when data is present

**Implementation Notes**:
- Same root cause as Task 3.1: `_on_selection_changed` was not updating the Raster tab
- Added calls to `set_raster_data()` when particle has raster scan, or `set_raster_unavailable()` when it doesn't
- Added `clear_raster_data()` call when selection is cleared
- Fixed alongside Task 3.1 since both had the identical issue

---

## Phase 4: Intensity Tab Improvements

### Task 4.1: Link histogram Y-axis to intensity plot `[DONE]` (2026-01-12)
**Objective**: Histogram vertical axis should match intensity plot Y range

**Actions**:
- Link the histogram Y-axis to the intensity plot's Y range
- When user zooms/pans intensity plot, histogram axis updates
- Ensure bi-directional linking if appropriate

**User Testing**: Ask user to zoom the intensity plot and verify histogram updates

**Verification**: Histogram Y-axis stays in sync with intensity plot

**Implementation Notes**:
- Refactored Intensity tab to use `dpg.subplots()` with `link_all_y=True` parameter
- This is DearPyGui's native axis linking feature that automatically synchronizes Y-axes
- Modified `IntensityPlot` and `IntensityHistogram` to support building within subplot context via `for_subplot` parameter
- Column ratios set to [5.0, 1.0] to give intensity plot 5x the width of histogram
- The linking is bidirectional - zooming/panning either plot updates the other

---

### Task 4.2: Fix histogram visibility and layout `[DONE]` (2026-01-12)
**Objective**: Histogram should always be visible without horizontal scrolling

**Actions**:
- Adjust intensity tab layout so histogram is always visible
- Reduce/wrap the text labels that cause overflow (photon count, level count, avg intensity)
- Consider using a fixed-width info panel or collapsible section

**User Testing**: Ask user to verify histogram is visible when switching to Intensity tab

**Verification**: Histogram visible without scrolling, info text doesn't cause overflow

**Implementation Notes**:
- Added `horizontal_scrollbar=False` to plot_container in intensity_tab.py
- Added `horizontal_scrollbar=False` to content_area and tab_intensity containers in layout.py
- Added `wrap=400` to info text element to prevent long statistics text from causing horizontal expansion
- These changes ensure the histogram is always visible and the subplot fills available width properly

---

### Task 4.3: Add user-editable bin size input `[DONE]` (2026-01-12)
**Objective**: Allow user to type in a specific bin size value

**Actions**:
- Replace or augment bin size slider with an input field
- User should be able to type an exact value (e.g., 10ms, 50ms)
- Validate input and update plot on Enter or focus loss
- Keep slider for quick adjustments if space permits

**User Testing**: Ask user to try entering specific bin sizes

**Verification**: User can type bin size values and plot updates

**Implementation Notes**:
- Added `dpg.add_input_float` next to the slider (70px wide)
- Slider format set to empty string to avoid duplicate display
- Bidirectional sync: slider updates input, input updates slider
- Input validates on Enter key press, clamps values to valid range (0.1-1000 ms)
- Slider width reduced from 200px to 150px to accommodate the input field

---

### Task 4.4: Replace level bars with step line `[DONE]` (2026-01-12)
**Objective**: Display resolved levels as a step line instead of filled bars

**Actions**:
- Replace shade series (filled rectangles) with a single line series for levels
- Step line traces intensity levels, stepping up/down at level boundaries
- Simplified group coloring methods (to be reimplemented with horizontal bands later)

**User Testing**: Ask user to resolve levels and verify step line display

**Verification**: Levels display as a red step line

**Implementation Notes**:
- Modified `_render_level_overlays()` in `intensity_plot.py` to build step line points from sorted levels
- Added `level_step_line` tag and `_apply_step_line_theme()` for styling (red, 2px weight)
- Simplified `set_color_by_group()` and `set_highlighted_group()` since they don't apply to the single step line
- Group coloring will be implemented later with horizontal band overlays

---

### Task 4.5: Fix histogram bar width on bin size change `[DONE]` (2026-01-12)
**Objective**: Histogram bars should remain full-width (bar-to-bar) when bin size changes

**Actions**:
- Investigate why bars become narrower when bin size changes
- Ensure bar width is calculated to fill the space between bins
- Bars should be adjacent with no gaps

**User Testing**: Ask user to change bin size and verify bar widths

**Verification**: Histogram bars remain full-width at all bin sizes

**Implementation Notes**:
- Added `_bin_width` field to track computed histogram bin width
- Calculate bin width from `bin_edges[1] - bin_edges[0]` in `_compute_histogram()`
- Apply `weight=self._bin_width` to bar series in `_update_series()`
- Added `no_inputs=True` to histogram plot to disable manual zoom/pan (controlled via intensity plot)
- Set fixed Y-axis limits on histogram (0 to max with 20% padding) for consistent display
- **Additional change**: Added plot legend to intensity plot with "Intensity" and "Levels" series
- **Additional change**: Removed "Show Levels" checkbox - visibility now toggled via legend click
- Level step line now appears as a proper series in the legend

---

### Task 4.6: Replace resolve buttons with combo box `[SKIP]` (2026-01-12)
**Objective**: Simplify resolve controls with a scope selector

**Actions**:
- Replace "Current", "Selected", "All" buttons with a single "Resolve" button
- Add a combo box/dropdown to select scope: Current, Selected, All
- Apply consistent pattern to other tabs that have similar controls

**User Testing**: Ask user if the new control arrangement is intuitive

**Verification**: Single resolve button with scope selector works correctly

**Skipped**: User not convinced this change is needed; may revisit later

---

## Phase 5: Lifetime Tab Improvements

### Task 5.1: Add intensity plot to Lifetime tab `[DONE]` (2026-01-12)
**Objective**: Show intensity plot with clickable level selection

**Actions**:
- Add an intensity trace plot to the Lifetime tab (above or beside decay plot)
- Display levels as colored regions on this plot
- Allow user to click on a level to select it
- When a level is selected, show that level's decay curve

**User Testing**: Ask user to click on different levels and verify decay updates

**Verification**: Clicking levels shows their respective decay curves

**Implementation Notes**:
- Added compact intensity plot (250px height) above decay plot, hidden until levels exist
- Click handler detects clicks on plot and finds level at that time position
- Selected level highlighted with light green vertical band
- Decay histogram updates to show only selected level's microtimes
- "Show All" button returns to full data view
- Legend hidden on compact plot for cleaner display
- Step line and band both extend to next level's start time for visual consistency
- Y-axis fixed to 0-max range
- Decay plot height is dynamic: fills space when no residuals, shrinks when fit is shown

---

### Task 5.2: Reorganize fitting controls `[DONE]` (2026-01-13)
**Objective**: Provide clear fit options with scope selection

**Actions**:
- Add 3 fit buttons: "Fit Particle", "Fit Levels", "Fit Both"
- Add combo box for scope: Current, Selected, All
- Consider moving these controls to the fitting dialog for cleaner UI
- Update fitting workflow to handle level-specific fits

**User Testing**: Ask user to test each fitting option

**Verification**: All fitting options work with scope selection

**Implementation Notes**:
- Added `FitTarget` enum (Particle (full decay), Selected Level, All Levels) and `FitScope` enum (Current, Selected, All) to fitting_dialog.py
- Added Fit Target and Scope combos to the top of the fitting dialog
- Selected Level option only appears when a level is selected in the Lifetime tab
- When Selected Level is chosen, Scope is locked to Current
- Created `FitResultData` class for scalar-only fit result persistence (no large arrays)
- Updated `SessionState` to have separate `particle_fits` and `level_fits` dictionaries
- Updated app.py to submit particle and level-specific fit tasks based on selection
- Updated session save/load to handle the new fit storage structure
- Level fits extract microtimes for just that level's time range

---

### Task 5.3: Fix IRF display behavior `[DONE]` (2026-01-20)
**Objective**: Show IRF button should display IRF (real or simulated)

**Actions**:
- Debug why clicking "Show IRF" unchecks immediately and shows nothing
- If real IRF exists, display it
- If no real IRF but simulated IRF was used in fit, display the simulated IRF
- Show informative message if no IRF available at all

**User Testing**: Ask user to test with files that have/don't have IRF data

**Verification**: Show IRF displays appropriate IRF data

**Implementation Notes**:
- Root cause: IRF was never passed to the fit task and never stored for display
- Modified `_build_fit_task_params` to generate simulated IRF when `use_simulated_irf` is True
- Added `_irf_cache` to store IRF data keyed by (particle_id, channel_id, level_index)
- Modified `_handle_fit_result` to set IRF on lifetime tab after fit completes
- Added IRF restoration in `_update_lifetime_display` and `_on_level_selection_changed` when switching particles/levels
- IRF cache is cleared when opening a new file
- Added full convolved fit curve computation including rising edge:
  - Added `set_raw_irf()` method to store raw IRF for convolution computation
  - Added `_compute_full_convolved_curve()` to compute convolution using fitted parameters
  - Fit curve now shows the full rising edge when IRF data is available
- Fixed IRF display position by applying fitted shift to IRF time axis
- IRF display trimmed to show only significant part (≥0.5 counts) for cleaner log scale visualization

---

### Task 5.4: Fix log scale toggle `[DONE]` (2026-01-23)
**Objective**: Log scale checkbox should properly toggle Y-axis scale

**Actions**:
- Debug why log scale toggle doesn't change the decay plot scale
- Implement proper log/linear scale switching
- Ensure all plot elements (data, fit, IRF) render correctly in both modes

**User Testing**: Ask user to toggle log scale and verify plot updates

**Verification**: Log scale toggle switches between log and linear Y-axis

**Implementation Notes**:
- The `_rebuild_y_axis()` method in `decay_plot.py` was already correctly implemented
- Y-axis is deleted and recreated with updated `log_scale` parameter when toggled
- All series (data, fit, IRF) are recreated and properly styled after rebuild
- Log scale transformation handles zero counts by replacing with 0.5 for visualization
- User confirmed the functionality is working as expected

---

### Task 5.5: Enlarge fitting dialog input boxes `[DONE]` (2026-01-23)
**Objective**: Make input fields in fitting dialog larger and easier to use

**Actions**:
- Increase width of input fields in fitting dialog
- Ensure values are fully visible without truncation
- Apply consistent sizing across all dialog inputs

**User Testing**: Ask user if dialog inputs are now appropriately sized

**Verification**: All fitting dialog inputs are readable and usable

**Implementation Notes**:
- Increased dialog dimensions to 550px width × 600px height (from 500×580)
- Widened all input fields for better readability:
  - Tau init fields: 80px → 100px
  - Tau bounds: 70px → 110px
  - Shift init: 70px → 100px
  - Shift bounds: 70px → 110px
  - Simulated IRF FWHM: 70px → 100px
  - FWHM bounds: 60px → 80px
  - Start/End channel: 80px → 100px
- Widened combo boxes:
  - Fit target: 180px → 220px
  - Start mode: 150px → 180px
- Widened buttons: 100px → 120px to prevent text cutoff
- Updated dialog centering calculation for new dimensions

---

## Phase 6: Grouping Tab Improvements

### Task 6.1: Add intensity plot to Grouping tab `[DONE]` (2026-01-23)
**Objective**: Show intensity plot with group visualization

**Actions**:
- Add intensity trace plot to Grouping tab
- After grouping, display horizontal bands in different colors for each group
- Bands should span the time range of levels within each group

**User Testing**: Ask user to run grouping and verify band visualization

**Verification**: Group bands display on intensity plot after clustering

**Implementation Notes**:
- Added compact intensity plot (400px height) above BIC plot, shown after levels are resolved
- Group visualization uses alternating horizontal bands with semi-transparent fill
- Dashed lines mark each group's average intensity level
- Band bounds calculated as midpoints between adjacent group intensities
- BIC plot improved: added scatter markers at all data points, Y-axis padding for visibility, legend moved to bottom-right
- Group list panel widened (240→310px) and column labels shortened for better fit
- Group bands update automatically when user changes the selected number of groups

---

### Task 6.2: Interactive BIC plot selection `[DONE]` (2026-01-24)
**Objective**: Replace slider with clickable BIC plot points

**Actions**:
- Add markers for each "number of groups" point on BIC curve
- Markers should be same color as the line
- Keep distinct markers for optimal and current selection
- When user clicks a point, change selection to that number of groups
- Remove the slider control

**User Testing**: Ask user to click different points on BIC plot

**Verification**: Clicking BIC points changes the selected grouping solution

**Implementation Notes**:
- BIC plot already had markers for all points, optimal (green), and selected (orange)
- Implemented proper mouse click handler using DearPyGui handler registry (plot callback doesn't work for clicks)
- Kept the slider control (user requested both methods for changing selection)
- Added session state persistence: selected group count is now saved/restored when navigating between particles
- Added `_on_group_count_changed` callback in app.py to update session state when selection changes
- Fixed intensity plot not showing immediately after grouping by explicitly setting intensity data in `_update_grouping_display`
- Fixed clearing of group bands/lines when switching between particles

---

## Phase 7: Export Tab Improvements

### Task 7.1: Fix export options layout `[DONE]` (2026-01-26)
**Objective**: All checkboxes visible without scrolling

**Actions**:
- Reorganize export tab layout to fit all options in visible area
- Use columns or compact arrangement for checkboxes
- Ensure all data and plot export options are visible

**User Testing**: Ask user if all export options are visible

**Verification**: No scrolling needed to see all export checkboxes

**Implementation Notes**:
- Increased child window dimensions to 420×480px (left) and 400×480px (right)
- All export options now visible without scrolling
- Added plot customization sub-options:
  - Intensity plot: Include Levels (functional), Include Groups (disabled for now)
  - Decay plot: Include Fit, Include IRF (both functional but currently not exported - see Task 7.5)
- Renamed "Intensity Trace" to "Intensity Plot" for consistency
- Renamed "Fit Results" to "Lifetime Fit Parameters" for clarity
- Consolidated levels export to include group_id and lifetime fit parameters in one file
- Smart column export: only includes tau_2/tau_3 columns if multi-exponential fits exist
- Updated group_id assignment: levels now get group_id populated after clustering completes

---

### Task 7.2: Add plot customization options `[DONE]` (2026-01-26)
**Objective**: Allow user to specify what's included in exported plots

**Actions**:
- Add checkboxes for Intensity plot: Include levels? Include groups?
- Add checkboxes for Lifetime plot: Include fit? Include IRF?
- Add placeholder options for Spectra and Raster (when implemented)
- Apply selections when exporting

**User Testing**: Ask user to test export with different options

**Verification**: Exported plots include/exclude elements based on selections

**Implementation Notes**:
- Combined with Task 7.1 implementation
- Intensity plot now uses correct style (red step line for levels, horizontal bands for groups)
- Matches visual style from Intensity tab and Grouping tab
- Decay plot customization UI exists but fit/IRF not yet exported (see Task 7.5)

---

### Task 7.3: Improve export input fields `[DONE]` (2026-01-26)
**Objective**: Wider DPI and bin size inputs with intensity tab link option

**Actions**:
- Increase width of DPI and bin size input fields
- Add checkbox: "Use bin size from Intensity tab"
- When checked, auto-populate bin size from intensity tab setting
- Ensure inputs accept reasonable value ranges

**User Testing**: Ask user to test the linked bin size option

**Verification**: Input fields are wider, bin size linking works

**Implementation Notes**:
- Increased Plot DPI input width from 80px to 120px
- Increased Bin Size input width from 80px to 120px
- Added "Use bin size from Intensity tab" checkbox (checked by default)
- Bin size input is greyed out when checkbox is checked
- Real-time sync: when Intensity tab bin size changes, export tab updates automatically
- Settings are stateful and persist in session:
  - `export_bin_size_ms`: Custom bin size value
  - `export_use_intensity_bin_size`: Whether to sync with intensity tab
- Values saved/restored correctly when saving/loading sessions

---

### Task 7.4: Default export directory to input file location `[DONE]` (2026-01-26)
**Objective**: Start export directory at input file's location

**Actions**:
- When opening export directory picker, start at the loaded file's directory
- If no file loaded, use last export directory or home
- Store last export directory in config

**User Testing**: Ask user to verify export directory default behavior

**Verification**: Export directory defaults to input file location

**Implementation Notes**:
- Removed `last_export_directory` from global settings (FileDialogSettings)
- Added `export_directory: Optional[Path]` field to SessionState
- Export tab now uses smart default logic in `_get_default_export_directory()`:
  1. Uses current `_output_dir` if already set
  2. Uses session state's `export_directory` if saved
  3. Uses input file's parent directory if file is loaded
  4. Falls back to Desktop only if none of the above apply
- Export directory is saved to session state when user browses or exports
- Export directory is persisted in session files and restored on load
- Removed unused export directory dialog methods from file_dialogs.py

---

### Task 7.5: Export decay plots with fit curves and IRF `[DONE]` (2026-01-27)
**Objective**: Include fitted curves and IRF in exported decay plots

**Actions**:
- Recompute fit curve from stored `FitResultData` parameters for export
- Reference existing recalculation code from `decay_plot.py::_compute_full_convolved_curve()` (lines 747-800+)
- This method already recomputes the convolved fit curve from parameters:
  - Takes tau values, amplitudes, shift, and background
  - Builds multi-exponential model
  - Applies IRF convolution using stored IRF data
  - Returns full curve for entire time range
- Adapt this logic to `plot_exporters.py::export_decay_plot()`
- Store/retrieve IRF data from session state for recomputation
- Respect the "Include Fit" and "Include IRF" checkboxes
- Ensure residuals can be recomputed from fit parameters and data

**User Testing**: Ask user to export decay plots with fitted data

**Verification**: Exported decay plots show fitted curves and IRF when selected

**Implementation Notes**:
- Created shared `compute_convolved_fit_curve()` function in `analysis/lifetime.py`
  - Takes FitResultData parameters + decay histogram + IRF (either FWHM or array)
  - Returns (fitted_curve, residuals) arrays
  - Used by both UI (decay_plot.py) and export (plot_exporters.py) for consistency
- Added `IRFData` dataclass in `models/fit.py`:
  - For simulated IRF: stores only `fwhm_ns` (efficient storage)
  - For loaded IRF: stores full `time_ns` and `counts` arrays
  - Includes `to_dict()`/`from_dict()` for JSON serialization
- Added IRF storage to `SessionState`:
  - `irf_data` dict for particle-level fits
  - `level_irf_data` dict for level-specific fits
  - Getter/setter methods for easy access
- Updated `decay_plot.py` to use shared function instead of duplicate code
- Updated `plot_exporters.py::export_decay_plot()`:
  - Now accepts `FitResultData` + `IRFData` parameters
  - Computes fit curve using shared function when FitResultData provided
  - Displays IRF (simulated regenerated from FWHM, loaded from stored array)
- Updated `app.py` to store IRFData when fit completes:
  - Tracks IRF config (simulated, FWHM) when fit task is submitted
  - Stores appropriate IRFData after fit completion
- Updated session persistence (`io/session.py`):
  - Saves/loads `irf_data` and `level_irf_data` to/from JSON sessions
- Updated `export_tab.py` to retrieve IRFData from session and pass to export

---

## Progress Summary

| Phase | Tasks | Completed | Skipped | Remaining |
|-------|-------|-----------|---------|-----------|
| 1. File Dialogs & Tree | 3 | 3 | 0 | 0 |
| 2. Layout & Scrolling | 2 | 2 | 0 | 0 |
| 3. Spectra & Raster | 2 | 2 | 0 | 0 |
| 4. Intensity Tab | 6 | 5 | 1 | 0 |
| 5. Lifetime Tab | 5 | 5 | 0 | 0 |
| 6. Grouping Tab | 2 | 2 | 0 | 0 |
| 7. Export Tab | 5 | 5 | 0 | 0 |
| **Total** | **25** | **24** | **1** | **0** |

---

## Notes

- Each task should be tested by the user before moving to the next
- Feedback from testing may result in follow-up tasks being added
- Tasks can be split if they prove too large for a single session
- Related tasks in the same area are grouped to minimize context switching
- The user should be asked to run the app and test after each task

---

*Created: 2026-01-07*
*Last Updated: 2026-01-27 (Task 7.5 completed - All tasks complete!)*
