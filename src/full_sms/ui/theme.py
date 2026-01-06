"""DearPyGui theme configuration for Full SMS.

Provides a dark, scientific-appropriate theme for the application.
"""

import dearpygui.dearpygui as dpg

# Application version
APP_VERSION = "0.8.0"

# Color palette for scientific visualization
# Based on a dark theme with good contrast for data analysis
COLORS = {
    # Base colors (RGBA)
    "background": (30, 30, 35, 255),
    "background_child": (35, 35, 40, 255),
    "background_popup": (40, 40, 45, 255),
    "background_frame": (45, 45, 55, 255),
    "background_frame_hover": (55, 55, 65, 255),
    "background_frame_active": (65, 65, 80, 255),
    # Text colors
    "text": (230, 230, 230, 255),
    "text_disabled": (128, 128, 128, 255),
    "text_selected": (255, 255, 255, 255),
    # Accent colors
    "accent": (80, 140, 200, 255),
    "accent_hover": (100, 160, 220, 255),
    "accent_active": (120, 180, 240, 255),
    # Button colors
    "button": (60, 60, 70, 255),
    "button_hover": (75, 75, 90, 255),
    "button_active": (90, 90, 110, 255),
    # Header colors (for collapsibles, tree nodes)
    "header": (50, 50, 60, 255),
    "header_hover": (60, 60, 75, 255),
    "header_active": (70, 70, 90, 255),
    # Tab colors
    "tab": (45, 45, 55, 255),
    "tab_hover": (60, 60, 75, 255),
    "tab_active": (75, 90, 110, 255),
    "tab_unfocused": (40, 40, 50, 255),
    "tab_unfocused_active": (55, 65, 80, 255),
    # Title bar
    "title_bg": (25, 25, 30, 255),
    "title_bg_active": (35, 45, 60, 255),
    "title_bg_collapsed": (25, 25, 30, 255),
    # Menu bar
    "menu_bar_bg": (35, 35, 40, 255),
    # Border colors
    "border": (70, 70, 80, 255),
    "border_shadow": (0, 0, 0, 0),
    # Scrollbar colors
    "scrollbar_bg": (25, 25, 30, 255),
    "scrollbar_grab": (60, 60, 70, 255),
    "scrollbar_grab_hover": (80, 80, 95, 255),
    "scrollbar_grab_active": (100, 100, 120, 255),
    # Slider/check colors
    "slider_grab": (80, 140, 200, 255),
    "slider_grab_active": (100, 160, 220, 255),
    "check_mark": (100, 180, 255, 255),
    # Table colors
    "table_header_bg": (45, 45, 55, 255),
    "table_border_strong": (70, 70, 80, 255),
    "table_border_light": (50, 50, 60, 255),
    "table_row_bg": (0, 0, 0, 0),
    "table_row_bg_alt": (35, 35, 40, 100),
    # Selection
    "text_selected_bg": (60, 100, 150, 200),
    # Plot-specific colors
    "plot_bg": (25, 25, 30, 255),
    "plot_border": (60, 60, 70, 255),
    "plot_legend_bg": (35, 35, 40, 230),
    "plot_legend_border": (60, 60, 70, 255),
    "plot_legend_text": (220, 220, 220, 255),
    "plot_title_text": (230, 230, 230, 255),
    "plot_axis_text": (200, 200, 200, 255),
    "plot_axis_grid": (60, 60, 70, 100),
    "plot_axis_bg": (0, 0, 0, 0),
    "plot_axis_bg_hover": (40, 40, 50, 255),
    "plot_axis_bg_active": (50, 50, 65, 255),
    "plot_crosshairs": (255, 255, 255, 150),
    "plot_selection": (80, 140, 200, 100),
    # Data series colors (for scientific plots)
    "series_1": (100, 180, 255, 255),  # Blue
    "series_2": (255, 150, 100, 255),  # Orange
    "series_3": (100, 220, 150, 255),  # Green
    "series_4": (255, 100, 150, 255),  # Pink
    "series_5": (180, 150, 255, 255),  # Purple
    "series_6": (255, 220, 100, 255),  # Yellow
}


def create_theme() -> int:
    """Create and return the application theme.

    Returns:
        The tag of the created theme.
    """
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            # Window backgrounds
            dpg.add_theme_color(
                dpg.mvThemeCol_WindowBg, COLORS["background"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ChildBg, COLORS["background_child"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PopupBg, COLORS["background_popup"], category=dpg.mvThemeCat_Core
            )

            # Frame backgrounds
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg, COLORS["background_frame"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered,
                COLORS["background_frame_hover"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive,
                COLORS["background_frame_active"],
                category=dpg.mvThemeCat_Core,
            )

            # Text
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, COLORS["text"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TextDisabled,
                COLORS["text_disabled"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TextSelectedBg,
                COLORS["text_selected_bg"],
                category=dpg.mvThemeCat_Core,
            )

            # Borders
            dpg.add_theme_color(
                dpg.mvThemeCol_Border, COLORS["border"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_BorderShadow,
                COLORS["border_shadow"],
                category=dpg.mvThemeCat_Core,
            )

            # Title bar
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBg, COLORS["title_bg"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBgActive,
                COLORS["title_bg_active"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBgCollapsed,
                COLORS["title_bg_collapsed"],
                category=dpg.mvThemeCat_Core,
            )

            # Menu bar
            dpg.add_theme_color(
                dpg.mvThemeCol_MenuBarBg, COLORS["menu_bar_bg"], category=dpg.mvThemeCat_Core
            )

            # Buttons
            dpg.add_theme_color(
                dpg.mvThemeCol_Button, COLORS["button"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered,
                COLORS["button_hover"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive,
                COLORS["button_active"],
                category=dpg.mvThemeCat_Core,
            )

            # Headers (collapsibles, tree nodes)
            dpg.add_theme_color(
                dpg.mvThemeCol_Header, COLORS["header"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderHovered,
                COLORS["header_hover"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderActive,
                COLORS["header_active"],
                category=dpg.mvThemeCat_Core,
            )

            # Tabs
            dpg.add_theme_color(
                dpg.mvThemeCol_Tab, COLORS["tab"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabHovered, COLORS["tab_hover"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabActive, COLORS["tab_active"], category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabUnfocused,
                COLORS["tab_unfocused"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabUnfocusedActive,
                COLORS["tab_unfocused_active"],
                category=dpg.mvThemeCat_Core,
            )

            # Scrollbar
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarBg,
                COLORS["scrollbar_bg"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrab,
                COLORS["scrollbar_grab"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabHovered,
                COLORS["scrollbar_grab_hover"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabActive,
                COLORS["scrollbar_grab_active"],
                category=dpg.mvThemeCat_Core,
            )

            # Sliders and checkmarks
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrab,
                COLORS["slider_grab"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrabActive,
                COLORS["slider_grab_active"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_CheckMark, COLORS["check_mark"], category=dpg.mvThemeCat_Core
            )

            # Tables
            dpg.add_theme_color(
                dpg.mvThemeCol_TableHeaderBg,
                COLORS["table_header_bg"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderStrong,
                COLORS["table_border_strong"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderLight,
                COLORS["table_border_light"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBg,
                COLORS["table_row_bg"],
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBgAlt,
                COLORS["table_row_bg_alt"],
                category=dpg.mvThemeCat_Core,
            )

            # Style settings
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 3, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(
                dpg.mvStyleVar_WindowPadding, 8, 8, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_FramePadding, 6, 4, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_ItemSpacing, 8, 6, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_ItemInnerSpacing, 6, 4, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 14, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 12, category=dpg.mvThemeCat_Core)

    return theme


def create_plot_theme() -> int:
    """Create and return a theme specifically for plots.

    Returns:
        The tag of the created theme.
    """
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBg, COLORS["plot_bg"], category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBorder,
                COLORS["plot_border"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBg,
                COLORS["plot_legend_bg"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBorder,
                COLORS["plot_legend_border"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendText,
                COLORS["plot_legend_text"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_TitleText,
                COLORS["plot_title_text"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisText,
                COLORS["plot_axis_text"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisGrid,
                COLORS["plot_axis_grid"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisBg, COLORS["plot_axis_bg"], category=dpg.mvThemeCat_Plots
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisBgHovered,
                COLORS["plot_axis_bg_hover"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_AxisBgActive,
                COLORS["plot_axis_bg_active"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Crosshairs,
                COLORS["plot_crosshairs"],
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Selection,
                COLORS["plot_selection"],
                category=dpg.mvThemeCat_Plots,
            )

    return theme


def get_series_colors() -> list[tuple[int, int, int, int]]:
    """Get the list of colors for data series.

    Returns:
        List of RGBA color tuples for plotting multiple data series.
    """
    return [
        COLORS["series_1"],
        COLORS["series_2"],
        COLORS["series_3"],
        COLORS["series_4"],
        COLORS["series_5"],
        COLORS["series_6"],
    ]
