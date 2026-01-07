"""Tab views: intensity, lifetime, grouping, spectra, export, etc."""

from full_sms.ui.views.export_tab import ExportTab, EXPORT_TAB_TAGS
from full_sms.ui.views.grouping_tab import GroupingTab, GROUPING_TAB_TAGS
from full_sms.ui.views.intensity_tab import IntensityTab, INTENSITY_TAB_TAGS
from full_sms.ui.views.lifetime_tab import LifetimeTab, LIFETIME_TAB_TAGS
from full_sms.ui.views.spectra_tab import SpectraTab, SPECTRA_TAB_TAGS

__all__ = [
    "ExportTab",
    "EXPORT_TAB_TAGS",
    "GroupingTab",
    "GROUPING_TAB_TAGS",
    "IntensityTab",
    "INTENSITY_TAB_TAGS",
    "LifetimeTab",
    "LIFETIME_TAB_TAGS",
    "SpectraTab",
    "SPECTRA_TAB_TAGS",
]
